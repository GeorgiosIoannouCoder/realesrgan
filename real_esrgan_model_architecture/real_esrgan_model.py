#################################################################################################
# Filename: real_esrgan_model.py
# Description: The archiarchitecture of the official REAL-ESRGAN model as defined in the paper
#              "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"
#################################################################################################
#
# Import libraries.
#
# NumPy library for numerical operations.
import numpy as np

# Random library for generating random numbers.
import random

# PyTorch library for deep learning.
import torch

# Import specific degradation-related functions from the Basicsr library.
# These functions are used for adding random Gaussian and Poisson noise to images.
from basicsr.data.degradations import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)

# Import a specific image transformation function for paired random cropping.
from basicsr.data.transforms import paired_random_crop

# Import the SRGANModel class from the Basicsr library.
# SRGANModel serves as the superclass for the RealEsrGanModel class defined in this script.
from basicsr.models.srgan_model import SRGANModel

# Import utility functions and classes from the Basicsr library.
# DiffJPEG and USMSharp are used for specific image processing tasks.
from basicsr.utils import DiffJPEG, USMSharp

# Import additional image processing utility functions.
# 'filter2D' is used for applying a 2D filter to an image.
from basicsr.utils.img_process_util import filter2D

# Import the MODEL_REGISTRY from the Basicsr library for model registration.
from basicsr.utils.registry import MODEL_REGISTRY

# Import the OrderedDict class from the collections module.
from collections import OrderedDict

# Import specific functions and classes for deep learning operations from PyTorch.
from torch.nn import functional as F


#################################################################################################
# Register the RealEsrGanModel class in the model registry.
@MODEL_REGISTRY.register()
class RealEsrGanModel(SRGANModel):
    def __init__(self, opt):
        super(RealEsrGanModel, self).__init__(opt)

        # Create an instance of the DiffJPEG class with differentiability set to False and move it to the GPU.
        self.jpeger = DiffJPEG(differentiable=False).cuda()

        # Create an instance of the USMSharp class and move it to the GPU.
        self.usm_sharpener = USMSharp().cuda()

        # Set the queue size from options or use a default value.
        self.queue_size = opt.get("queue_size", 180)

    @torch.no_grad()
    # Increase the degradation diversity in a batch.
    def _dequeue_and_enqueue(self):
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, "queue_lr"):
            # Check if queue size is divisible by batch size.
            assert (
                self.queue_size % b == 0
            ), f"queue size {self.queue_size} should be divisible by batch size {b}!"
            # Initialize queues for LR (Low Resolution) images.
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()

            # Initialize queues for HR (High Resolution) images.
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0

        # Check if the queue is full.
        if self.queue_ptr == self.queue_size:
            # Generate a random permutation of indices for shuffling.
            idx = torch.randperm(self.queue_size)

            # Shuffle the LR queue based on the generated indices.
            self.queue_lr = self.queue_lr[idx]
            # Shuffle the HR queue based on the generated indices.
            self.queue_gt = self.queue_gt[idx]

            # Clone the first 'b' samples from the shuffled LR queue.
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            # Clone the first 'b' samples from the shuffled HR queue.
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()

            # Update the first 'b' samples in the LR queue with the current LR images.
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            # Update the first 'b' samples in the HR queue with the current LR images.
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            # Update the current LR Images with the dequeued LR samples.
            self.lq = lq_dequeue
            # Update the current HR Images with the dequeued LR samples.
            self.gt = gt_dequeue
        # If the queue is not full, only perform enqueue operation.
        else:
            # Enqueue the current LR images starting from the current queue pointer.
            self.queue_lr[
                self.queue_ptr : self.queue_ptr + b, :, :, :
            ] = self.lq.clone()
            # Enqueue the current HR images starting from the current queue pointer.
            self.queue_gt[
                self.queue_ptr : self.queue_ptr + b, :, :, :
            ] = self.gt.clone()
            # Update the queue pointer to the next available position.
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        # Creating LR images coming from the data loader during the training process.
        # Create LR images from two degradation proceses.

        # Check if the model is in training mode and high-order degradation is enabled.
        if self.is_train and self.opt.get("high_order_degradation", True):
            # Training data synthesis.
            # Load ground truth image and apply unsharp mask sharpening.
            self.gt = data["gt"].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            # Load degradation kernels and sinc kernel from the input data.
            self.kernel1 = data["kernel1"].to(self.device)
            self.kernel2 = data["kernel2"].to(self.device)
            self.sinc_kernel = data["sinc_kernel"].to(self.device)

            # Get the original height and width of the ground truth image.
            ori_h, ori_w = self.gt.size()[2:4]

            # First Degradation Process to obtain LQ images during training.
            # See line 26 in the file real_esrgan_x4.yml.
            #
            # Apply blur using filter2D with the first kernel.
            out = filter2D(self.gt_usm, self.kernel1)

            # Apply random resize with specified probabilities and ranges.
            updown_type = random.choices(
                ["up", "down", "keep"], self.opt["resize_prob"]
            )[0]
            if updown_type == "up":
                scale = np.random.uniform(1, self.opt["resize_range"][1])
            elif updown_type == "down":
                scale = np.random.uniform(self.opt["resize_range"][0], 1)
            else:
                scale = 1

            # Randomly select one of the interpolation modes ('area', 'bilinear', or 'bicubic').
            mode = random.choice(["area", "bilinear", "bicubic"])

            # Resize the tensor out based on the chosen interpolation mode and scaling factor.
            out = F.interpolate(out, scale_factor=scale, mode=mode)

            # Add noise (Gaussian or Poisson) based on specified probabilities and ranges.
            gray_noise_prob = self.opt["gray_noise_prob"]
            if np.random.uniform() < self.opt["gaussian_noise_prob"]:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt["noise_range"],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt["poisson_scale_range"],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )

            # Apply JPEG compression.
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range"])
            out = torch.clamp(
                out, 0, 1
            )  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # Second Degradation Process to obtain LQ images during training.
            # See line 43 in the file real_esrgan_x4.yml.
            #
            # Apply blur using filter2D with the first kernel.
            if np.random.uniform() < self.opt["second_blur_prob"]:
                out = filter2D(out, self.kernel2)

            # Apply random resize with specified probabilities and ranges.
            updown_type = random.choices(
                ["up", "down", "keep"], self.opt["resize_prob2"]
            )[0]
            if updown_type == "up":
                scale = np.random.uniform(1, self.opt["resize_range2"][1])
            elif updown_type == "down":
                scale = np.random.uniform(self.opt["resize_range2"][0], 1)
            else:
                scale = 1

            # Randomly select one of the interpolation modes ('area', 'bilinear', or 'bicubic').
            mode = random.choice(["area", "bilinear", "bicubic"])
            # Resize the tensor out based on the chosen interpolation mode and scaling factor.
            out = F.interpolate(
                out,
                size=(
                    int(ori_h / self.opt["scale"] * scale),
                    int(ori_w / self.opt["scale"] * scale),
                ),
                mode=mode,
            )

            # Add noise (Gaussian or Poisson) based on specified probabilities and ranges.
            gray_noise_prob = self.opt["gray_noise_prob2"]
            if np.random.uniform() < self.opt["gaussian_noise_prob2"]:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt["noise_range2"],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt["poisson_scale_range2"],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )

            # Apply JPEG compression and the final sinc filter.
            # Resize images to desired sizes. Two orders are considered:
            #   1. [Resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [Resize back + sinc filter]
            if np.random.uniform() < 0.5:
                # Randomly select one of the interpolation modes ('area', 'bilinear', or 'bicubic').
                mode = random.choice(["area", "bilinear", "bicubic"])
                # Resize the tensor out based on the chosen interpolation mode and scaling factor.
                out = F.interpolate(
                    out,
                    size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                    mode=mode,
                )
                # Apply a 2D convolution to the tensor out using the sinc kernel self.sinc_kernel.
                # Apply various filters to the image.
                out = filter2D(out, self.sinc_kernel)

                # JPEG compression.
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range2"])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression.
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range2"])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)

                # Randomly select one of the interpolation modes ('area', 'bilinear', or 'bicubic').
                mode = random.choice(["area", "bilinear", "bicubic"])
                # Resize the tensor out based on the chosen interpolation mode and scaling factor.
                out = F.interpolate(
                    out,
                    size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                    mode=mode,
                )
                # Apply a 2D convolution to the tensor out using the sinc kernel self.sinc_kernel.
                # Apply various filters to the image.
                out = filter2D(out, self.sinc_kernel)

            # Clamp and round the resulting low-quality image.
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

            # Random cropping both the ground truth and low-quality images.
            gt_size = self.opt["gt_size"]
            (self.gt, self.gt_usm), self.lq = paired_random_crop(
                [self.gt, self.gt_usm], self.lq, gt_size, self.opt["scale"]
            )

            # Training pair pool maintenance.
            self._dequeue_and_enqueue()
            # Resharpen the ground truth image (self.gt)
            # because it has been modified during the _dequeue_and_enqueue operation.
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()
        else:
            # For paired training or validation.
            self.lq = data["lq"].to(self.device)
            if "gt" in data:
                self.gt = data["gt"].to(self.device)
                # USM sharpen the GT images.
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # Perform non-distributed validation for the RealEsrGanModel model.
        # During validation, set the model to not use the synthetic process,
        # then call the parent class's nondist_validation method.

        # Do not use the synthetic process during validation.
        self.is_train = False
        super(RealEsrGanModel, self).nondist_validation(
            dataloader, current_iter, tb_logger, save_img
        )

        # Set the model back to training mode after validation.
        self.is_train = True

    def optimize_parameters(self, current_iter):
        # Optimize the parameters of the RealEsrGanModel model during the training.

        # Select the appropriate ground truth for different losses based on sharpening options.
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt["l1_gt_usm"] is False:
            l1_gt = self.gt
        if self.opt["percep_gt_usm"] is False:
            percep_gt = self.gt
        if self.opt["gan_gt_usm"] is False:
            gan_gt = self.gt

        # Freeze discriminator parameters during generator optimization.
        for p in self.net_d.parameters():
            p.requires_grad = False

        # Zero the gradients and perform generator optimization.
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        # Total generator loss
        l_g_total = 0

        loss_dict = OrderedDict()

        # Only perform generator optimization when conditions are met.
        if (
            current_iter % self.net_d_iters == 0
            and current_iter > self.net_d_init_iters
        ):
            # Pixel Loss.
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict["l_g_pix"] = l_g_pix

            # Perceptual Loss.
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict["l_g_percep"] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict["l_g_style"] = l_g_style

            # GAN Loss.
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict["l_g_gan"] = l_g_gan

            # Generator backward propagation.
            l_g_total.backward()

            # Perform generator optimization step.
            self.optimizer_g.step()

        # Unfreeze discriminator parameters for discriminator optimization.
        for p in self.net_d.parameters():
            p.requires_grad = True

        # Zero the gradients and perform discriminator optimization.
        self.optimizer_d.zero_grad()

        # Real.
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict["l_d_real"] = l_d_real
        loss_dict["out_d_real"] = torch.mean(real_d_pred.detach())
        # Real discriminator backward propagation.
        l_d_real.backward()

        # Fake.
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict["l_d_fake"] = l_d_fake
        loss_dict["out_d_fake"] = torch.mean(fake_d_pred.detach())
        # Fake discriminator backward propagation.
        l_d_fake.backward()

        # Perform discriminator optimization step.
        self.optimizer_d.step()

        # Exponential Moving Average (EMA) for the generator
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # Reduce and store the loss dictionary.
        self.log_dict = self.reduce_loss_dict(loss_dict)


#################################################################################################

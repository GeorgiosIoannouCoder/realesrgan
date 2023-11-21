####################################################################################################
# Filename: data_used_during_training.py
# Description: REAL-ESRGAN was trained on pure synthetic data.
#              Therefore, the data needs to be augmented during the training.
#              This file is used during the training to:
#                 1. Load GT (Ground Truth) images.
#                 2. Augment the laoded GT images.
#                 3. Generate blur kernels.
#                 4. Sinc kernels for generating LQ (Low Quality) images.
####################################################################################################
#
# Import libraries.
#
# Import the OpenCV library for image processing.
import cv2

# Import the math library for mathematical operations.
import math

# NumPy library for numerical operations.
import numpy as np

# Operating System library for file and path operations.
import os
import os.path as osp

# Random library for generating random numbers.
import random

# Time library for time-related functions.
import time

# PyTorch library for deep learning.
import torch

# Import functions and classes from Basicsr for data degradation.
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels

# Import image augmentation functions from Basicsr.
from basicsr.data.transforms import augment

# Import utility functions from Basicsr.
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor

# Import the DATASET_REGISTRY from Basicsr for dataset registration.
from basicsr.utils.registry import DATASET_REGISTRY

# Import data-related functions from PyTorch.
from torch.utils import data as data


####################################################################################################
# Register the RealEsrGanDataset class in the dataset registry.
@DATASET_REGISTRY.register()
class RealEsrGanDataset(data.Dataset):
    def __init__(self, opt):
        super(RealEsrGanDataset, self).__init__()
        # Store the options dictionary for the dataset.
        self.opt = opt
        self.file_client = None
        # Extract IO backend options from the provided options.
        self.io_backend_opt = opt["io_backend"]
        # Get the path to the directory containing GT (Ground Truth) images.
        self.gt_folder = opt["dataroot_gt"]

        # Check if the IO backend type is lmdb.
        if self.io_backend_opt["type"] == "lmdb":
            # Set the LMDB database path and client keys for GT images.
            self.io_backend_opt["db_paths"] = [self.gt_folder]
            self.io_backend_opt["client_keys"] = ["gt"]

            # Check if the GT folder path ends with '.lmdb'.
            if not self.gt_folder.endswith(".lmdb"):
                raise ValueError(
                    f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}"
                )

            # Read image paths from the 'meta_info.txt' file inside the LMDB folder.
            with open(osp.join(self.gt_folder, "meta_info.txt")) as fin:
                self.paths = [line.split(".")[0] for line in fin]
        else:
            # For disk backend with meta_info.
            # Read image paths from the 'meta_info' file specified in options.
            with open(self.opt["meta_info"]) as fin:
                paths = [line.strip().split(" ")[0] for line in fin]
                self.paths = [os.path.join(self.gt_folder, v) for v in paths]

        # Blur settings for the first Degradation Process.
        #
        # Set the size of the blur kernel used in the first degradation process.
        self.blur_kernel_size = opt["blur_kernel_size"]
        # Specify the list of possible kernels for degradation (e.g., Gaussian, motion blur).
        self.kernel_list = opt["kernel_list"]
        # Probability distribution for choosing each kernel during the first degradation.
        self.kernel_prob = opt["kernel_prob"]
        # Standard deviation of the Gaussian blur kernel used in the first degradation.
        self.blur_sigma = opt["blur_sigma"]
        # Range of parameters for the generalized Gaussian blur kernel's betag during the first degradation.
        self.betag_range = opt["betag_range"]
        # Range of parameters for the plateau blur kernel's betap during the first degradation.
        self.betap_range = opt["betap_range"]
        # Probability of applying sinc filters during the first degradation.
        self.sinc_prob = opt["sinc_prob"]

        # Blur settings for the first Degradation Process.
        self.blur_kernel_size2 = opt["blur_kernel_size2"]
        self.kernel_list2 = opt["kernel_list2"]
        self.kernel_prob2 = opt["kernel_prob2"]
        self.blur_sigma2 = opt["blur_sigma2"]
        self.betag_range2 = opt["betag_range2"]
        self.betap_range2 = opt["betap_range2"]
        self.sinc_prob2 = opt["sinc_prob2"]

        # A final sinc filter.
        self.final_sinc_prob = opt["final_sinc_prob"]

        # Kernel size range for degradation (ranges from 7 to 21).
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]

        # Pulse tensor to be used for convolutions for a no blurry effect.
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        # Define the function to get an item from the dataset based on the given index.

        # If file client is not initialized, create a new FileClient for lmdb IO backend.
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        # Load ground truth (gt) image.
        gt_path = self.paths[index]

        # How many times to request the file to be loaded in case there is an IO error.
        retry = 1

        while retry > 0:
            try:
                # Read the image bytes from the file client.
                img_bytes = self.file_client.get(gt_path, "gt")
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(
                    f"File client error: {e}, remaining retry times: {retry - 1}"
                )
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
            else:
                break
            finally:
                retry -= 1

        # Convert image bytes to a float32 image.
        img_gt = imfrombytes(img_bytes, float32=True)

        # Perform data augmentation for training: horizontal flip and rotation.
        img_gt = augment(img_gt, self.opt["use_hflip"], self.opt["use_rot"])

        # Crop or pad the image to a size of 400x400.
        h, w = img_gt.shape[0:2]
        crop_pad_size = 400

        # Pad the image if its size is smaller than the desired size.
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(
                img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )

        # Crop the image if its size is larger than the desired size.
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            # Get the height and width of the image.
            h, w = img_gt.shape[0:2]
            # Randomly choose the top coordinate for cropping.
            top = random.randint(0, h - crop_pad_size)
            # Randomly choose the left coordinate for cropping.
            left = random.randint(0, w - crop_pad_size)
            # Crop the image.
            img_gt = img_gt[top : top + crop_pad_size, left : left + crop_pad_size, ...]

        # Generate kernels for the First Degradation Process.
        #
        # Randomly select a kernel size from the specified range.
        kernel_size = random.choice(self.kernel_range)

        # Check if a sinc kernel should be used for the second degradation.
        if np.random.uniform() < self.opt["sinc_prob"]:
            if kernel_size < 13:
                # Randomly choose an angular cutoff frequency for small kernels.
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                # Randomly choose an angular cutoff frequency for large kernels.
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            # Generate a circular lowpass sinc kernel.
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            # Randomly generate mixed kernels.
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )

        # Pad the kernel to a size of 21x21.
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # Generate kernels for the Second Degradation Process.
        #
        # Randomly select a kernel size from the specified range.
        kernel_size = random.choice(self.kernel_range)

        # Check if a sinc kernel should be used for the second degradation.
        if np.random.uniform() < self.opt["sinc_prob2"]:
            if kernel_size < 13:
                # Randomly choose an angular cutoff frequency for small kernels.
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                # Randomly choose an angular cutoff frequency for large kernels.
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            # Generate a circular lowpass sinc kernel.
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            # Randomly generate mixed kernels.
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )

        # Pad the kernel to a size of 21x21.
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # Generate the final sinc kernel.
        if np.random.uniform() < self.opt["final_sinc_prob"]:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # Convert BGR to RGB, HWC to CHW, and numpy to tensor.
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]

        # Convert the kernel array to a PyTorch FloatTensor.
        kernel = torch.FloatTensor(kernel)

        # Convert the kernel2 array to a PyTorch FloatTensor.
        kernel2 = torch.FloatTensor(kernel2)

        # Return a dictionary containing:
        # 1. GT (Ground Truth) image.
        # 2. Kernels
        # 3. Sinc kernel.
        # 4. Path to the GT image.
        return_d = {
            "gt": img_gt,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "gt_path": gt_path,
        }
        return return_d

    def __len__(self):
        # Return the total number of items in the dataset.
        return len(self.paths)


####################################################################################################

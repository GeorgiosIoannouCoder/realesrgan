###########################################################################################
# Filename: realsrgan.py
# Description: Upscale images using the trained REALESRGAN model.
###########################################################################################
#
# Import libraries.
#
# Import OpenCV library for image processing.
import cv2

# Import the math module for mathematical operations.
import math

# Import NumPy for numerical operations on arrays.
import numpy as np

# Import the os module for operating system functionalities.
import os

# Import the queue module for implementing queues.
import queue

# Import the threading module for multi-threading support.
import threading

# Import PyTorch for deep learning.
import torch

# Import a utility function for downloading files.
from basicsr.utils.download_util import load_file_from_url

# Import functional module from PyTorch's neural network library.
from torch.nn import functional as F

###########################################################################################
# Define the root directory.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


###########################################################################################
class RealEsrGan:
    def __init__(
        self,
        scale,  # Upsampling scale factor used in the networks.
        model_path,  # The path to the pretrained model.
        dni_weight=None,  # Performing the interpolation between two networks.
        model=None,  # The pretained model weights.
        pre_pad=10,  # Pad the input images to avoid border artifacts.
        half=False,  # Whether to use half precision during inference or not.
        device=None,  # What device to run inference on. cpu or cuda.
        gpu_id=None,  # ID of GPU to be used if there are more than one GPUs.
    ):
        self.scale = scale
        self.model_path = model_path
        self.dni_weight = dni_weight
        self.model = model
        self.pre_pad = pre_pad
        self.half = half
        self.device = device
        self.gpu_id = gpu_id

        self.mod_scale = None

        # Initialize device based on GPU availability and user preference.
        if self.gpu_id:
            self.device = (
                torch.device(
                    f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
                )
                if self.device is None
                else self.device
            )
        else:
            self.device = (
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self.device is None
                else self.device
            )

        # Load the RealESRGAN model from the specified path or URL.
        if isinstance(self.model_path, list):
            assert len(self.model_path) == len(self.dni_weight)
            loadnet = self.dni(self.model_path[0], self.model_path[1], self.dni_weight)
        else:
            # Download model if model path is a URL.
            if self.model_path.startswith("https://"):
                self.model_path = load_file_from_url(
                    url=model_path,
                    model_dir=os.path.join(ROOT_DIR, "weights"),
                    progress=True,
                    file_name=None,
                )
            loadnet = torch.load(model_path, map_location=torch.device("cpu"))

        # Use params_ema if available, otherwise use params.
        if "params_ema" in loadnet:
            keyname = "params_ema"
        else:
            keyname = "params"

        # Load model weights.
        model.load_state_dict(loadnet[keyname], strict=True)

        # Put the model in evaluation mode.
        model.eval()

        # Move the model to the specified device.
        self.model = model.to(self.device)

        if self.half:
            self.model = self.model.half()

    def dni(self, net_a, net_b, dni_weight, key="params", loc="cpu"):
        # Define a method for Domain-Adversarial Neural Interface (DNI).

        # Load the parameters of neural network A from a file, considering the specified device location.
        net_a = torch.load(net_a, map_location=torch.device(loc))

        # Load the parameters of neural network B from a file, considering the specified device location.
        net_b = torch.load(net_b, map_location=torch.device(loc))

        # Iterate over each key-value pair in the parameters of neural network A.
        for k, v_a in net_a[key].items():
            # Update the parameters of neural network A using a weighted combination
            # of its own parameters and those of neural network B.
            net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]

        # Return the updated model.
        return net_a

    def pre_process(self, img):
        # Convert image to PyTorch tensor and adjust dimensions.
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

        # Add a batch dimension and move the tensor to the specified device.
        self.img = img.unsqueeze(0).to(self.device)

        # If half precision is enabled, convert the tensor to half precision.
        if self.half:
            self.img = self.img.half()

        # Apply reflective padding to the image if pre_pad is not zero.
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), "reflect")

        # Set mod_scale based on the scale factor.
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4

        # Check if mod_scale is specified and perform padding accordingly.
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()

            # Calculate padding required to make dimensions divisible by mod_scale.
            if h % self.mod_scale != 0:
                self.mod_pad_h = self.mod_scale - h % self.mod_scale

            if w % self.mod_scale != 0:
                self.mod_pad_w = self.mod_scale - w % self.mod_scale

            # Apply reflective padding to the image based on mod_pad_h and mod_pad_w.
            self.img = F.pad(
                self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect"
            )

    def process(self):
        # Process/inference on the image.
        self.output = self.model(self.img)

    def post_process(self):
        # Check if a modification scale is specified.
        if self.mod_scale is not None:
            # Get the height and width of the output tensor.
            _, _, h, w = self.output.size()

            # Crop the output tensor based on the specified modification scale and padding
            self.output = self.output[
                :,
                :,
                0 : h - self.mod_pad_h * self.scale,
                0 : w - self.mod_pad_w * self.scale,
            ]

        # Check if there is pre-padding applied.
        if self.pre_pad != 0:
            # Get the height and width of the output tensor.
            _, _, h, w = self.output.size()

            # Crop the output tensor based on the specified pre-padding.
            self.output = self.output[
                :,
                :,
                0 : h - self.pre_pad * self.scale,
                0 : w - self.pre_pad * self.scale,
            ]

        # Return the processed output tensor after modification and cropping.
        return self.output

    def enhance(self, img, upscale=None, alpha_upsampler="realesrgan"):
        # Get the height and width of the input image.
        h_input, w_input = img.shape[0:2]
        img = img.astype(np.float32)

        # Determine if the input image is 16-bit.
        if np.max(img) > 256:
            max_range = 65535
            print("\tInput is a 16-bit image")
        else:
            max_range = 255

        # Normalize the image to the range [0, 1].
        img = img / max_range

        # Identify the image mode based on its number of channels.
        if len(img.shape) == 2:
            img_mode = "L"  # Gray image.
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = "RGBA"  # RGBA image with alpha channel.
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert alpha channel to RGB if using realesrgan alpha upsampling.
            if alpha_upsampler == "realesrgan":
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = "RGB"  # RGB image.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pre-process the image using a method not provided in the code.
        self.pre_process(img)

        # Process the image.
        self.process()

        # Post-process the image and retrieve the enhanced output.
        output_img = self.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

        # Convert output image back to grayscale if the original image was grayscale.
        if img_mode == "L":
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # Process alpha channel if the original image had RGBA mode.
        if img_mode == "RGBA":
            # Check if RealESRGAN should be used for alpha channel upsampling.
            if alpha_upsampler == "realesrgan":
                # Pre-process the alpha channel using a method not provided in this code.
                self.pre_process(alpha)

                # Process the image.
                self.process()

                # Post-process the alpha channel and retrieve the enhanced output.
                output_alpha = self.post_process()

                # Convert the alpha channel output to a NumPy array in the range [0, 1].
                output_alpha = (
                    output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                )

                # Transpose the alpha channel array for proper channel ordering.
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))

                # Convert the alpha channel to grayscale.
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:
                # Resize the alpha channel using linear interpolation if not using realesrgan.
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(
                    alpha,
                    (w * self.scale, h * self.scale),
                    interpolation=cv2.INTER_LINEAR,
                )

            # Convert output image to BGRA format and assign the processed alpha channel.
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # Scale the output image back to the original size if specified.
        if max_range == 65535:
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        # Resize the output image if a different scale is specified.
        if upscale is not None and upscale != float(self.scale):
            output = cv2.resize(
                output,
                (
                    int(w_input * upscale),
                    int(h_input * upscale),
                ),
                interpolation=cv2.INTER_LANCZOS4,
            )

        return output, img_mode


###########################################################################################

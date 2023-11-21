#########################################################################################
# Filename: inference.py
# Description: Do inference on the REALESRGAN model.
#########################################################################################
#
# Import libraries.
#
# Import the module for parsing command-line arguments.
import argparse

# OpenCV library for computer vision.
import cv2

# Import the module for retrieving file paths matching specified patterns.
import glob

# Operating system module for interacting with the file system.
import os

# RRDBNet architecture from the basicsr library.
from basicsr.archs.rrdbnet_arch import RRDBNet

# Utility function for downloading files from URLs.
from basicsr.utils.download_util import load_file_from_url

# RealESRGANer class for REALESRGAN inference.
from real_esrgan import RealEsrGan


#########################################################################################
def main():
    # Create an ArgumentParser object for parsing command-line arguments.
    parser = argparse.ArgumentParser()

    # Define a command-line argument for the input image or folder.
    # Default value is "ccny.jpg".
    parser.add_argument("-input", type=str, default="ccny.jpg")

    # Define a command-line argument for the model to be used for inference.
    # Models: 1. REALESRGAN_x4.
    #         3. REALESRNet_x4.
    # Default value is "REALESRGAN_x4".
    parser.add_argument(
        "-model_name",
        type=str,
        default="REALESRGAN_x4",
    )

    # Define a command-line argument for the output folder to sotre the results.
    # Default value is "inferences".
    parser.add_argument("-output", type=str, default="inferences")

    # Define a command-line argument for the final upsampling scale of the image.
    # Increase it for better image resolution when zooming in.
    # Default value is "4".
    parser.add_argument("-upscale", type=float, default=4)

    # Define a command-line argument for the model path.
    # Default value is None.
    parser.add_argument("-model_path", type=str, default=None)

    # Define a command-line argument for the image extension that will be used
    # to store the super resolution image (the resulting image).
    # Default value is auto meaning using the same extension as the input image.
    # values can be auto, or jpg, or png.
    parser.add_argument("-extension", type=str, default="auto")

    # Define a command-line argument for the gpu device to use.
    # Default value is None meaning no GPU to be used for inference.
    # values can range from 0 up to n-1 if n GPUs are available.
    parser.add_argument("-gpu-id", type=int, default=None)

    # Parse the command-line arguments and store the values in the "args" object.
    args = parser.parse_args()

    # Selecting the model to be used for inference.
    args.model_name = args.model_name.split(".")[0]

    if args.model_name == "REALSRGAN_x4":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        scale = 4
        model_path = "./models/REALESRGAN_x4.pth"
    elif args.model_name == "REALESRNET_x4":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        scale = 4
        model_path = "./models/REALESRNET_x4.pth"

    # Selecting the model path. If the user provides a model path,
    # then it will overwrite the model_name command-line argument which
    # has a default value of REALESRGAN_x4.
    if args.model_path is not None:
        model_path = args.model_path
    else:
        # If model_path is not provided, construct it based on the model_name and check its existence.
        model_path = os.path.join("models", args.model_name + ".pth")
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated with the downloaded file location.
                model_path = load_file_from_url(
                    url=url,
                    model_dir=os.path.join(ROOT_DIR, "models"),
                    progress=True,
                    file_name=None,
                )

    # Instantiate the RealEsrGan class to upsample images for face enhancement.
    upsampler = RealEsrGan(
        scale=scale,
        model_path=model_path,
        dni_weight=None,
        model=model,
        pre_pad=10,
        half=False,
        device=None,
        gpu_id=args.gpu_id,
    )

    # Create the output folder if it does not exist.
    os.makedirs(args.output, exist_ok=True)

    # Process each input image.
    #
    # Check if the input is a single file.
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        # Get a list of all files in the input directory.
        paths = sorted(glob.glob(os.path.join(args.input, "*")))

    # Loop through each input image.
    for idx, path in enumerate(paths):
        # Extract the filename and extension.
        imgname, extension = os.path.splitext(os.path.basename(path))

        # Read the input image using OpenCV.
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # Determine if the image has an alpha channel (RGBA).
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = "RGBA"
        else:
            img_mode = None

        try:
            output, _ = upsampler.enhance(img, upscale=args.upscale)
            # Determine the file extension for saving the output image.
            #
            # All RGBA images should be saved in png format
            # because the png format supports the storage of the alpha channel data.
            # Saving RGBA images in PNG format, the transparency information is preserved,
            # allowing for a more accurate representation of the original image.
            if img_mode == "RGBA":
                extension = "png"
            else:
                if args.extension == "auto":
                    extension = extension[1:]
                else:
                    extension = args.extension

            # Save the super resolution (resulting) image.
            save_path = os.path.join(
                args.output, "{}_inference.{}".format(imgname, extension)
            )
            # save_path = os.path.join(args.output, f'{imgname}.{extension}')
            cv2.imwrite(save_path, output)

            print("Image ", imgname, ".", extension, " is ready!")
            print("Saved at directory: ", args.output, "/")
        except Exception as e:
            print("Error:", e)


#########################################################################################
if __name__ == "__main__":
    main()
#########################################################################################

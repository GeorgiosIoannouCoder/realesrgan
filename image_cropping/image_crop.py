#########################################################################################
# Filename: image_crop.py
# Description: Crop large images to sub-images for faster IO using multithreading.
#########################################################################################
#
# Import libraries.
#
# Module for parsing command-line arguments.
import argparse

# OpenCV library for image processing.
import cv2

# NumPy library for numerical operations.
import numpy as np

# Operating system library for interacting with the file system.
import os

# Module providing access to some variables used or maintained by the Python interpreter.
import sys

# Custom utility function for scanning directories.
from basicsr.utils import scandir

# Module for parallel processing.
from multiprocessing import Pool

# Module for handling file paths.
from os import path as osp

# Module for displaying progress bars.
from tqdm import tqdm


#########################################################################################
def main(args):
    # Initialize an empty dictionary to store the user options from the command line.
    opt = {}

    # Set options based on command-line arguments.
    opt["n_thread"] = args.n_thread
    opt["compression_level"] = args.compression_level
    opt["input_folder"] = args.input
    opt["save_folder"] = args.output
    opt["crop_size"] = args.crop_size
    opt["step"] = args.step
    opt["thresh_size"] = args.thresh_size

    # Call the function to extract subimages based on the provided user options.
    extract_subimages(opt)


#########################################################################################
def extract_subimages(opt):
    # Extract input and output folder paths from the configuration dictionary options.
    # opt is the dictionary storing the user options from the command line.
    input_folder = opt["input_folder"]
    save_folder = opt["save_folder"]

    # Check if the output folder already exists and create it if not.
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f"mkdir {save_folder} ...")
    else:
        # If the output folder already exists, print a message and exit the script.
        print(f"Folder {save_folder} already exists! Exit!")
        sys.exit(1)

    # Get a list of all image files in the input folder.
    img_list = list(scandir(input_folder, full_path=True))

    # Initialize a progress bar with the total number of images to process using tqdm.
    pbar = tqdm(total=len(img_list), unit="image", desc="Extract")

    # Create a Pool for parallel processing with the specified number of threads.
    pool = Pool(opt["n_thread"])

    # Iterate over the list of image paths and apply the "worker" function asynchronously.
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))

    # Close the pool to indicate that no more tasks will be submitted.
    pool.close()

    # Block the program until all tasks in the pool have completed.
    pool.join()

    # Close the progress bar from tqdm.
    pbar.close()

    # Notify the user that all processes are done.
    print("All processes done!")


#########################################################################################
def worker(path, opt):
    # Extract relevant options from the configuration dictionary opt.
    # opt is the dictionary storing the user options from the command line.
    crop_size = opt["crop_size"]
    step = opt["step"]
    thresh_size = opt["thresh_size"]

    # Extract the name and extension of the image file.
    img_name, extension = osp.splitext(osp.basename(path))

    # Remove scaling factors (x2, x3, x4, x8) if there are any from the image name.
    img_name = (
        img_name.replace("x2", "").replace("x3", "").replace("x4", "").replace("x8", "")
    )

    # Read the image using OpenCV in unchanged mode to retain the alpha channel.
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Get the height and width of the image.
    h, w = img.shape[0:2]

    # Generate ranges for sliding window along the height and width.
    h_space = np.arange(0, h - crop_size + 1, step)

    # Check if there"s remaining space after the last window along the height.
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)

    # Check if there"s remaining space after the last window along the width.
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    # Initialize an index for subimage naming.
    index = 0

    # Iterate over the sliding windows along the height and width.
    for x in h_space:
        for y in w_space:
            index += 1
            # Crop the subimage based on the current window position.
            cropped_img = img[x : x + crop_size, y : y + crop_size, ...]
            # Ensure contiguous memory layout for the subimage.
            cropped_img = np.ascontiguousarray(cropped_img)
            # Save the subimage with a formatted name to the specified save folder.
            cv2.imwrite(
                osp.join(opt["save_folder"], f"{img_name}_s{index:03d}{extension}"),
                cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt["compression_level"]],
            )

    # Tracking the progress and notifying the user.
    process_info = f"Processing {img_name}."
    return process_info


#########################################################################################
if __name__ == "__main__":
    # Create an ArgumentParser object for parsing command-line arguments.
    parser = argparse.ArgumentParser()

    # Define a command-line argument for the input folder.
    # Default value is "datasets/DF2K/DF2K_HR".
    parser.add_argument("--input", type=str, default="datasets/DF2K/DF2K_HR_multiscale")

    # Define a command-line argument for the output folder.
    # Default value is "datasets/DF2K/DF2K_HR_sub".
    parser.add_argument(
        "--output", type=str, default="datasets/DF2K/DF2K_HR_multiscale_subimages"
    )

    # Define a command-line argument for the crop size.
    # Default value is 480.
    parser.add_argument("--crop_size", type=int, default=480)

    # Define a command-line argument for the step in the overlapped sliding window.
    # Default value is 240.
    parser.add_argument("--step", type=int, default=240)

    # Define a command-line argument for the threshold size.
    # Patches whose size is lower than thresh_size will be dropped.
    # Default value is 0.
    parser.add_argument(
        "--thresh_size",
        type=int,
        default=0,
    )

    # Define a command-line argument for the number of threads.
    # Default value is 20.
    parser.add_argument("--n_thread", type=int, default=20)

    # Define a command-line argument for the compression level.
    # Default value is 3.
    parser.add_argument("--compression_level", type=int, default=3)

    # Parse the command-line arguments and store the values in the "args" object.
    args = parser.parse_args()

    # Call the "main" function with the parsed command-line arguments.
    main(args)
#########################################################################################

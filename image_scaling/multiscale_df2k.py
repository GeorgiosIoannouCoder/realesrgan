#########################################################################################
# Filename: multiscale_df2k.py
# Description: Scale up and dwon images.
#########################################################################################
#
# Import libraries.
#
# Module for parsing command-line arguments.
import argparse

# Module for retrieving file paths matching specified patterns.
import glob

# Operating system module for interacting with the file system.
import os

# Python Imaging Library for image processing.
from PIL import Image

#########################################################################################


def main(args):
    # Scales to be used for the DF2K dataset.
    scale_list = [0.75, 0.5, 1 / 3]
    # Define the shortest edge for the smallest image.
    shortest_edge = 400

    # Get a sorted list of all files in the input folder.
    path_list = sorted(glob.glob(os.path.join(args.input, "*")))

    # Iterate over each file in the input folder.
    for path in path_list:
        # Print the current file path.
        print(path)

        # Extract the basename of the file (without extension).
        basename = os.path.splitext(os.path.basename(path))[0]

        # Open the image using the Python Imaging Library (PIL).
        img = Image.open(path)
        width, height = img.size

        # Iterate over the predefined scale list and resize the image accordingly.
        for idx, scale in enumerate(scale_list):
            # Print the scale value.
            print(f"\t{scale:.2f}")
            # Resize the image using Lanczos resampling.
            rlt = img.resize(
                (int(width * scale), int(height * scale)), resample=Image.LANCZOS
            )
            # Save the resized image with a modified filename.
            rlt.save(os.path.join(args.output, f"{basename}T{idx}.png"))

        # Save the smallest image whose shortest edge is 400.
        if width < height:
            ratio = height / width
            width = shortest_edge
            height = int(width * ratio)
        else:
            ratio = width / height
            height = shortest_edge
            width = int(height * ratio)

        # Resize the image and save it with a modified filename.
        rlt = img.resize((int(width), int(height)), resample=Image.LANCZOS)
        rlt.save(os.path.join(args.output, f"{basename}T{idx+1}.png"))


#########################################################################################

if __name__ == "__main__":
    # Create an ArgumentParser object for parsing command-line arguments.
    parser = argparse.ArgumentParser()

    # Define a command-line argument for the input folder.
    # Default value is "datasets/DF2K/DF2K_HR".
    parser.add_argument("--input", type=str, default="datasets/DF2K/DF2K_HR")

    # Define a command-line argument for the output folder.
    # Default value is "datasets/DF2K/DF2K_multiscale".
    parser.add_argument(
        "--output", type=str, default="datasets/DF2K/DF2K_HR_multiscale"
    )
    args = parser.parse_args()

    # Create the output folder if it doesn't exist.
    os.makedirs(args.output, exist_ok=True)

    # Call the main function with the parsed command-line arguments.
    main(args)
#########################################################################################

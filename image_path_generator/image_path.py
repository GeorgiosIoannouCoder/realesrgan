# Filename: information.py
# Description: Generate all the GT image paths to be used during training.
#              GT = Ground-Truth
#########################################################################################
#
# Import libraries.
#
# Import the module for parsing command-line arguments.
import argparse

# OpenCV library for image processing.
import cv2

# Import the glob module for file path pattern matching.
import glob

# Import the os module for interacting with the operating system.
import os


#########################################################################################
def main(args):
    # Open a text file in write mode to store all the image paths.
    txt_file = open(args.path, "w")

    for folder, root in zip(args.input, args.root):
        img_paths = sorted(glob.glob(os.path.join(folder, "*")))
        for img_path in img_paths:
            status = True
            if args.check:
                # read the image once for check, as some images may have errors
                try:
                    img = cv2.imread(img_path)
                except (IOError, OSError) as error:
                    print(f"Read {img_path} error: {error}")
                    status = False
                if img is None:
                    status = False
                    print(f"Img is None: {img_path}")
            if status:
                # get the relative path
                img_name = os.path.relpath(img_path, root)
                txt_file.write(f"../{img_name}\n")


#########################################################################################
if __name__ == "__main__":
    # Create an ArgumentParser object for parsing command-line arguments.
    parser = argparse.ArgumentParser()

    # Define a command-line argument for the input folder(s).
    # Default value is ["datasets/DF2K/DF2K_multiscale_subimages",
    # "datasets/OST/ANIMAL",
    # "datasets/OST/BUILDING"]
    # value can be a single folder or a list of folders.
    parser.add_argument(
        "-input",
        nargs="+",
        default=[
            "datasets/DF2K/DF2K_HR_multiscale_subimages",
            "datasets/OST/ANIMAL",
            "datasets/OST/BUILDING",
        ],
    )

    # Define a command-line argument for the root folder(s).
    # Default value is ["","", ""]
    # value can be a single folder or a list of folders.
    parser.add_argument(
        "-root",
        nargs="+",
        default=["", "", ""],
    )

    # Define a command-line argument for the path to store the .txt file at.
    # Default value is "datasets/gt_image_paths.txt".
    parser.add_argument(
        "-path",
        type=str,
        default="./image_path_generator/gt_image_paths.txt",
    )

    # Define a command-line argument for checking if the image is valid for training.
    # Default value is False.
    # To set it to True --check must be presented as an argument in the command.
    parser.add_argument(
        "-check",
        action="store_true",
    )

    # Parse the command-line arguments and store the values in the "args" object.
    args = parser.parse_args()

    # Assert that the input folder and folder root have the same length.
    assert len(args.input) == len(args.root), (
        "Input folder and folder root should have the same length, but got "
        f"{len(args.input)} and {len(args.root)}."
    )
    # Create the text file and populate it with paths to the GT images.
    os.makedirs(os.path.dirname(args.path), exist_ok=True)

    # Call the main function with the parsed command-line arguments.
    main(args)
#########################################################################################

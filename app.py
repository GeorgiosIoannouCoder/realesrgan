##############################################################################################################
# Filename: app.py
# Description: A Streamlit application to test our implemnetation of the model,
# as descirbed in the paper "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"
##############################################################################################################
#
# Import libraries.
#
import cv2
import numpy as np
import requests
import streamlit as st

from basicsr.archs.rrdbnet_arch import RRDBNet
from inference.real_esrgan import RealEsrGan
from io import BytesIO
from PIL import Image

##############################################################################################################


# Function to run inference using the RealEsrGan model.
def run_inference(
    uploaded_file,
    model_name="REALESRGAN_x4",
    output_path="inferences",
    upscale=4,
    extension="auto",
    gpu_id=None,
):
    try:
        # Create an RRDBNet model instance.
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=upscale,
        )

        # Set default model path based on the selected model name
        if model_name == None:
            model_path = "./models/REALESRGAN_x4.pth"
        elif model_name == "REALESRGAN_x4":
            model_path = "./models/REALESRGAN_x4.pth"
        elif model_name == "REALESRNET_x4":
            model_path = "./models/REALESRNET_x4.pth"

        # Create an RealEsrGan model instance.
        upsampler = RealEsrGan(
            scale=upscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            pre_pad=10,
            half=False,
            device=None,
            gpu_id=gpu_id,
        )

        # Process the input image.
        if hasattr(
            uploaded_file, "read"
        ):  # Check if it's a file uploaded from the local system.
            img_pil = Image.open(uploaded_file)
        elif uploaded_file.startswith("http"):  # If it's an image URL.
            response = requests.get(uploaded_file)
            img_pil = Image.open(BytesIO(response.content))
        else:
            st.warning(
                "Invalid input. Please provide either an image file or an image URL."
            )
            return

        # Convert PIL image to OpenCV format.
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        # Perform super-resolution using Real-ESRGAN.
        output, _ = upsampler.enhance(img, upscale=upscale)

        # Determine the file extension for saving the output image.
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = "RGBA"
            extension = "png"
        else:
            img_mode = None
            if extension == "auto":
                extension = "png"  # Default extension for images from URL.

        # Save the super resolution image
        save_path = f"{output_path}/{model_name}_inference.{extension}"
        cv2.imwrite(save_path, output)
    except Exception as e:
        st.error(e)
    return save_path


##############################################################################################################


# Function to apply local CSS.
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


##############################################################################################################
# Main function to create the Streamlit web application.
def main():
    try:
        # Load CSS.
        local_css("styles/style.css")

        # Title.
        title = f"""<p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.3rem;">
                    Super Upscale Resolution with Real-ESRGAN</p>"""
        st.markdown(title, unsafe_allow_html=True)

        # Toggle button for displaying text input or file uploader.
        title = f"""<p style="font-family: monospace; color: white;">
                    Enter Image URL or Upload Image</p>"""
        st.markdown(title, unsafe_allow_html=True)

        use_image_url = st.checkbox(
            label="Enter Image URL or Upload Image", label_visibility="collapsed"
        )

        # Input for image URL or file uploader based on the checkbox state.
        if use_image_url:
            image_url_label = f"""
                <p style="font-family: monospace; color: white;">Enter Image URL:</p>"""
            st.markdown(image_url_label, unsafe_allow_html=True)

            image_url = st.text_input(
                label="Enter Image URL:",
                value="",
                label_visibility="collapsed",
            )
        else:
            uploaded_file_label = f"""
                <p style="font-family: monospace; color: white;">Upload Image:</p>"""
            st.markdown(uploaded_file_label, unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                label="Upload Image:",
                type=["jpg", "png", "jpeg"],
                label_visibility="collapsed",
            )

        # Dropdown menu for model selection.
        model_name_label = f"""
                <p style="font-family: monospace; color: white;">Select Model:</p>"""
        st.markdown(model_name_label, unsafe_allow_html=True)

        model_name = st.selectbox(
            label="Select Model:",
            options=[
                "REALESRGAN_x4",
                "REALESRNET_x4",
            ],
            label_visibility="collapsed",
        )

        # Slider for upscale selection.
        model_name_label = f"""
                <p style="font-family: monospace; color: white;">Select Upscale Factor. Model works best with x4 upscale:</p>"""
        st.markdown(model_name_label, unsafe_allow_html=True)

        upscale = st.slider(
            label="Select Upscale Factor. Model works best with x4 upscale:",
            min_value=3,
            max_value=10,
            value=4,
            step=1,
            label_visibility="collapsed",
        )

        if not use_image_url and uploaded_file is not None:
            # Image caption.
            image_caption = f"""<p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.3rem;">
                        Uploaded Image:</p>"""
            st.markdown(image_caption, unsafe_allow_html=True)
            st.image(uploaded_file)

        with st.spinner():
            if st.button("Run Inference"):
                if use_image_url and image_url != "":
                    result_path = run_inference(
                        uploaded_file=image_url,
                        model_name=model_name,
                        upscale=upscale,
                    )
                    # Image caption.
                    image_caption = f"""<p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.3rem;">
                                Resulting Image:</p>"""
                    st.markdown(image_caption, unsafe_allow_html=True)
                    st.image(result_path)

                    st.success("Inference completed!")
                elif not use_image_url and uploaded_file is not None:
                    result_path = run_inference(
                        uploaded_file=uploaded_file,
                        model_name=model_name,
                        upscale=upscale,
                    )

                    # Image caption.
                    image_caption = f"""<p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.3rem;">
                                Resulting Image:</p>"""
                    st.markdown(image_caption, unsafe_allow_html=True)
                    st.image(result_path)

                    st.success("Inference completed!")
                else:
                    st.warning("Please provide either an image file or an image URL.")

        # GitHub repository of this project.
        st.markdown(
            f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 1rem;">
                <b>Check out our <a href="https://github.com/GeorgiosIoannouCoder/realesrgan" style="color: #FAF9F6;">GitHub repository</a></b>
                </p>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(e)


if __name__ == "__main__":
    main()
##############################################################################################################

import streamlit as st
import io

# Disable warning
st.set_option("deprecation.showfileUploaderEncoding", False)


@st.cache(show_spinner=False)
def load_image_file(image_path):
    """
    Loads an Image file
    """
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = np.array(image)
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def random_image():
    if st.button("Press Me!"):
        # Just load an image from sample_images folder
        random_image = random.choice(SAMPLE_IMAGES)
        st.image(random_image)
        image = load_image_file(random_image)
        flag = 1
        st.image(
            image, use_column_width=True,
        )


# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():
    st.sidebar.markdown("# Model Thresholds")
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.01
    )
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold


# st.header("Hello world !")
st.image("images/icevision.png")

st.sidebar.image("images/airctic-logo-medium.png")

page = st.sidebar.selectbox(
    "Choose a dataset", ["PETS", "PennFundan", "Fridge Objects", "Raccoon"]
)  # pages

# Draw the UI element to select parameters for the YOLO object detector.
confidence_threshold, overlap_threshold = object_detector_ui()


random_image()

# st.write("# Upload an Image to get its predictions")

# img_file_buffer = st.file_uploader("", type=["png", "jpg", "jpeg"])
# # img_file_buffer = io.TextIOWrapper(img_file_buffer)
# if img_file_buffer is not None:
#     image = load_image_file(img_file_buffer)
#     if image is not None:
#         st.image(
#             image,
#             caption=f"Your amazing image has shape {image.shape[0:2]}",
#             use_column_width=True,
#         )
#         flag = 1
#     else:
#         print("Invalid input")

st.balloons()

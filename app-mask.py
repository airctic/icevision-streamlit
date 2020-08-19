import streamlit as st
import PIL, requests
from mantisshrimp.all import *
from PIL import Image

WEIGHTS_URL = "https://mantisshrimp-models.s3.us-east-2.amazonaws.com/pennfundan_maskrcnn_resnet50fpn.zip"
CLASS_MAP = datasets.pennfundan.class_map()

pennfundan_images = [
    "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image1.png",
    "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image2.png",
    "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image3.png",
    "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image4.png",
    "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image5.png",
    "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image6.png",
]


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
        random_image = random.choice(pennfundan_images)
        image_url = st.text_input(label="Image url", value=random_image,)
        # st.image(random_image)
        # image = load_image_file(random_image)
        flag = 1


# This sidebar UI lets the user select model thresholds.
def object_detector_ui():
    st.sidebar.markdown("# Model Thresholds")
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.01
    )
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold


def sidebar_ui():
    st.sidebar.image("images/airctic-logo-medium.png")

    page = st.sidebar.selectbox(
        "Choose a dataset", ["PETS", "PennFundan", "Fridge Objects", "Raccoon"]
    )  # pages

    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold, overlap_threshold = object_detector_ui()


@st.cache(allow_output_mutation=True)
def load_model():
    model = mask_rcnn.model(num_classes=len(CLASS_MAP))
    state_dict = torch.hub.load_state_dict_from_url(
        WEIGHTS_URL, map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)
    return model


def image_from_url(url):
    res = requests.get(url, stream=True)
    img = PIL.Image.open(res.raw)
    return np.array(img)


def predict(model, image_url):
    img = image_from_url(image_url)

    tfms_ = tfms.A.Adapter([tfms.A.Normalize()])
    # Whenever you have images in memory (numpy arrays) you can use `Dataset.from_images`
    infer_ds = Dataset.from_images([img], tfms_)

    batch, samples = mask_rcnn.build_infer_batch(infer_ds)
    preds = mask_rcnn.predict(model=model, batch=batch)

    return samples[0]["img"], preds[0]


def show_prediction(img, pred):
    show_pred(
        img=img,
        pred=pred,
        class_map=CLASS_MAP,
        denormalize_fn=denormalize_imagenet,
        show=True,
        bbox=False,
    )

    # Grab image from the current matplotlib figure
    fig = plt.gcf()
    fig.canvas.draw()
    fig_arr = np.array(fig.canvas.renderer.buffer_rgba())

    st.image(fig_arr, use_column_width=True)


def run_app():
    st.image("images/icevision.png")
    # st.image("images/image3.png")
    sidebar_ui()

    image_url = st.text_input(
        label="Image url",
        value="https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image1.png",
    )

    if st.button("Press Me!"):
        # Just load an image from sample_images folder
        random_image = random.choice(pennfundan_images)
        image_url = st.text_input(label="Image url", value=random_image,)
    # random_image()

    model = load_model()

    img, pred = predict(model=model, image_url=image_url)
    show_prediction(img=img, pred=pred)


if __name__ == "__main__":
    run_app()

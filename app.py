import streamlit as st
import PIL, requests
from icevision.all import *
from PIL import Image
from random import randrange
from streamlit import caching

MASK_PENNFUNDAN_WEIGHTS_URL = "https://github.com/airctic/model_zoo/releases/download/pennfudan_maskrcnn_resnet50fpn/pennfudan_maskrcnn_resnet50fpn.zip"


# Just for tests. Remove after
# caching.clear_cache()

# @st.cache(allow_output_mutation=True)
def load_model(class_map=class_map, url=None):
    if url is None:
        # print("Please provide a valid URL")
        return None
    else:
        backbone = backbones.resnet_fpn.resnet50(pretrained=False) 
        model = mask_rcnn.model(backbone=backbone, num_classes=len(class_map), pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)
        return model

class_map = icedata.pennfudan.class_map()
model = load_model(class_map=class_map, url=MASK_PENNFUNDAN_WEIGHTS_URL)


def sidebar_ui():
    # st.sidebar.image("images/airctic-logo-medium.png")
    st.sidebar.image(
        "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/icevision-deploy-small.png"
    )


# This sidebar UI lets the user select model thresholds.
def object_detector_ui():
    # st.sidebar.markdown("# Model Thresholds")
    detection_threshold = st.sidebar.slider("Detection threshold", 0.0, 1.0, 0.5, 0.01)
    mask_threshold = st.sidebar.slider("Mask threshold", 0.0, 1.0, 0.5, 0.01)
    return detection_threshold, mask_threshold



def image_from_url(url):
    res = requests.get(url, stream=True)
    img = PIL.Image.open(res.raw)
    return np.array(img)


def predict(
    model,
    image,
    detection_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    display_label=True,
    display_bbox=True,
    display_mask=True,
):
    img = np.array(image)
    tfms_ = tfms.A.Adapter([tfms.A.Normalize()])
    # Whenever you have images in memory (numpy arrays) you can use `Dataset.from_images`
    infer_ds = Dataset.from_images([img], tfms_)

    batch, samples = mask_rcnn.build_infer_batch(infer_ds)
    preds = mask_rcnn.predict(
        model=model,
        batch=batch,
        detection_threshold=detection_threshold,
        mask_threshold=mask_threshold,
    )
    return samples[0]["img"], preds[0]


def get_masks(
    model,
    image_url,
    class_map=None,
    detection_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    display_label=True,
    display_bbox=True,
    display_mask=True,
):

    # Loading image from url
    input_image = np.array(image_url)

    img, pred = predict(
        model=model,
        image=input_image,
        detection_threshold=detection_threshold,
        mask_threshold=mask_threshold,
        display_label=display_label,
        display_bbox=display_bbox,
        display_mask=display_mask,
    )

    img = draw_pred(
        img=img,
        pred=pred,
        class_map=class_map,
        denormalize_fn=denormalize_imagenet,
        display_label=display_label,
        display_bbox=display_bbox,
        display_mask=display_mask,
    )
    img = PIL.Image.fromarray(img)
    # print("Output Image: ", img.size, type(img))
    return img


def run_app():   
    st.set_option('deprecation.showfileUploaderEncoding', False)
    sidebar_ui()

    # Draw the threshold parameters for object detection model.
    detection_threshold, mask_threshold = object_detector_ui()

    label = st.sidebar.checkbox(label="Label", value=True)
    bbox = st.sidebar.checkbox(label="Bounding Box", value=True)
    mask = st.sidebar.checkbox(label="Mask", value=True)

    st.sidebar.image(
        "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/airctic-logo-medium.png"
    )

    st.markdown("### ** Insert an image**")
    uploaded_file = st.file_uploader("")  # image upload widget
    my_placeholder = st.empty()
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        my_placeholder.image(image, caption="", use_column_width=True)
    
    if image:
        segmented_image = get_masks(
            model,
            image,
            class_map=class_map,
            detection_threshold=float(detection_threshold),
            mask_threshold=float(mask_threshold),
            display_label=label,
            display_bbox=bbox,
            display_mask=mask,
        )
        my_placeholder.image(segmented_image)


if __name__ == "__main__":
    run_app()

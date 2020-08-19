import streamlit as st
import PIL, requests
from mantisshrimp.all import *

WEIGHTS_URL = "https://mantisshrimp-models.s3.us-east-2.amazonaws.com/pennfundan_maskrcnn_resnet50fpn.zip"
class_map = datasets.pennfundan.class_map()


@st.cache(allow_output_mutation=True)
def load_model():
    model = mask_rcnn.model(num_classes=len(class_map))
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
        class_map=class_map,
        denormalize_fn=denormalize_imagenet,
        show=True,
    )

    # Grab image from the current matplotlib figure
    fig = plt.gcf()
    fig.canvas.draw()
    fig_arr = np.array(fig.canvas.renderer.buffer_rgba())

    st.image(fig_arr, use_column_width=True)


def run_app():
    st.title("MantisShrimp Demo App")

    image_url = st.text_input(
        label="Image url",
        value="https://petcaramelo.com/wp-content/uploads/2018/06/beagle-cachorro.jpg",
    )

    model = load_model()

    img, pred = predict(model=model, image_url=image_url)
    show_prediction(img=img, pred=pred)


if __name__ == "__main__":
    run_app()

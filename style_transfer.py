import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub


@st.cache_resource(show_spinner=False)
def load_hub_model():
    return hub.load(
        "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    )


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = tf.cast(tensor, tf.uint8)
    return tf.keras.preprocessing.image.array_to_img(tensor[0])


def load_image(img_bytes, max_dim=512):
    img = tf.io.decode_image(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    h, w, _ = img.shape
    scale = max_dim / max(h, w)
    img = tf.image.resize(img, (int(h * scale), int(w * scale)))
    return img[tf.newaxis, :]


def run_style_transfer(content_bytes, style_bytes):
    content_image = load_image(content_bytes)
    style_image = load_image(style_bytes, max_dim=256)
    hub_model = load_hub_model()
    stylized = hub_model(tf.constant(content_image),
                         tf.constant(style_image))[0]
    return tensor_to_image(stylized)


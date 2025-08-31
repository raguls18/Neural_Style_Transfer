import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import array_to_img
import gc

# Cache and load the TF-Hub model once per session
@st.cache_resource(show_spinner=False)
def load_hub_model():
    return hub.load(
        "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    )

def clear_tf_session():
    """
    Clear TensorFlow session and release GPU/CPU memory.
    """
    tf.keras.backend.clear_session()
    gc.collect()

def load_image(img_bytes: bytes, max_dim: int = 512) -> tf.Tensor:
    """
    Decode image bytes and resize to max_dim, preserving aspect ratio.
    Returns a 4D float32 tensor in [0,1].
    """
    img = tf.io.decode_image(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    h, w, _ = img.shape
    scale = max_dim / max(h, w)
    new_size = (int(h * scale), int(w * scale))
    img = tf.image.resize(img, new_size)
    return img[tf.newaxis, :]

def tensor_to_image(tensor: tf.Tensor) -> 'PIL.Image.Image':
    """
    Convert a 4D float32 tensor in [0,1] to a PIL Image.
    """
    tensor = tf.clip_by_value(tensor, 0.0, 1.0)
    return array_to_img(tensor[0])

def run_style_transfer(content_bytes: bytes, style_bytes: bytes):
    """
    Perform style transfer on the given image bytes.
    Returns a PIL Image.
    """
    # Load and preprocess images
    content_image = load_image(content_bytes, max_dim=512)
    style_image   = load_image(style_bytes,   max_dim=256)

    # Load the model (cached)
    hub_model = load_hub_model()

    # Perform style transfer
    stylized_tensor = hub_model(
        tf.constant(content_image), tf.constant(style_image)
    )[0]

    # Convert to PIL Image and return
    return tensor_to_image(stylized_tensor)

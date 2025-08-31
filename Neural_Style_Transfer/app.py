import streamlit as st
from io import BytesIO
from PIL import Image
from style_transfer import run_style_transfer, clear_tf_session

# Page config
st.set_page_config(page_title="Neural Style Transfer", page_icon="🎨")

# Title and description
st.title("🎨 Neural Style Transfer")
st.write(
    "Upload a **content image** and a **style image** – the model will paint "
    "the content in the style you chose."
)

# File upload columns
col1, col2 = st.columns(2)
with col1:
    content_file = st.file_uploader(
        "Content image", type=["jpg", "jpeg", "png"], key="content"
    )
with col2:
    style_file = st.file_uploader(
        "Style image", type=["jpg", "jpeg", "png"], key="style"
    )

# Stylize button
if content_file and style_file and st.button("Stylize!"):
    with st.spinner("🖌️ Applying style…"):
        # Run style transfer
        try:
            result_img = run_style_transfer(
                content_file.read(), style_file.read()
            )
            st.success("✅ Style transfer complete!")
            st.image(result_img, caption="🎉 Stylized Image", use_container_width=True)

            # Download button
            buf = BytesIO()
            result_img.save(buf, format="JPEG")
            byte_data = buf.getvalue()
            st.download_button(
                "📥 Download Image",
                data=byte_data,
                file_name="stylized.jpg",
                mime="image/jpeg",
            )
        finally:
            # Clear TF session to free memory
            clear_tf_session()

# Footer
st.markdown("---")
st.markdown(
    "Model: [Arbitrary-Image Stylization - TF Hub]"
    "(https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)"
)

import streamlit as st
import os
import gdown
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

# --- 1. Global Configurations ---
# Configuration for model storage and Google Drive integration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'Defect_Detection_VGG16.keras')
LABEL_PATH = os.path.join(MODEL_DIR, 'class_names.pkl')
# Replace with your actual Google Drive file ID
GOOGLE_DRIVE_ID = '1WsTobG4K-wkinbzQ9fP1Vy_5YImJb9rJ'

# --- 2. Page Setup ---
st.set_page_config(
    page_title="AI Surface Defect Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Industrial Surface Defect Detection (VGG16 + XAI)")
st.subheader("Interactive Model Demo & Root Cause Analysis")
st.markdown("""
This demo utilizes a **VGG16-based Deep Learning model** to detect surface defects in manufacturing. 
It integrates **Explainable AI (Grad-CAM)** to visualize the model's decision-making process.
""")

# --- 3. Asset Initialization (Model & Labels) ---
@st.cache_resource
def load_assets():
    """
    Downloads the model from Google Drive if not present and loads all assets.
    'custom_objects' is used to map preprocess_input for the VGG16 architecture.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Automated download for Streamlit Cloud deployment
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading pre-trained model from Google Drive..."):
            url = f'https://drive.google.com/uc?id={'GOOGLE_DRIVE_ID'}'
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    # Load Model
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'preprocess_input': preprocess_input}
    )
    
    # Load Class Labels
    with open(LABEL_PATH, 'rb') as f:
        labels = pickle.load(f)
        
    return model, labels

try:
    model, class_names = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# --- 4. Explainable AI Logic (Grad-CAM) ---
def generate_gradcam(img_tensor, model):
    """
    Computes Grad-CAM heatmap to visualize 'where' the model is looking.
    """
    # Locating the target convolutional layer (VGG16 backbone)
    vgg_layer = next(l for l in model.layers if 'vgg' in l.name.lower())
    last_conv_layer = vgg_layer.get_layer("block5_conv3")
    
    # Gradient model setup
    grad_model = tf.keras.models.Model(
        [vgg_layer.inputs], 
        [last_conv_layer.output, vgg_layer.output]
    )

    with tf.GradientTape() as tape:
        # Pre-processing flow through initial layers
        x = img_tensor
        for layer in model.layers:
            if layer == vgg_layer: break
            x = layer(x)
            
        conv_outputs, predictions = grad_model(x)
        loss = tf.reduce_max(predictions[0])

    # Computing gradients for visual importance
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))[0]
    
    # Heatmap generation and normalization
    heatmap = conv_outputs.numpy()[0] @ pooled_grads.numpy()[..., np.newaxis]
    heatmap = np.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    
    return heatmap

# --- 5. User Interface (Sidebar & Upload) ---
st.sidebar.header("Deployment Settings")
uploaded_file = st.sidebar.file_uploader("Upload Surface Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Image Preprocessing for VGG16
    raw_image = Image.open(uploaded_file).convert('RGB')
    display_img = raw_image.resize((200, 200))
    img_array = np.array(display_img).astype('float32')
    img_tensor = np.expand_dims(img_array, axis=0)

    # Model Inference
    preds = model.predict(img_tensor)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx]

    # --- 6. Results Visualization ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Inspection Result")
        st.image(raw_image, caption="Uploaded Specimen", use_container_width=True)
        st.metric(label="Predicted Class", value=class_names[pred_idx].upper())
        st.write(f"**Confidence Score:** {confidence:.2%}")

    with col2:
        st.subheader("XAI Analysis (Grad-CAM)")
        with st.spinner("Generating attention map..."):
            heatmap = generate_gradcam(img_tensor, model)
            
            fig, ax = plt.subplots()
            ax.imshow(display_img)
            ax.imshow(heatmap, cmap='jet', alpha=0.4) 
            ax.axis('off')
            st.pyplot(fig)
            
        st.caption("The highlighted regions indicate feature patterns significant to the model's decision.")

else:
    st.info("Awaiting image upload for automated inspection.")
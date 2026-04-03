import streamlit as st
import os
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.vgg16 import preprocess_input

# --- 1. Global Configurations ---
# Define directory paths relative to the current script location for portability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'Defect_Detection_VGG16.keras')
LABEL_PATH = os.path.join(MODEL_DIR, 'class_names.pkl')

# Hugging Face Repository Information
HF_REPO_ID = 'SeokyoungHwang/Industrial_Defect_Detection' 

# --- 2. Page Setup ---
st.set_page_config(
    page_title="AI Surface Defect Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Industrial Surface Defect Detection (VGG16 + XAI)")
st.subheader("Interactive Model Demo & Root Cause Analysis")
st.markdown("""
This application utilizes a **VGG16-based Deep Learning model** to automate surface defect detection. 
It features **Explainable AI (Grad-CAM)** to visualize the features influencing the model's decision.
""")

# --- 3. Asset Initialization (Model & Labels) ---
@st.cache_resource
def load_assets():
    """
    Downloads the model from Hugging Face Hub if not cached and loads inference assets.
    """
    # Ensure the local model directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Programmatic download from Hugging Face Hub for stable deployment
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading pre-trained model from Hugging Face Hub..."):
            try:
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename="Defect_Detection_VGG16.keras",
                    local_dir=MODEL_DIR
                )
            except Exception as e:
                st.error(f"Failed to download model from Hugging Face: {e}")
                st.stop()

    # Load the Keras model with necessary custom objects
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'preprocess_input': preprocess_input}
    )
    
    # Load class labels for human-readable output
    if not os.path.exists(LABEL_PATH):
        st.error(f"Label file missing at: {LABEL_PATH}")
        st.stop()

    with open(LABEL_PATH, 'rb') as f:
        labels = pickle.load(f)
        
    return model, labels

# Execute asset loading
try:
    model, class_names = load_assets()
except Exception as e:
    st.error(f"Application failed to initialize: {e}")
    st.stop()

# --- 4. Explainable AI Logic (Grad-CAM) ---
def generate_gradcam(img_tensor, model):
    """
    Computes the Grad-CAM heatmap to visualize model attention.
    """
    # Access the VGG16 backbone within the model architecture
    vgg_layer = next(l for l in model.layers if 'vgg' in l.name.lower())
    last_conv_layer = vgg_layer.get_layer("block5_conv3")
    
    # Construct a sub-model that outputs both activations and final predictions
    grad_model = tf.keras.models.Model(
        inputs=vgg_layer.input, 
        outputs=[last_conv_layer.output, vgg_layer.output]
    )

    with tf.GradientTape() as tape:
        # Route input through initial layers if any exist before the VGG block
        x = img_tensor
        for layer in model.layers:
            if layer == vgg_layer: break
            x = layer(x)
            
        conv_outputs, predictions = grad_model(x, training=False)
        loss = tf.reduce_max(predictions)

    # Compute gradients and global average pooling for channel weights
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))[0]
    
    # Generate weighted heatmap from convolutional outputs
    heatmap = conv_outputs.numpy()[0] @ pooled_grads.numpy()[..., np.newaxis]
    heatmap = np.squeeze(heatmap)
    
    # ReLU activation and Normalization
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    return heatmap

# --- 5. User Interface (Sidebar & Upload) ---
st.sidebar.header("Operations")
uploaded_file = st.sidebar.file_uploader("Upload Surface Specimen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Image preprocessing for VGG16 model requirements
    raw_image = Image.open(uploaded_file).convert('RGB')
    display_img = raw_image.resize((200, 200))
    img_array = np.array(display_img).astype('float32')
    img_tensor = np.expand_dims(img_array, axis=0)

    # Perform inference
    preds = model.predict(img_tensor)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx]

    # --- 6. Results Visualization ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Diagnostic Report")
        st.image(raw_image, caption="Uploaded Specimen", use_container_width=True)
        st.metric(label="Predicted Classification", value=class_names[pred_idx].upper())
        st.write(f"**Confidence Score:** {confidence:.2%}")

    with col2:
        st.subheader("Decision Evidence (XAI)")
        with st.spinner("Generating Grad-CAM heatmap..."):
            heatmap = generate_gradcam(img_tensor, model)
            
            fig, ax = plt.subplots()
            ax.imshow(display_img)
            # Overlay heatmap with transparency for better visualization
            ax.imshow(heatmap, cmap='jet', alpha=0.4) 
            ax.axis('off')
            st.pyplot(fig)
            
        st.caption("The colored regions highlight specific features (e.g., textures, cracks) that dictated the model's prediction.")

else:
    st.info("System Ready. Please upload a surface image via the sidebar for inspection.")
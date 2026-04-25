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
# Define path for sample images
SAMPLE_DIR = os.path.join(BASE_DIR, 'samples')

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

# --- 5. User Interface (Sidebar, Upload & 6-Class Samples) ---
# Quick Test Section
st.sidebar.subheader("🚀 Try Instant Demo (Select a Defect)")
st.sidebar.info("Click a defect type to see the AI analysis in action.")

# Defining the 6 specific defect classes from the NEU dataset
SAMPLES = {
    "Crazing": "crazing_sample.jpg",
    "Inclusion": "inclusion_sample.jpg",
    "Patches": "patches_sample.jpg",
    "Pitted": "pitted_sample.jpg",
    "Rolled": "rolled_sample.jpg",
    "Scratches": "scratches_sample.jpg"
}

# Initialize session state for sample tracking
if "sample_path" not in st.session_state:
    st.session_state.sample_path = None

# Create a 2-column layout for sample buttons to save space
col1, col2 = st.sidebar.columns(2)
for i, (label, filename) in enumerate(SAMPLES.items()):
    with col1 if i % 2 == 0 else col2:
        if st.button(label, width="stretch"):
            st.session_state.sample_path = os.path.join(SAMPLE_DIR, filename)

# Manual Upload
st.sidebar.markdown("---")
st.sidebar.header("Operations")
uploaded_file = st.sidebar.file_uploader("Upload Surface Specimen", type=["jpg", "jpeg", "png"])

# Determine the active image (Priority: Upload > Sample)
raw_image = None
if uploaded_file:
    # If a user uploads a file, it takes the highest priority.
    raw_image = Image.open(uploaded_file).convert('RGB')
    # Clear sample selection to ensure the uploaded file is the one being analyzed.
    st.session_state.sample_path = None 
elif st.session_state.sample_path:
    # If no file is uploaded, use the sample image selected via buttons.
    try:
        raw_image = Image.open(st.session_state.sample_path).convert('RGB')
    except Exception as e:
        st.error(f"Error loading sample image: {e}")

# --- 6. Inference & Results Visualization ---
if raw_image:
    # Image preprocessing
    display_img = raw_image.resize((200, 200))
    img_array = np.array(display_img).astype('float32')
    img_tensor = np.expand_dims(img_array, axis=0)

    # Perform inference
    with st.spinner("Analyzing surface..."):
        preds = model.predict(img_tensor)
        pred_idx = np.argmax(preds[0])
        confidence = preds[0][pred_idx]
        result_text = class_names[pred_idx].upper()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Diagnostic Report")
        st.image(raw_image, caption="Inspected Specimen", width="stretch")
        
        # Since all 6 classes are defects, we use st.error for visibility
        st.error(f"**Detected Defect:** {result_text}")
        st.metric(label="Model Confidence", value=f"{confidence:.2%}")
        
        # Simple Logic: If confidence is low, add a cautionary note
        if confidence < 0.70:
            st.warning("⚠️ Low confidence detected. Manual inspection is recommended.")

    with col2:
        st.subheader("Decision Evidence (XAI)")
        with st.spinner("Generating Grad-CAM heatmap..."):
            heatmap = generate_gradcam(img_tensor, model)
            
            fig, ax = plt.subplots()
            ax.imshow(display_img)
            # Overlaying the heatmap to show the 'Root Cause' area
            ax.imshow(heatmap, cmap='jet', alpha=0.4)
            ax.axis('off')
            st.pyplot(fig)
            
        st.caption(f"The heatmap visualizes the specific features that the VGG16 model identified as '{result_text.lower()}'.")

else:
    # Initial landing state
    st.info("System Ready. Please upload a surface image or select a sample defect from the gallery.")
import streamlit as st
import os
import gdown
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AI Surface Defect Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Manufacturing Defect Audit System (VGG16 + XAI)")
st.markdown("""
This application utilizes a **VGG16-based Deep Learning model** to detect surface defects in manufacturing. 
It integrates **Explainable AI (Grad-CAM)** to visualize the model's decision-making process.
""")

# --- 2. Model & Metadata Loading ---
@st.cache_resource
def load_assets():
    """
    Load the trained model and class labels.
    Note: 'custom_objects' is required to deserialize the Lambda layer with preprocess_input.
    """
    model_path = 'models/Defect_Detection_VGG16.keras'
    label_path = 'models/class_names.pkl'
    
    # Load Model with custom preprocessing mapping
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'preprocess_input': preprocess_input}
    )
    
    # Load Class Labels
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
        
    return model, labels

try:
    model, class_names = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# --- 3. Grad-CAM Logic (Explainable AI) ---
def generate_gradcam(img_tensor, model):
    """
    Computes Grad-CAM heatmap to visualize 'where' the model is looking.
    """
    # Locating the last convolutional layer of the VGG16 backbone
    vgg_layer = next(l for l in model.layers if 'vgg' in l.name.lower())
    last_conv_layer = vgg_layer.get_layer("block5_conv3")
    
    # Create a sub-model that outputs both the last conv layer and the final prediction
    grad_model = tf.keras.models.Model(
        [vgg_layer.inputs], 
        [last_conv_layer.output, vgg_layer.output]
    )

    with tf.GradientTape() as tape:
        # Forward pass through the initial layers (Augmentation/Lambda)
        x = img_tensor
        for layer in model.layers:
            if layer == vgg_layer: break
            x = layer(x)
            
        conv_outputs, predictions = grad_model(x)
        loss = tf.reduce_max(predictions[0])

    # Gradient of the loss with respect to the output feature map
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))[0]
    
    # Weight the channels of the feature map by the gradient importance
    heatmap = conv_outputs.numpy()[0] @ pooled_grads.numpy()[..., np.newaxis]
    heatmap = np.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10) # Normalize
    
    return heatmap

# --- 4. User Interface: Sidebar & Upload ---
st.sidebar.header("Deployment Settings")
uploaded_file = st.sidebar.file_uploader("Upload Surface Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Image Preprocessing
    raw_image = Image.open(uploaded_file).convert('RGB')
    display_img = raw_image.resize((200, 200))
    img_array = np.array(display_img).astype('float32')
    img_tensor = np.expand_dims(img_array, axis=0)

    # Model Inference
    preds = model.predict(img_tensor)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx]

    # --- 5. Display Results ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Inspection Result")
        st.image(raw_image, caption="Uploaded Image", use_container_width=True)
        
        # Professional Metric Display
        st.metric(label="Predicted Class", value=class_names[pred_idx].upper())
        st.write(f"**Confidence Score:** {confidence:.2%}")

    with col2:
        st.subheader("XAI Analysis (Grad-CAM)")
        with st.spinner("Generating attention map..."):
            heatmap = generate_gradcam(img_tensor, model)
            
            # Overlay Heatmap on Image
            fig, ax = plt.subplots()
            ax.imshow(display_img)
            ax.imshow(heatmap, cmap='jet', alpha=0.4) # Overlay with 40% transparency
            ax.axis('off')
            st.pyplot(fig)
            
        st.caption("The red regions highlight where the AI detected defect patterns.")

else:
    st.info("Please upload an image from the sidebar to begin the automated inspection.")
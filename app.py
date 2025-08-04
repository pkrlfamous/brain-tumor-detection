# Import libraries for Streamlit app
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        model = tf.keras.models.load_model("brain_tumor_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_model_summary(model):
    """Get simplified model information"""
    try:
        total_params = model.count_params()
        layer_count = len(model.layers)
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        return {
            'total_params': total_params,
            'layer_count': layer_count,
            'input_shape': input_shape,
            'output_shape': output_shape
        }
    except Exception as e:
        return {'error': str(e)}

def make_gradcam_heatmap_robust(img_array, model, layer_name=None):
    """
    Robust Grad-CAM implementation that handles transfer learning models better
    """
    try:
        # Convert input to tensor if it's numpy array
        if isinstance(img_array, np.ndarray):
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        else:
            img_tensor = img_array
        
        # Find the target layer
        target_layer = None
        
        if layer_name:
            try:
                target_layer = model.get_layer(layer_name)
            except:
                pass
        
        # If no specific layer or layer not found, find the last conv layer
        if target_layer is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    target_layer = layer
                    break
                # Check if it's a nested functional model (like MobileNetV2)
                elif hasattr(layer, 'layers'):
                    for sublayer in reversed(layer.layers):
                        if isinstance(sublayer, tf.keras.layers.Conv2D):
                            # Create a new model that outputs this sublayer
                            try:
                                grad_model = tf.keras.models.Model(
                                    inputs=model.input,
                                    outputs=[sublayer.output, model.output]
                                )
                                
                                with tf.GradientTape() as tape:
                                    tape.watch(img_tensor)
                                    conv_outputs, predictions = grad_model(img_tensor)
                                    loss = predictions[:, 0]
                                
                                grads = tape.gradient(loss, conv_outputs)
                                
                                if grads is not None:
                                    # Global average pooling on gradients
                                    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                                    
                                    # Multiply each channel by its gradient
                                    conv_outputs = conv_outputs[0]
                                    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                                    heatmap = tf.squeeze(heatmap)
                                    
                                    # Normalize heatmap
                                    heatmap = tf.maximum(heatmap, 0)
                                    if tf.reduce_max(heatmap) > 0:
                                        heatmap = heatmap / tf.reduce_max(heatmap)
                                    
                                    return heatmap.numpy()
                            except Exception as e:
                                continue
        
        # If we found a direct conv layer
        if target_layer is not None:
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[target_layer.output, model.output]
            )
            
            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                conv_outputs, predictions = grad_model(img_tensor)
                loss = predictions[:, 0]
            
            grads = tape.gradient(loss, conv_outputs)
            
            if grads is not None:
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_outputs = conv_outputs[0]
                heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                
                heatmap = tf.maximum(heatmap, 0)
                if tf.reduce_max(heatmap) > 0:
                    heatmap = heatmap / tf.reduce_max(heatmap)
                
                return heatmap.numpy()
        
        return None
        
    except Exception as e:
        st.warning(f"Grad-CAM failed: {str(e)}")
        return None

def create_input_gradient_heatmap(img_array, model):
    """
    Create heatmap based on input gradients - more reliable method
    """
    try:
        # Ensure img_array is a tensor
        if isinstance(img_array, np.ndarray):
            img_tensor = tf.Variable(img_array, dtype=tf.float32)
        else:
            img_tensor = tf.Variable(img_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            # Use the class prediction (tumor vs no tumor)
            class_idx = tf.argmax(predictions, axis=1)
            loss = predictions[:, 0]  # For binary classification
        
        # Get gradients
        grads = tape.gradient(loss, img_tensor)
        
        if grads is not None:
            # Convert to numpy
            grads_np = grads.numpy()[0]  # Remove batch dimension
            
            # Take absolute values and sum across channels
            grads_abs = np.abs(grads_np)
            heatmap = np.sum(grads_abs, axis=-1)
            
            # Normalize
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            # Apply Gaussian smoothing
            heatmap_smooth = cv2.GaussianBlur(heatmap, (15, 15), 0)
            
            return heatmap_smooth
        
        return None
        
    except Exception as e:
        st.warning(f"Input gradient method failed: {str(e)}")
        return None

def create_occlusion_heatmap(img_array, model, patch_size=16, stride=8):
    """
    Create heatmap using occlusion sensitivity - most reliable but slower
    """
    try:
        original_img = img_array[0].copy()  # Remove batch dimension
        original_pred = model.predict(img_array, verbose=0)[0][0]
        
        h, w = original_img.shape[:2]
        heatmap = np.zeros((h, w))
        
        # Create patches to occlude
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # Create occluded image
                occluded_img = original_img.copy()
                occluded_img[y:y+patch_size, x:x+patch_size] = 0  # Black patch
                
                # Reshape for model input
                occluded_input = occluded_img.reshape(1, *occluded_img.shape)
                
                # Get prediction
                occluded_pred = model.predict(occluded_input, verbose=0)[0][0]
                
                # Calculate importance as difference in prediction
                importance = abs(original_pred - occluded_pred)
                
                # Fill heatmap region
                heatmap[y:y+patch_size, x:x+patch_size] = np.maximum(
                    heatmap[y:y+patch_size, x:x+patch_size], importance
                )
        
        # Normalize
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Smooth the heatmap
        heatmap_smooth = cv2.GaussianBlur(heatmap, (21, 21), 0)
        
        return heatmap_smooth
        
    except Exception as e:
        st.warning(f"Occlusion method failed: {str(e)}")
        return None

def create_simple_attention_map(img_array, model, prediction_confidence):
    """
    Fallback method: create a simple attention map based on image features
    """
    try:
        img = img_array[0]  # Remove batch dimension
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            img_gray = (img * 255).astype(np.uint8)
        
        # Edge detection
        edges = cv2.Canny(img_gray, 50, 150)
        
        # Morphological operations to enhance regions
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Gaussian blur to create smooth attention
        attention = cv2.GaussianBlur(edges_dilated, (25, 25), 0)
        
        # Normalize
        attention = attention.astype(np.float32) / 255.0
        
        # Scale by prediction confidence
        confidence_factor = abs(prediction_confidence - 0.5) * 2  # 0 to 1
        attention = attention * confidence_factor + 0.1  # Add small baseline
        
        # Final normalization
        if np.max(attention) > 0:
            attention = attention / np.max(attention)
        
        return attention
        
    except Exception as e:
        st.warning(f"Simple attention method failed: {str(e)}")
        return None

def create_overlay_image(original_img, heatmap, alpha=0.6):
    """Create overlay of heatmap on original image with better handling"""
    try:
        # Ensure original_img is in correct format
        if len(original_img.shape) == 3 and original_img.shape[-1] == 3:
            # RGB image
            original_display = (original_img * 255).astype(np.uint8)
        else:
            # Grayscale - convert to RGB
            if len(original_img.shape) == 3:
                original_gray = original_img[:, :, 0]
            else:
                original_gray = original_img
            original_display = cv2.cvtColor((original_gray * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Ensure heatmap is the right size
        if heatmap.shape != original_display.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_display.shape[1], original_display.shape[0]))
        
        # Create colored heatmap using matplotlib colormap
        heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Create overlay
        overlay = cv2.addWeighted(original_display, 1-alpha, heatmap_colored, alpha, 0)
        
        return overlay, heatmap_colored
        
    except Exception as e:
        st.error(f"Error creating overlay: {e}")
        return None, None

def preprocess_image(uploaded_file):
    """Preprocess uploaded image for model input"""
    try:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert to grayscale if RGB
        if len(image_array.shape) == 3:
            image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image_array
        
        # Resize to model input size
        image_resized = cv2.resize(image_gray, (128, 128))
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to RGB format for model (repeat grayscale across 3 channels)
        image_rgb = np.stack([image_normalized, image_normalized, image_normalized], axis=-1)
        
        return image_normalized, image_rgb
    
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None

def main():
    st.title("Brain Tumor Detection Demo")
    st.markdown(
    """
    **Medical Disclaimer**: This is a demonstration tool for educational purposes only.  
    It should **never** be used for actual medical diagnosis. Always consult qualified healthcare professionals.
    """
)
    st.markdown("Upload an MRI image to detect potential brain tumors using deep learning.")
  

    # Load model
    model = load_model()
    if model is None:
        st.error("Could not load the model. Please ensure 'brain_tumor_model.h5' exists in the current directory.")
        return
    
    # Model information
    with st.expander("Model Information"):
        model_info = get_model_summary(model)
        if 'error' not in model_info:
            st.write(f"**Total Parameters:** {model_info['total_params']:,}")
            st.write(f"**Number of Layers:** {model_info['layer_count']}")
            st.write(f"**Input Shape:** {model_info['input_shape']}")
            st.write(f"**Output Shape:** {model_info['output_shape']}")
        else:
            st.write(f"Error getting model info: {model_info['error']}")
    
    uploaded_file = st.file_uploader(
        "Choose an MRI image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI scan image"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            
            image_gray, image_rgb = preprocess_image(uploaded_file)
            
            if image_gray is not None and image_rgb is not None:
                st.image(image_gray, caption="Uploaded MRI Image", use_container_width=True, clamp=True)
                
                # Prepare input for model
                image_input = image_rgb.reshape(1, 128, 128, 3)
                
                # Make prediction
                with st.spinner("Analyzing image..."):
                    prediction = model.predict(image_input, verbose=0)[0][0]
                
                # Display results
                confidence = prediction if prediction > 0.5 else 1 - prediction
                label = "🔴 Tumor Detected" if prediction > 0.5 else "🟢 No Tumor"
                
                st.markdown(f"### {label}")
                st.markdown(f"**Confidence:** {confidence:.2%}")
                
                # Color-coded progress bar
                if prediction > 0.5:
                    st.error(f"Tumor probability: {prediction:.2%}")
                else:
                    st.success(f"No tumor probability: {(1-prediction):.2%}")
                
                st.progress(float(confidence))
                
        with col2:
            st.subheader("AI Analysis")
            
            if image_gray is not None and image_rgb is not None:
                # Visualization method selector
                viz_method = st.selectbox(
                    "Visualization Method:",
                    ["Auto (Try Multiple)", "Input Gradients", "Occlusion Map", "Simple Attention"]
                )
                
                heatmap = None
                method_used = "None"
                
                with st.spinner("Generating visualization..."):
                    if viz_method == "Auto (Try Multiple)":
                        # Try methods in order of preference
                        
                        # Method 1: Grad-CAM
                        heatmap = make_gradcam_heatmap_robust(image_input, model)
                        if heatmap is not None:
                            method_used = "Grad-CAM"
                        
                        # Method 2: Input gradients
                        if heatmap is None:
                            heatmap = create_input_gradient_heatmap(image_input, model)
                            if heatmap is not None:
                                method_used = "Input Gradients"
                        
                        # Method 3: Simple attention (always works)
                        if heatmap is None:
                            heatmap = create_simple_attention_map(image_input, model, prediction)
                            method_used = "Simple Attention"
                    
                    elif viz_method == "Input Gradients":
                        heatmap = create_input_gradient_heatmap(image_input, model)
                        method_used = "Input Gradients"
                    
                    elif viz_method == "Occlusion Map":
                        with st.info("This may take a moment..."):
                            heatmap = create_occlusion_heatmap(image_input, model)
                        method_used = "Occlusion Map"
                    
                    elif viz_method == "Simple Attention":
                        heatmap = create_simple_attention_map(image_input, model, prediction)
                        method_used = "Simple Attention"
                
                # Display results
                if heatmap is not None:
                    st.success(f"Generated visualization using: {method_used}")
                    
                    overlay, heatmap_colored = create_overlay_image(image_rgb, heatmap)
                    
                    if overlay is not None:
                        st.image(
                            overlay, 
                            caption=f"AI Attention Map ({method_used})", 
                            use_container_width=True
                        )
                        
                        # Show pure heatmap
                        st.image(
                            heatmap_colored, 
                            caption="Pure Attention Map", 
                            use_container_width=True
                        )
                        
                        # Add interpretation
                        max_attention = np.max(heatmap)
                        avg_attention = np.mean(heatmap)
                        
                        st.markdown(f"""
                        **Attention Statistics:**
                        - Max attention: {max_attention:.3f}
                        - Average attention: {avg_attention:.3f}
                        - Method: {method_used}
                        """)
                    else:
                        st.error("Could not create overlay visualization")
                else:
                    st.error("Could not generate any visualization")
        
        # Explanation section
        st.markdown("---")
        st.markdown("""
        ### How it works:
        
        **Model Architecture:**
        - Uses MobileNetV2 backbone pre-trained on ImageNet
        - Fine-tuned for binary classification (tumor vs no tumor)
        - Input: 128x128 grayscale MRI images converted to 3-channel RGB
        
        **Visualization Methods:**
        1. **Grad-CAM**: Shows where gradients are highest in convolutional layers
        2. **Input Gradients**: Highlights input pixels most important for the prediction
        3. **Occlusion Map**: Tests importance by covering different image regions
        4. **Simple Attention**: Edge-based attention weighted by prediction confidence
        
        **Color Coding:**
        - 🔴 **Red/Hot Areas**: High attention regions (important for decision)
        - 🔵 **Blue/Cool Areas**: Low attention regions (less important)
        """)
        
        # Additional info in expander
        with st.expander("Technical Details"):
            st.markdown(f"""
            **Prediction Details:**
            - Raw prediction score: {prediction:.6f}
            - Classification threshold: 0.5
            - Model architecture: Transfer learning with MobileNetV2
            - Input preprocessing: Resize to 128x128, normalize to [0,1], convert grayscale to RGB
            
            **Visualization Method Used:** {method_used}
            """)

if __name__ == "__main__":
    main()
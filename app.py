import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Complete Custom CSS styles with improved visibility
CUSTOM_CSS = """
<style>
    /* Global styling */
    .stApp {
        background-color: #1a1f2c;
    }
    
    /* Main container */
    .main {
        background-color: #1a1f2c;
        padding: 2rem;
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar specific styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #242a38;
    }
    
    .css-1d391kg .custom-card {
        background-color: #2d3545;
        border: 1px solid #3d4658;
    }
    
    /* Card styling */
    .custom-card {
        background-color: #242a38;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
        border: 1px solid #2d3545;
    }
    
    /* Text colors */
    .white-text {
        color: #ffffff !important;
    }
    
    .light-text {
        color: #e2e8f0 !important;
    }
    
    .muted-text {
        color: #94a3b8 !important;
    }
    
    /* Header styling */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* List styling in sidebar */
    .sidebar-list {
        color: #e2e8f0;
        margin-left: 1.5rem;
        line-height: 1.75;
    }
    
    .sidebar-list li {
        margin-bottom: 0.5rem;
    }
    
    /* Alert styling */
    .stAlert > div {
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        color: #ffffff;
    }
    
    /* Success alert */
    .stSuccess > div {
        background-color: #064e3b;
        border: 1px solid #065f46;
    }
    
    /* Error alert */
    .stError > div {
        background-color: #7f1d1d;
        border: 1px solid #991b1b;
    }
    
    /* Info alert */
    .stInfo > div {
        background-color: #1e3a8a;
        border: 1px solid #1e40af;
    }
    
    /* File uploader */
    .stUploader {
        background-color: #2d3545;
        border: 1px dashed #4b5563;
        border-radius: 0.75rem;
    }
    
    /* Image container */
    .stImage {
        background-color: #242a38;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #3d4658;
    }
    
    /* Analysis results */
    .analysis-results {
        background-color: #2d3545;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #3d4658;
        color: #ffffff;
    }
    
    /* Confidence indicator */
    .confidence-indicator {
        background-color: #064e3b;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #065f46;
        text-align: center;
        color: #ffffff;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #242a38;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #2563eb;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #2563eb transparent transparent transparent;
    }
</style>
"""

@st.cache_resource
def load_model():
    """Load the TensorFlow model for currency detection"""
    try:
        model = tf.keras.models.load_model('Model/fake_currrecy_detection_mobilev2.h5')
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error("Failed to load the detection model.")
        return None

def process_image(image_data, target_size=(224, 224)):
    """Process and prepare the image for model prediction"""
    try:
        image = image_data.convert("RGB")
        image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None

def predict_note(model, preprocessed_image):
    """Make predictions on the processed image"""
    predictions = model.predict(preprocessed_image, verbose=0)
    logger.info(f"Raw model output: {predictions}")
    
    class_labels = ['Fake', 'Real']
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index]) * 100
    predicted_label = class_labels[predicted_index]
    
    return predicted_label, confidence

def create_sidebar():
    """Create and populate the sidebar with improved visibility"""
    with st.sidebar:
        st.markdown("""
            <div class="custom-card">
                <h1 class="white-text">💵 CashScan</h1>
                <p class="light-text" style="font-size: 1.1rem;">
                    Advanced currency authentication system
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="custom-card">
                <h3 class="white-text">🔍 How it works</h3>
                <ol class="sidebar-list">
                    <li>Upload an image of the currency note</li>
                    <li>AI model analyzes the note</li>
                    <li>Get instant results</li>
                </ol>
            </div>
            
            <div class="custom-card">
                <h3 class="white-text">📸 Tips for Best Results</h3>
                <ul class="sidebar-list">
                    <li>Use good lighting</li>
                    <li>Capture the entire note</li>
                    <li>Ensure the image is clear</li>
                    <li>Avoid shadows and glare</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

def display_results(predicted_label, confidence):
    """Display the prediction results with improved visibility"""
    result_color = "#10b981" if predicted_label == "Real" else "#ef4444"
    
    st.markdown(f"""
        <div class="analysis-results">
            <h4 class="white-text" style="margin-bottom: 1rem;">Analysis Results</h4>
            <ul style="list-style-type: none; padding: 0;">
                <li style="margin-bottom: 0.5rem;">
                    <strong class="light-text">Prediction:</strong> 
                    <span style="color: {result_color};">
                        {predicted_label} Note
                    </span>
                </li>
                <li>
                    <strong class="light-text">Confidence Score:</strong> 
                    <span class="white-text">{confidence:.2f}%</span>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    st.set_page_config(
        page_title="CashScan - Fake Note Detection",
        page_icon="💵",
        initial_sidebar_state='auto',
        layout="wide"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    create_sidebar()
    
    st.markdown("""
        <h1 class="white-text">🔍 Currency Authentication System</h1>
        <p class="light-text" style="font-size: 1.2rem; margin-bottom: 2rem;">
            Upload an image of a currency note for instant authenticity verification using our advanced AI detection system.
        </p>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner('Loading detection model...'):
        model = load_model()
    
    if model is None:
        st.stop()
    
    # File upload section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if file is not None:
        try:
            image = Image.open(file)
            st.image(image, use_column_width=True, caption="Uploaded Note")
            
            with st.spinner('🔍 Analyzing the note...'):
                processed_image = process_image(image)
                
                if processed_image is not None:
                    predicted_label, confidence = predict_note(model, processed_image)
                    
                    # Display confidence in sidebar
                    st.sidebar.markdown(f"""
                        <div class="confidence-indicator">
                            <h4 class="white-text" style="margin-bottom: 0.5rem;">Analysis Confidence</h4>
                            <p style="font-size: 1.5rem; font-weight: bold;">{confidence:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display result
                    if predicted_label == "Real":
                        st.success("✅ This note appears to be genuine!")
                        st.balloons()
                    else:
                        st.error("⚠️ Warning! This note is detected as fake. Please verify with your bank.")
                    
                    # Detailed analysis
                    with st.expander("📊 See detailed analysis"):
                        display_results(predicted_label, confidence)
                else:
                    st.error("Failed to process the image.")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error processing upload: {str(e)}")
    else:
        st.info("👆 Please upload an image of a currency note to begin analysis")

if __name__ == "__main__":
    main()
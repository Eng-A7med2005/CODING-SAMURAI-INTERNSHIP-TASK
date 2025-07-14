import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess, decode_predictions as resnet_decode
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess, decode_predictions as vgg_decode
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess, decode_predictions as mobile_decode
import numpy as np
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(page_title="AI Image Classifier", page_icon="🎯", layout="wide")

st.markdown("""
<style>
@media (prefers-color-scheme: dark) {
    .main-title { color: #e2e8f0; }
    .subtitle { color: #94a3b8; }
    .upload-section { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); }
    .prediction-card { background: #1e293b; color: #e2e8f0; }
    .prediction-item { background: #334155; }
    .prediction-class { color: #e2e8f0; }
}

@media (prefers-color-scheme: light) {
    .main-title { color: #1e293b; }
    .subtitle { color: #64748b; }
    .upload-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .prediction-card { background: white; color: #1e293b; }
    .prediction-item { background: #f8fafc; }
    .prediction-class { color: #1e293b; }
}

.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 1rem;
}

.subtitle {
    font-size: 1.1rem;
    text-align: center;
    margin-bottom: 2rem;
}

.upload-section {
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    color: white;
}

.prediction-card {
    padding: 1.5rem;
    border-radius: 15px;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.prediction-item {
    display: flex;
    justify-content: space-between;
    padding: 0.8rem;
    margin: 0.5rem 0;
    border-radius: 8px;
    border-left: 3px solid #667eea;
}

.prediction-class {
    font-weight: 600;
}

.prediction-confidence {
    background: #10b981;
    color: white;
    padding: 0.2rem 0.6rem;
    border-radius: 15px;
    font-size: 0.9rem;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🎯 AI Image Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and discover what AI sees</p>', unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name):
    models = {
        "ResNet50": (ResNet50(weights='imagenet'), resnet_preprocess, resnet_decode),
        "VGG16": (VGG16(weights='imagenet'), vgg_preprocess, vgg_decode),
        "MobileNetV2": (MobileNetV2(weights='imagenet'), mobile_preprocess, mobile_decode)
    }
    return models[model_name]

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    return np.expand_dims(img_array, axis=0)

def create_chart(predictions, top_k=5):
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_predictions = predictions[0][top_indices]
    
    fig = go.Figure(data=[go.Bar(
        x=top_predictions,
        y=[f"Prediction {i+1}" for i in range(len(top_predictions))],
        orientation='h',
        marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4ecdc4'][:len(top_predictions)],
        text=[f'{pred:.1%}' for pred in top_predictions],
        textposition='auto',
    )])
    
    fig.update_layout(
        title='Confidence Levels',
        xaxis_title='Confidence',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

model_choice = st.sidebar.selectbox("Select Model", ["ResNet50", "VGG16", "MobileNetV2"])
top_k = st.sidebar.slider("Top Predictions", 1, 10, 5)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.1)

with st.spinner(f"Loading {model_choice}..."):
    model, preprocess_func, decode_func = load_model(model_choice)
    st.sidebar.success(f"✅ {model_choice} ready!")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<h3>📸 Upload Your Image</h3>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png', 'bmp'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    if uploaded_file:
        if st.button("🎯 Classify Image"):
            with st.spinner("Analyzing..."):
                try:
                    img_array = preprocess_image(image)
                    img_array = preprocess_func(img_array)
                    
                    predictions = model.predict(img_array)
                    decoded_predictions = decode_func(predictions, top=top_k)
                    
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.markdown('<h3>🎯 Results</h3>', unsafe_allow_html=True)
                    
                    for i, (_, class_name, confidence) in enumerate(decoded_predictions[0]):
                        if confidence > confidence_threshold:
                            st.markdown(f'''
                            <div class="prediction-item">
                                <span class="prediction-class">{i+1}. {class_name}</span>
                                <span class="prediction-confidence">{confidence:.1%}</span>
                            </div>
                            ''', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    confidences = [pred[2] for pred in decoded_predictions[0]]
                    fig = create_chart(np.array([confidences]), top_k)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("👆 Upload an image to start")

st.markdown("---")
st.markdown("🚀 Powered by TensorFlow & Streamlit")
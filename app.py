import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Klasifikasi Sampah AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 20px;
        background-color: #2E8B57;
        text-align: center;
        line-height: 20px;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_classification_model():
    try:
        model_path = './trash_classification_model.h5'
        if os.path.exists(model_path):
            model = load_model(model_path)
            return model
        else:
            st.error("Model file tidak ditemukan. Pastikan model sudah dilatih.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Class names
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Class descriptions and recycling tips
CLASS_INFO = {
    'cardboard': {
        'description': 'Kardus/Karton',
        'tips': 'Dapat didaur ulang menjadi kertas baru. Pastikan kardus bersih dan kering.',
        'color': '#8B4513'
    },
    'glass': {
        'description': 'Kaca',
        'tips': 'Dapat didaur ulang berkali-kali tanpa kehilangan kualitas. Pisahkan berdasarkan warna.',
        'color': '#4169E1'
    },
    'metal': {
        'description': 'Logam',
        'tips': 'Aluminium dan besi dapat didaur ulang. Bersihkan dari sisa makanan sebelum didaur ulang.',
        'color': '#708090'
    },
    'paper': {
        'description': 'Kertas',
        'tips': 'Dapat didaur ulang menjadi kertas baru. Hindari kertas yang terlalu kotor atau berminyak.',
        'color': '#DEB887'
    },
    'plastic': {
        'description': 'Plastik',
        'tips': 'Periksa kode daur ulang pada kemasan. Bersihkan sebelum didaur ulang.',
        'color': '#FF6347'
    },
    'trash': {
        'description': 'Sampah Umum',
        'tips': 'Sampah yang tidak dapat didaur ulang. Buang ke tempat sampah biasa.',
        'color': '#696969'
    }
}

def preprocess_image(img):
    """Preprocess image for prediction"""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(model, img_array):
    """Make prediction on preprocessed image"""
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = CLASS_NAMES[predicted_class_idx]
    return predicted_class, confidence, predictions[0]

def main():
    # Header
    st.markdown('<h1 class="main-header">♻️ Klasifikasi Sampah AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload gambar sampah untuk mengetahui jenis dan cara daur ulangnya</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_classification_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Informasi Aplikasi")
        st.write("""
        Aplikasi ini menggunakan AI untuk mengklasifikasikan sampah ke dalam 6 kategori:
        - 📦 Kardus/Karton
        - 🪟 Kaca
        - 🔩 Logam
        - 📄 Kertas
        - 🥤 Plastik
        - 🗑️ Sampah Umum
        """)
        
        st.header("🎯 Cara Penggunaan")
        st.write("""
        1. Upload gambar sampah
        2. Tunggu hasil prediksi
        3. Lihat tips daur ulang
        """)
        
        st.header("📊 Akurasi Model")
        st.info("Model dilatih dengan dataset TrashNet dan mencapai akurasi ~80%")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Gambar")
        uploaded_file = st.file_uploader(
            "Pilih gambar sampah...",
            type=['jpg', 'jpeg', 'png'],
            help="Format yang didukung: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption="Gambar yang diupload", use_column_width=True)
            
            # Predict button
            if st.button("🔍 Klasifikasi Sampah", type="primary"):
                with st.spinner("Menganalisis gambar..."):
                    # Preprocess and predict
                    img_array = preprocess_image(img)
                    predicted_class, confidence, all_predictions = predict_image(model, img_array)
                    
                    # Store results in session state
                    st.session_state.prediction_results = {
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'all_predictions': all_predictions
                    }
    
    with col2:
        st.header("🎯 Hasil Prediksi")
        
        if hasattr(st.session_state, 'prediction_results'):
            results = st.session_state.prediction_results
            predicted_class = results['predicted_class']
            confidence = results['confidence']
            all_predictions = results['all_predictions']
            
            # Main prediction
            class_info = CLASS_INFO[predicted_class]
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="color: {class_info['color']};">
                    Jenis Sampah: {class_info['description']}
                </h3>
                <p><strong>Tingkat Kepercayaan: {confidence:.2%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence*100:.1f}%;">
                    {confidence:.2%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Recycling tips
            st.markdown("### 💡 Tips Daur Ulang")
            st.info(class_info['tips'])
            
            # All predictions
            st.markdown("### 📊 Semua Prediksi")
            for i, class_name in enumerate(CLASS_NAMES):
                prob = all_predictions[i]
                st.write(f"**{CLASS_INFO[class_name]['description']}**: {prob:.2%}")
                st.progress(prob)
        
        else:
            st.info("Upload gambar dan klik 'Klasifikasi Sampah' untuk melihat hasil prediksi.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🌱 Mari bersama-sama menjaga lingkungan dengan daur ulang yang tepat! 🌱</p>
        <p>Dibuat dengan ❤️ menggunakan Streamlit dan TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


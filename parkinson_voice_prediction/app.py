import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
from prediction.predictor import Predictor

@st.cache_resource
def load_predictor():
    return Predictor()

def main():
    st.set_page_config(
        page_title="Parkinson's Speech Analysis", 
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.5rem;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #6B7280;
            text-align: center;
            margin-bottom: 2rem;
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-top: 1rem;
        }
        .healthy-box {
            background-color: #D1FAE5;
            border: 2px solid #10B981;
            color: #065F46;
        }
        .parkinson-box {
            background-color: #FEE2E2;
            border: 2px solid #EF4444;
            color: #991B1B;
        }
        .info-text {
            font-size: 1.1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #F3F4F6;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #E0E7FF;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3750/3750569.png", width=100)
        st.title("🎙️ Analysis Options")
        st.markdown("---")
        
        st.subheader("Prediction Settings")
        pred_type = st.radio(
            "Select Input Method:", 
            ["🎧 Audio Upload", "📊 Clinical Features", "✍️ Spiral Drawing Upload"],
            help="Choose how you want to provide data for the analysis."
        )
        
        st.markdown("---")
        st.subheader("Model Selection")
        use_wav2vec = st.toggle(
            "Enable Wav2Vec 2.0 (Deep Learning)", 
            value=False,
            help="Use advanced audio feature embeddings from HuggingFace's Wav2Vec 2.0 model (Requires Audio Upload)."
        )
        
        if use_wav2vec and pred_type != "🎧 Audio Upload":
            st.error("⚠️ Wav2Vec 2.0 only works with Audio Upload. Please switch the input method.")
            
        st.markdown("---")
        st.info("💡 **Tip:** Audio upload extracts acoustic features automatically or uses Deep Learning embeddings when enabled.")

    # Main Content
    st.markdown('<p class="main-title">Parkinson\'s Disease Speech Analysis Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced acoustic analysis for early detection using Multiclass Machine Learning & Deep Learning approaches.</p>', unsafe_allow_html=True)
    
    predictor = load_predictor()
    
    tab1, tab2, tab3 = st.tabs(["🔍 Make a Prediction", "📈 Model Analytics", "ℹ️ About the Architecture"])
    
    with tab1:
        st.write("### New Analysis")
        st.write("Please provide the required input below and click 'Analyze'.")
        
        if pred_type == "📊 Clinical Features":
            st.info("Provide the extracted clinical features from Praat or similar acoustic analysis software.")
            try:
                selected_features = joblib.load("models/selected_features.pkl")
                st.write(f"**Required:** {len(selected_features)} acoustic features.")
                
                # Use a slightly better text area with placeholder
                feature_input = st.text_area(
                    "Paste comma-separated feature values here (All biomedical voice metrics):", 
                    placeholder=f"e.g., {'1.23, 0.45, ' * (len(selected_features)//2)}...",
                    height=150
                )
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    analyze_btn = st.button("🔍 Analyze Features", use_container_width=True, type="primary")
                
                if analyze_btn:
                    if use_wav2vec:
                        st.error("Cannot use Wav2Vec with clinical features. Disable the toggle in the sidebar.", icon="🚨")
                    elif not feature_input:
                        st.warning("Please paste the feature values first.", icon="⚠️")
                    else:
                        with st.spinner("Analyzing clinical features..."):
                            try:
                                features_list = [float(x.strip()) for x in feature_input.split(",")]
                                if len(features_list) != len(selected_features):
                                    st.error(f"Mismatch! Expected {len(selected_features)} features, but got {len(features_list)}.", icon="🚨")
                                else:
                                    result, prob = predictor.predict_from_features(features_list, use_wav2vec=False)
                                    
                                    st.markdown("---")
                                    st.write("### Analysis Results")
                                    
                                    if result == "Parkinson Detected":
                                        st.markdown(f"""
                                            <div class="prediction-box parkinson-box">
                                                <h1>🚨 {result}</h1>
                                                <p style="font-size: 1.2rem;">Model Confidence: <strong>{prob*100:.1f}%</strong></p>
                                                <p>The analysis indicates acoustic patterns consistent with Parkinson's Disease.</p>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                            <div class="prediction-box healthy-box">
                                                <h1>✅ {result}</h1>
                                                <p style="font-size: 1.2rem;">Model Confidence: <strong>{prob*100:.1f}%</strong></p>
                                                <p>No significant acoustic markers associated with Parkinson's Disease were detected.</p>
                                            </div>
                                        """, unsafe_allow_html=True)
                            except ValueError:
                                st.error("Invalid format. Please ensure all values are numbers separated by commas.", icon="🚨")
                            except Exception as e:
                                st.error(f"Error during analysis: {e}", icon="🚨")
            except FileNotFoundError:
                st.warning("⚠️ Feature models are not loaded. Please run the training pipeline first.")
                
        elif pred_type == "✍️ Spiral Drawing Upload":
            st.info("Provide an image of a spiral drawing for CNN-based analysis.")
            
            uploaded_image = st.file_uploader("Upload Spiral Drawing (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])
            
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Spiral Drawing", width=300)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    analyze_img_btn = st.button("🔍 Analyze Drawing", use_container_width=True, type="primary")
                    
                if analyze_img_btn:
                    os.makedirs("dataset/temp_uploads", exist_ok=True)
                    # Use a safely constructed filename or the original name
                    file_name = uploaded_image.name if hasattr(uploaded_image, "name") and uploaded_image.name else "uploaded_spiral.png"
                    file_path = f"dataset/temp_uploads/{file_name}"
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_image.getbuffer())
                        
                    with st.spinner("Analyzing spiral drawing with CNN..."):
                        try:
                            result, prob = predictor.predict_from_spiral_image(file_path)
                            
                            st.markdown("---")
                            st.write("### Analysis Results")
                            st.write("*Analysis performed using: PyTorch ResNet Image Classifier*")
                            
                            if result == "Parkinson Detected":
                                st.markdown(f'''
                                    <div class="prediction-box parkinson-box">
                                        <h1>🚨 {result}</h1>
                                        <p style="font-size: 1.2rem;">Model Confidence: <strong>{prob*100:.1f}%</strong></p>
                                        <p>The model detected traits in the spiral drawing often associated with Parkinson's Disease.</p>
                                    </div>
                                ''', unsafe_allow_html=True)
                            else:
                                st.markdown(f'''
                                    <div class="prediction-box healthy-box">
                                        <h1>✅ {result}</h1>
                                        <p style="font-size: 1.2rem;">Model Confidence: <strong>{prob*100:.1f}%</strong></p>
                                        <p>The spiral drawing appears typical and healthy.</p>
                                    </div>
                                ''', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Prediction error: {e}", icon="🚨")

        else: # Audio Upload or Record
            st.info("Provide a voice recording for automated feature extraction and analysis.")
            
            st.markdown("""
            <div style="background-color: #F3F4F6; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #1E3A8A;">
                <p style="margin-bottom: 5px; font-weight: bold; color: #1E3A8A;">🗣️ Please read the following phrase clearly into the microphone:</p>
                <p style="font-style: italic; color: #4B5563; margin-bottom: 0;">"Hello, today is a beautiful day. I am feeling great and I am ready to record my voice."</p>
            </div>
            """, unsafe_allow_html=True)
            
            audio_source = st.radio("Choose Audio Source:", ["📁 Upload File", "🎤 Record Audio"], horizontal=True)
            uploaded_file = None
            
            if audio_source == "📁 Upload File":
                uploaded_file = st.file_uploader("Drop an audio file here", type=["wav", "mp3", "webm"])
            else:
                st.write("Click the microphone below to record directly from your browser.")
                uploaded_file = st.audio_input("Record Voice Sample")
            
            if uploaded_file is not None:
                st.audio(uploaded_file, format='audio/wav')
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    analyze_btn = st.button("🔍 Analyze Audio", use_container_width=True, type="primary")
                    
                if analyze_btn:
                    os.makedirs("dataset/temp_uploads", exist_ok=True)
                    # For recordings, name might be 'audio_record.wav'. Adding index or timestamp is safer.
                    file_name = uploaded_file.name if hasattr(uploaded_file, "name") and uploaded_file.name else "recorded_audio.wav"
                    file_path = f"dataset/temp_uploads/{file_name}"
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    with st.spinner("Extracting features and analyzing audio... (This may take a moment)"):
                        try:
                            result, prob = predictor.predict_from_audio(file_path, use_wav2vec=use_wav2vec)
                            
                            st.markdown("---")
                            st.write("### Analysis Results")
                            
                            model_used = "Wav2Vec 2.0 (Deep Learning)" if use_wav2vec else "Classical ML Pipeline"
                            st.write(f"*Analysis performed using: {model_used}*")
                            
                            if result == "Parkinson Detected":
                                st.markdown(f"""
                                    <div class="prediction-box parkinson-box">
                                        <h1>🚨 {result}</h1>
                                        <p style="font-size: 1.2rem;">Model Confidence: <strong>{prob*100:.1f}%</strong></p>
                                        <p>The model detected voice variations often associated with Parkinson's Disease.</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div class="prediction-box healthy-box">
                                        <h1>✅ {result}</h1>
                                        <p style="font-size: 1.2rem;">Model Confidence: <strong>{(1-prob)*100 if prob < 0.5 else prob*100:.1f}%</strong></p>
                                        <p>The voice characteristics appear typical and healthy.</p>
                                    </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Prediction error: {e}", icon="🚨")
                            
    with tab2:
        st.write("### Comprehensive Model Evaluation")
        st.write("This dashboard displays the training metrics and visualizations for all models executed in the pipeline.")
        
        if os.path.exists("models/evaluation_metrics.csv"):
            metrics_df = pd.read_csv("models/evaluation_metrics.csv")
            
            # Highlight max values in the dataframe
            st.dataframe(
                metrics_df.style.highlight_max(axis=0, subset=['accuracy', 'precision', 'recall', 'f1'], color='#10B981')
                                .format({col: "{:.4f}" for col in ['accuracy', 'precision', 'recall', 'f1']}),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")
            
            # Display charts in a cleaner grid
            st.write("### Performance Visualizations")
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists("frontend/static/plots/model_comparison.png"):
                    st.image(Image.open("frontend/static/plots/model_comparison.png"), use_container_width=True)
                else:
                    st.info("Model comparison chart not available.")
                    
                if os.path.exists("frontend/static/plots/roc_curves.png"):
                    st.image(Image.open("frontend/static/plots/roc_curves.png"), use_container_width=True)
                else:
                    st.info("ROC curves not available.")
                    
            with col2:
                if os.path.exists("frontend/static/plots/cm_random_forest.png"):
                    st.image(Image.open("frontend/static/plots/cm_random_forest.png"), use_container_width=True)
                else:
                    st.info("Confusion matrix not available.")
                    
                if os.path.exists("frontend/static/plots/feature_importance.png"):
                    st.image(Image.open("frontend/static/plots/feature_importance.png"), use_container_width=True)
                else:
                    st.info("Feature importance chart not available.")
        else:
            st.info("📊 No evaluation data found. Run the training pipeline first.", icon="ℹ️")

    with tab3:
        st.write("### Architecture Overview")
        
        st.write("""
        This application is an implementation of a **Multiclass Machine Learning Approach** for the early detection of Parkinson's Disease utilizing voice biomarkers.
        
        #### Core Components:
        1. **Data Processing Pipeline**:
           - Utilizes the UCI Machine Learning Repository *parkinsons* dataset.
           - Employs **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalances.
           - Standardizes numerical variables using `StandardScaler`.
           
        2. **Classical Machine Learning Models**:
           - **KSVM** (Kernel Support Vector Machine)
           - **Random Forest** (RF)
           - **Decision Tree** (DT)
           - **K-Nearest Neighbors** (KNN)
           *All models are fine-tuned using `RandomizedSearchCV`.*
           
        3. **Deep Learning Implementations**:
           - **FNN**: A Feed Forward Neural Network acting on clinical acoustic features.
           - **Wav2Vec 2.0**: A custom integration using HuggingFace's pre-trained model to extract high-dimensional semantic voice embeddings directly from raw audio, feeding into a Neural Classification Head.
        """)
        
if __name__ == "__main__":
    main()

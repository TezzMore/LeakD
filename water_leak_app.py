import streamlit as st
import torch
import pandas as pd
import numpy as np
import sys
import os
from torch.utils.data import DataLoader
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Water Leak Detection AI",
    page_icon="💧",
    layout="wide"
)

# Add model path (adjust this to your model location)
sys.path.append('./model')  # Update this path to where your GGNN.py file is

try:
    from GGNN import GGNNModel
    MODEL_LOADED = True
except ImportError:
    MODEL_LOADED = False
    st.error("⚠️ GGNN model file not found! Please ensure GGNN.py is in the correct directory.")

# Title and description
st.title("💧 Water Leak Detection AI System")
st.markdown("### Upload your water network dataset and get instant leak predictions!")
st.markdown("**Model Performance**: 83.1% accuracy on unseen networks | **Training**: Mixed-dataset approach")

# Sidebar for model information
st.sidebar.header("🔧 Model Information")
st.sidebar.info("""
**Model Type**: Graph Neural Network (GNN)
**Training Data**: 8 water networks  
**Validation Accuracy**: 89.05%
**Test Accuracy**: 83.1%
**Generalization Gap**: 5.9%
""")

# Function to load your trained model
@st.cache_resource
def load_model():
    """Load the trained water leak detection model"""
    try:
        # Update this path to your actual model file
        model_path = "mixed_model_2906_2012.pth"  # Change this to your model path
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None, None
            
        # Load checkpoint
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Recreate model
        arch = ckpt['arch']
        model = GGNNModel(arch['in'], arch['hidden'], arch['win'])
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        
        return model, ckpt
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to make predictions
def predict_leaks(model, dataset_path):
    """Make leak predictions on uploaded dataset"""
    try:
        # Load the dataset
        data = torch.load(os.path.join(dataset_path, 'dataset.pt'), weights_only=False)
        adj = torch.load(os.path.join(dataset_path, 'adj_matrices.pt'), weights_only=False)
        
        # Create data loader
        loader = DataLoader(data, batch_size=100, shuffle=False)
        
        predictions = []
        true_labels = []
        confidences = []
        
        with torch.no_grad():
            for X, y, i in loader:
                out = model(X, adj[i])
                probs = torch.exp(out)
                pred = probs.argmax(-1)
                conf = probs.max(-1)[0]
                
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(y.cpu().numpy())
                confidences.extend(conf.cpu().numpy())
        
        # Calculate accuracy
        accuracy = (np.array(predictions) == np.array(true_labels)).mean() * 100
        
        return predictions, true_labels, confidences, accuracy
        
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None, None, None, None

# Main app interface
def main():
    if not MODEL_LOADED:
        st.stop()
    
    # Load model
    model, ckpt = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check the model path.")
        st.stop()
    
    # Success message
    st.success(f"✅ Model loaded successfully! Validation accuracy: {ckpt['val_acc']:.2f}%")
    
    # File upload section
    st.header("📁 Upload Water Network Dataset")
    
    uploaded_files = st.file_uploader(
        "Upload dataset files (dataset.pt and adj_matrices.pt)",
        accept_multiple_files=True,
        type=['pt'],
        help="Upload both dataset.pt and adj_matrices.pt files from your water network data"
    )
    
    if uploaded_files and len(uploaded_files) >= 2:
        # Create temporary directory for uploaded files
        temp_dir = "temp_dataset"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded files
        dataset_file = None
        adj_file = None
        
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            if file.name == 'dataset.pt':
                dataset_file = file_path
            elif file.name == 'adj_matrices.pt':
                adj_file = file_path
        
        if dataset_file and adj_file:
            st.success("✅ Dataset files uploaded successfully!")
            
            # Predict button
            if st.button("🔍 Analyze Dataset for Leaks", type="primary"):
                with st.spinner("🧠 AI is analyzing your water network..."):
                    predictions, true_labels, confidences, accuracy = predict_leaks(model, temp_dir)
                
                if predictions is not None:
                    # Display results
                    st.header("🎯 Leak Detection Results")
                    
                    # Accuracy metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 Overall Accuracy", f"{accuracy:.1f}%")
                    with col2:
                        st.metric("📊 Total Samples", len(predictions))
                    with col3:
                        avg_confidence = np.mean(confidences) * 100
                        st.metric("🔍 Avg Confidence", f"{avg_confidence:.1f}%")
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'Sample_ID': range(len(predictions)),
                        'Predicted_Location': predictions,
                        'True_Location': true_labels,
                        'Confidence': [f"{c*100:.1f}%" for c in confidences],
                        'Correct': predictions == np.array(true_labels)
                    })
                    
                    # Show detailed results
                    st.subheader("📋 Detailed Predictions")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Visualization
                    st.subheader("📊 Results Visualization")
                    
                    # Accuracy by location
                    location_accuracy = results_df.groupby('Predicted_Location')['Correct'].mean() * 100
                    
                    fig = px.bar(
                        x=location_accuracy.index,
                        y=location_accuracy.values,
                        title="Accuracy by Predicted Leak Location",
                        labels={'x': 'Leak Location', 'y': 'Accuracy (%)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Confidence distribution
                    fig2 = px.histogram(
                        confidences,
                        title="Confidence Score Distribution",
                        labels={'value': 'Confidence Score', 'count': 'Number of Predictions'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Results as CSV",
                        data=csv,
                        file_name="leak_detection_results.csv",
                        mime="text/csv"
                    )
                    
                    # Interpretation
                    st.subheader("🧠 AI Analysis Summary")
                    
                    if accuracy > 80:
                        st.success(f"🎉 Excellent performance! The model achieved {accuracy:.1f}% accuracy on your dataset.")
                    elif accuracy > 70:
                        st.info(f"✅ Good performance! The model achieved {accuracy:.1f}% accuracy on your dataset.")
                    else:
                        st.warning(f"⚠️ The model achieved {accuracy:.1f}% accuracy. This dataset may have different characteristics than the training data.")
                    
                    # Clean up temporary files
                    import shutil
                    shutil.rmtree(temp_dir)
        else:
            st.warning("⚠️ Please upload both dataset.pt and adj_matrices.pt files")
    
    # Instructions section
    with st.expander("📖 How to Use This Application"):
        st.markdown("""
        **Step 1**: Prepare your water network dataset files:
        - `dataset.pt`: Contains the sensor data and leak labels
        - `adj_matrices.pt`: Contains the network topology information
        
        **Step 2**: Upload both files using the file uploader above
        
        **Step 3**: Click "Analyze Dataset for Leaks" to run the AI model
        
        **Step 4**: Review the results:
        - Overall accuracy shows how well the model performed
        - Detailed predictions show each leak location prediction
        - Visualizations help understand the model's performance
        
        **Step 5**: Download the results for further analysis
        """)

if __name__ == "__main__":
    main()

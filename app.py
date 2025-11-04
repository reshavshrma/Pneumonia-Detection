import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import time

# 2. Define the CNN model class
class CNNModel(nn.Module):
    def __init__(self, image_shape):
        super(CNNModel, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(image_shape[0], 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn2_2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.bn3_2 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten and FC with Global Average Pooling and a single Linear layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_final = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn1_2(x)
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = F.relu(self.conv4(x))
        x = self.bn2_2(x)
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = self.bn3(x)
        x = F.relu(self.conv6(x))
        x = self.bn3_2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_final(x)
        return x

# 3. Define a function to load the trained model
@st.cache_resource
def load_model(model_path):
    model = CNNModel(image_shape=(3, 224, 224))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 4. Define a function to preprocess an uploaded image
def preprocess_image(image_file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(image_file).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# 5. Define a function to make a prediction
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        prediction = torch.sigmoid(outputs) > 0.5
        confidence = torch.sigmoid(outputs).item()
    return prediction.item(), confidence

# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Enhanced Custom Styling
# --------------------------------------------------
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
    }
    
    /* Header Card */
    .header-card {
        background: #E6FFF5;
        padding: 3rem 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInDown 0.8s ease-out;
        margin-top: 1.5rem; /*UPPER MARGIN*/
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #64748b;
        font-weight: 400;
    }
    
    /* Upload Section */
    .upload-card {
        background: #E6FFF5;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Result Cards */
    .result-positive {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(245,87,108,0.4);
        animation: slideIn 0.5s ease-out;
    }
    
    .result-negative {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(79,172,254,0.4);
        animation: slideIn 0.5s ease-out;
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .result-text {
        font-size: 1.1rem;
        opacity: 0.95;
        line-height: 1.6;
    }
    
    /* Confidence Badge */
    .confidence-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: 600;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg, 
    [data-testid="stSidebar"] .css-1cypcdb {
        color: white;
    }
    
    .sidebar-content {
        color: white;
        padding: 1rem;
    }
    
    .info-card {
        background: rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.6);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Image Container */
    .image-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
    }
    
    /* Progress Bar Custom */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: white;
        padding: 2rem;
        margin-top: 3rem;
        font-size: 0.9rem;
        opacity: 0.8;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
# --------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-content">
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="font-size: 2rem;">🫁</h1>
                <h2 style="margin-top: 0.5rem;">About This App</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-card">
            <h3 style="margin-top: 0;">🔬 Technology</h3>
            <p>Powered by Deep Learning CNN architecture trained on thousands of chest X-ray images.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-card">
            <h3 style="margin-top: 0;">📊 Accuracy</h3>
            <p>Our model achieves high accuracy in detecting pneumonia patterns in chest radiographs.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-card">
            <h3 style="margin-top: 0;">⚠️ Disclaimer</h3>
            <p>This tool is for educational purposes. Always consult healthcare professionals for medical diagnosis.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**👨‍💻 Developed by Reshav Sharma and Pratik Parhad**")
    st.markdown("*Powered by PyTorch & Streamlit*")

# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.markdown("""
    <div class="header-card">
        <h1 class="main-title">AI-Powered Pneumonia Detection</h1>
        <p class="subtitle">Upload a chest X-ray image for instant analysis using deep learning</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload X-Ray Image")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded X-Ray Image", use_container_width=True)

with col2:
    if uploaded_file is not None:
        # Progress and spinner
        with st.spinner("🔍 Analyzing X-ray image..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Model and prediction logic
            model_path = 'best_model.pth'
            model = load_model(model_path)
            image_tensor = preprocess_image(uploaded_file)
            prediction, confidence = predict(model, image_tensor)
        
        st.balloons()
        
        # Show prediction result
        if prediction == 1:
            st.markdown(f"""
                <div class="result-positive">
                    <div class="result-title">⚠️ Pneumonia Detected</div>
                    <div class="result-text">
                        The AI model has identified patterns consistent with pneumonia in the uploaded X-ray image.
                        <br><br>
                        <strong>⚕️ Recommended Action:</strong><br>
                        Please consult with a healthcare professional immediately for proper diagnosis and treatment.
                    </div>
                    <div class="confidence-badge">
                        Confidence: {confidence*100:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-negative">
                    <div class="result-title">✅ Normal Lungs</div>
                    <div class="result-text">
                        The AI model did not detect significant patterns of pneumonia in the uploaded X-ray image.
                        <br><br>
                        <strong>🌟 Keep Healthy:</strong><br>
                        No concerning signs detected. Continue maintaining good health practices!
                    </div>
                    <div class="confidence-badge">
                        Confidence: {(1-confidence)*100:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="upload-card" style="text-align: center; padding: 3rem;">
                <h2 style="color: #667eea;">👈 Upload an X-Ray to Begin</h2>
                <p style="color: #64748b; font-size: 1.1rem; margin-top: 1rem;">
                    Select a chest X-ray image from your device to start the AI analysis
                </p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>© 2025 AI Pneumonia Detection System | Educational Purpose Only</p>
        <p>Made with ❤️ using PyTorch & Streamlit</p>
    </div>
""", unsafe_allow_html=True)
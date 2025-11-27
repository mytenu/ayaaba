import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Set page config
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

# Define the same model architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256*3*3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load('best_fruit_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Define class names
CLASS_NAMES = [
    'anthracnose',
    'gumosis',
    'healthy_cashew',
    'healthy_tomato',
    'leaf_blight',
    'leaf_curl',
    'leaf_miner',
    'red_rust',
    'septorial_leaf_spot',
    'verticulium_wilt'
]

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image, model, device):
    image_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()

# Format class name for display
def format_class_name(class_name):
    return class_name.replace('_', ' ').title()

# Streamlit UI
st.title("🌿 Plant Disease Classifier")
st.write("Upload an image to classify plant diseases in cashew and tomato plants")

# Load model
try:
    model, device = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    with st.spinner('🔍 Analyzing image...'):
        predicted_class, confidence, probabilities = predict(image, model, device)
    
    with col2:
        st.subheader("Prediction Results")
        
        # Display prediction with appropriate emoji
        prediction_text = format_class_name(CLASS_NAMES[predicted_class])
        if 'healthy' in CLASS_NAMES[predicted_class]:
            st.success(f"✅ {prediction_text}")
        else:
            st.warning(f"⚠️ {prediction_text}")
        
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        
        # Show progress bar for confidence
        st.progress(confidence)
    
    # Show all class probabilities
    st.subheader("📊 All Class Probabilities")
    prob_dict = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    
    for class_name, prob in sorted_probs.items():
        formatted_name = format_class_name(class_name)
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.progress(prob)
        with col_b:
            st.write(f"{prob * 100:.2f}%")
        st.caption(formatted_name)

# Sidebar information
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This app uses a PST-CNN to classify "
    "plant diseases in cashew and tomato plants. Upload an image and "
    "get instant predictions!"
)

st.sidebar.subheader("🏷️ Supported Classes")
st.sidebar.write("**Diseases:**")
for class_name in CLASS_NAMES:
    if 'healthy' not in class_name:
        st.sidebar.write(f"• {format_class_name(class_name)}")

st.sidebar.write("\n**Healthy:**")
for class_name in CLASS_NAMES:
    if 'healthy' in class_name:
        st.sidebar.write(f"• {format_class_name(class_name)}")

st.sidebar.divider()
st.sidebar.write(f"**Device:** {device}")
st.sidebar.write(f"**Total Classes:** {len(CLASS_NAMES)}")

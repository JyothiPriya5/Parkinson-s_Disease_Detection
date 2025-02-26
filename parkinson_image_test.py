import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18
import functions as pds

# Define the model and load the saved weights
CNN_backbone = resnet18(weights='ResNet18_Weights.DEFAULT')
CNN_backbone.fc = torch.nn.Flatten()
model = pds.PrototypicalNetworks(CNN_backbone)

# Load the model weights
model.load_state_dict(torch.load('C:/Users/perso/Downloads/Parkinson_Disease_Detection/parkinson_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Transform for the input image
resize_transform = transforms.Compose([
    transforms.CenterCrop([300, 600]),  # Adjust to match your training dimensions
    transforms.ToTensor(),
])

# Load and preprocess the image
img_path = 'C:/Users/perso/Downloads/Parkinson_Disease_Detection/dataset/train/healthy/V01HE02.png'
image = Image.open(img_path)
input_tensor = resize_transform(image).unsqueeze(0)  # Add batch dimension

# Create dummy support images and labels for prediction compatibility
support_images = input_tensor.clone()  # Using the same image as a dummy support set
support_labels = torch.tensor([0])     # Dummy label, adjust if needed
query_images = input_tensor

# Predict class
with torch.no_grad():
    output = model(support_images, support_labels, query_images)
    predicted_class = output.argmax(dim=1).item()

print(f"Predicted class: {'Parkinson' if predicted_class < 0.5 else 'Healthy'}")

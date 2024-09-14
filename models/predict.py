import torch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torchvision import models
from src.utils import download_images
from src.constants import allowed_units

# Define the image preprocessing function
def preprocess_image(image_path):
    import cv2
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    img = img / 255.0  # Normalize pixel values
    img = torch.tensor(img).permute(2, 0, 1)  # Convert to tensor and reorder dimensions (channels, height, width)
    return img.unsqueeze(0)  # Add batch dimension

# Define the function to extract image features using ResNet
def extract_image_features(image_path):
    model = models.resnet50(pretrained=True)
    model.eval()  # Set the model to evaluation mode (no training)
    
    # Preprocess the image and extract features
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        features = model(img_tensor)
    return features

# Define the multimodal model that combines image features and entity name
class MultimodalModel(torch.nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        self.fc_image = torch.nn.Linear(1000, 512)  # Image features from ResNet
        self.fc_entity = torch.nn.Linear(len(encoder.categories_[0]), 512)  # Encoded entity name
        self.fc_value = torch.nn.Linear(1024, 1)  # Predict entity value (regression)
        self.fc_unit = torch.nn.Linear(1024, len(allowed_units))  # Predict unit (classification)

    def forward(self, image_features, entity_name_features):
        image_out = self.fc_image(image_features)
        entity_out = self.fc_entity(entity_name_features)
        combined = torch.cat((image_out, entity_out), dim=1)
        value_out = self.fc_value(combined)
        unit_out = self.fc_unit(combined)
        return value_out, unit_out

# Load the test dataset
test_df = pd.read_csv('dataset/test.csv')

# Download the test images
download_images(test_df['image_link'].tolist(), './test_images/')

# Load the trained model
model = MultimodalModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# One-hot encode the entity names
encoder = OneHotEncoder()
entity_name_encoded = encoder.fit_transform(test_df['entity_name'].values.reshape(-1, 1)).toarray()

# Make predictions
predictions = []

for _, row in test_df.iterrows():
    # Extract image features
    image_features = extract_image_features(f"./test_images/{row['index']}.jpg")
    
    # Encode the entity name
    entity_name_features = encoder.transform([[row['entity_name']]]).toarray()
    
    # Convert to tensors
    entity_name_features = torch.tensor(entity_name_features, dtype=torch.float32)
    
    # Forward pass to get predictions
    value_pred, unit_pred = model(image_features, entity_name_features)
    
    # Format the prediction as "value unit"
    prediction = f"{value_pred.item():.2f} {list(allowed_units)[unit_pred.argmax()]}"
    predictions.append(prediction)

# Save the predictions to a CSV file
output_df = pd.DataFrame({'index': test_df['index'], 'prediction': predictions})
output_df.to_csv('output/test_out.csv', index=False)

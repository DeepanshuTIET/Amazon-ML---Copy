import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import parse_string
from constants import allowed_units
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import cv2
from sklearn.preprocessing import OneHotEncoder
from torchvision import models
import requests
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the image preprocessing function
def preprocess_image_from_url(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))  # Resize to 224x224
    img = torch.tensor(np.array(img) / 255.0, dtype=torch.float32).permute(2, 0, 1)  # Normalize and convert to tensor
    return img.unsqueeze(0)  # Add batch dimension

# Define the function to extract image features using ResNet
def extract_image_features(image_url):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Use 'weights' instead of 'pretrained'
    model.eval()  # Set the model to evaluation mode (no training)
    
    # Preprocess the image and extract features
    img_tensor = preprocess_image_from_url(image_url)
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze(0)  # Remove batch dimension after feature extraction

# Define the multimodal model that combines image features and entity name
class MultimodalModel(nn.Module):
    def __init__(self, entity_size, unit_size):
        super(MultimodalModel, self).__init__()
        self.fc_image = nn.Linear(1000, 512)  # Image features from ResNet
        self.fc_entity = nn.Linear(entity_size, 512)  # Encoded entity name
        self.fc_value = nn.Linear(1024, 1)  # Predict entity value (regression)
        self.fc_unit = nn.Linear(1024, unit_size)  # Predict unit (classification)

    def forward(self, image_features, entity_name_features):
        image_out = self.fc_image(image_features)
        entity_out = self.fc_entity(entity_name_features)
        combined = torch.cat((image_out, entity_out), dim=1)
        value_out = self.fc_value(combined)
        unit_out = self.fc_unit(combined)
        return value_out, unit_out

# Modified parse_string function to handle ranges
def parse_string(s):
    # Check if the string contains a range
    if " to " in s:
        parts = s.split(" to ")
        if len(parts) == 2:
            value_min = float(parts[0].split()[0])  # Extract minimum value
            unit = parts[0].split()[1]  # Assume the unit is the same for both values
            value_max = float(parts[1].split()[0])  # Extract maximum value
            return (value_min + value_max) / 2, unit  # Return the average value and unit
        else:
            raise ValueError(f"Invalid format in {s}")
    else:
        # Original handling of a single value and unit
        parts = s.split()
        if len(parts) == 2:
            value = float(parts[0])
            unit = parts[1]
            return value, unit
        else:
            raise ValueError(f"Invalid format in {s}")

# Dataset class for batch processing
class CustomDataset(Dataset):
    def __init__(self, dataframe, encoder, image_cache):
        self.dataframe = dataframe
        self.encoder = encoder
        self.image_cache = image_cache

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        entity_name = row['entity_name']
        entity_name_features = self.encoder.transform([[entity_name]]).toarray()
        image_features = self.image_cache[row['image_link']]

        value_target, unit_target = parse_string(row['entity_value'])
        return (torch.tensor(entity_name_features, dtype=torch.float32), 
                torch.tensor(image_features, dtype=torch.float32), 
                value_target, unit_target)

# Training loop
if __name__ == '__main__':
    # Load the dataset
    train_df = pd.read_csv('dataset/train.csv')

    # Optionally use a smaller dataset for testing
    # train_df = train_df.sample(frac=0.1)

    # One-hot encode the entity names
    encoder = OneHotEncoder()
    entity_name_encoded = encoder.fit_transform(train_df['entity_name'].values.reshape(-1, 1)).toarray()

    # Preprocess and cache image features
    image_features_cache = {}
    for i, row in train_df.iterrows():
        if row['image_link'] not in image_features_cache:
            image_features_cache[row['image_link']] = extract_image_features(row['image_link'])

    # Define the dataset and dataloader for batch processing
    dataset = CustomDataset(train_df, encoder, image_features_cache)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the model, loss functions, and optimizer
    model = MultimodalModel(entity_size=len(encoder.categories_[0]), unit_size=len(allowed_units))
    model = model.to(device)  # Move model to GPU if available

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_value = nn.MSELoss()  # For predicting the entity value
    criterion_unit = nn.CrossEntropyLoss()  # For predicting the unit

    # Training loop
    for epoch in range(5):  # Train for 5 epochs (adjust as needed)
        for entity_name_features, image_features, value_target, unit_target in dataloader:
            entity_name_features, image_features = entity_name_features.to(device), image_features.to(device)

            # Forward pass
            value_pred, unit_pred = model(image_features, entity_name_features)

            # Compute loss
            loss_value = criterion_value(value_pred, torch.tensor([[value_target]], dtype=torch.float32).to(device))
            loss_unit = criterion_unit(unit_pred, torch.tensor([list(allowed_units).index(unit_target)], dtype=torch.long).to(device))

            # Backward pass and optimization
            loss = loss_value + loss_unit
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')

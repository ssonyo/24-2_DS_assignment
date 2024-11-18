import torch
import clip
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, classification_report

from model import CustomCLIPClassifier
from utils import CustomDataset, compute_ece, plot_confidence_and_accuracy, visualize_embeddings_with_tsne

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the saved model
model_load_path = "/root/DS_assignment_raw/Representational-Learning/saved_model.pth"
classifier_model = CustomCLIPClassifier(model).to(device)
classifier_model.load_state_dict(torch.load(model_load_path))
classifier_model.eval()
print(f"Model loaded from {model_load_path}")

# Load and preprocess data
dataset = load_from_disk("/root/DS_assignment_raw/Representational-Learning/dataset/val")
custom_dataset = CustomDataset(dataset, preprocess)
dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=False)

# Evaluate model and measure metrics
all_probs = []
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = classifier_model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)  # Predicted labels
        
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)
print(all_labels.shape)
all_preds = np.concatenate(all_preds)
print(all_preds.shape)

# Compute Accuracy and F1 Score
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy:.4f}")

# Compute classification report (includes F1 Score, Precision, Recall)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

# Compute ECE
ece_score = compute_ece(all_probs, all_labels)
print(f"ECE Score: {ece_score:.4f}")

# Visualizations
visualize_embeddings_with_tsne(classifier_model, dataloader)
plot_confidence_and_accuracy(all_probs, all_labels)

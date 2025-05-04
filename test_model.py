import torch
import numpy as np
import joblib
from task_match_model import TaskMatchModel
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

class MatchDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading validation data including embeddings, structured features, and labels.
    """

    def __init__(self, task_embeds, team_embeds, struct_feats, labels):
        self.task = torch.tensor(task_embeds, dtype=torch.float32).unsqueeze(1)
        self.team = torch.tensor(team_embeds, dtype=torch.float32).unsqueeze(1)
        self.struct = torch.tensor(struct_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.task[idx], self.team[idx], self.struct[idx], self.labels[idx]

# Load validation data from files
val_task = np.load("val_task_embeddings.npy")
val_team = np.load("val_team_embeddings.npy")
val_struct = np.load("val_structured_features.npy")
val_labels = np.load("val_labels.npy")

dataset = MatchDataset(val_task, val_team, val_struct, val_labels)
loader = torch.utils.data.DataLoader(dataset, batch_size=64)

# Load the trained model
model = TaskMatchModel()
model.load_state_dict(joblib.load("task_matcher.pkl"))
model.eval()

# Run predictions on validation data
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for task, team, struct, label in loader:
        output = model(task, team, struct)
        probs = torch.sigmoid(output)
        preds = (probs >= 0.5).int()

        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        all_labels.extend(label.tolist())

# Print classification metrics
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# Save predictions to CSV
df = pd.DataFrame({
    "true_label": all_labels,
    "predicted": all_preds,
    "confidence": all_probs
})
df.to_csv("validation_predictions.csv", index=False)

print("Predictions saved to validation_predictions.csv.")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from task_match_model import TaskMatchModel
import numpy as np
import joblib

class MatchDataset(Dataset):
    """
    Custom dataset for loading task and team embeddings along with structured features and labels.
    """

    def __init__(self, task_embeds, team_embeds, struct_feats, labels):
        self.task = torch.tensor(task_embeds, dtype=torch.float32).unsqueeze(1)  # (N, 1, embed_dim)
        self.team = torch.tensor(team_embeds, dtype=torch.float32).unsqueeze(1)
        self.struct = torch.tensor(struct_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.task[idx], self.team[idx], self.struct[idx], self.labels[idx]

# Load training data
task_embeds = np.load("task_embeddings.npy")
team_embeds = np.load("team_embeddings.npy")
struct_feats = np.load("structured_features.npy")
labels = np.load("labels.npy")

dataset = MatchDataset(task_embeds, team_embeds, struct_feats, labels)

# Split dataset into training and validation sets
train_size = 30000
val_size = 10000
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# Save validation data for use in test script
val_task = torch.cat([val_ds[i][0] for i in range(val_size)]).numpy().reshape(val_size, -1)
val_team = torch.cat([val_ds[i][1] for i in range(val_size)]).numpy().reshape(val_size, -1)
val_struct = torch.stack([val_ds[i][2] for i in range(val_size)]).numpy()
val_labels = torch.tensor([val_ds[i][3].item() for i in range(val_size)]).numpy()

np.save("val_task_embeddings.npy", val_task)
np.save("val_team_embeddings.npy", val_team)
np.save("val_structured_features.npy", val_struct)
np.save("val_labels.npy", val_labels)

print("Validation dataset saved to disk (10,000 samples).")

# Initialize model, loss function, optimizer, and scheduler
model = TaskMatchModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5, verbose=True
)

# Training loop
best_val_loss = float('inf')
early_stop_counter = 0
EPOCHS = 10

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for task, team, struct, label in train_loader:
        optimizer.zero_grad()
        output = model(task, team, struct)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for task, team, struct, label in val_loader:
            output = model(task, team, struct)
            loss = criterion(output, label)
            val_loss += loss.item()

            pred = (torch.sigmoid(output) >= 0.5).int()
            correct += (pred == label.int()).sum().item()
            total += label.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total

    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

    scheduler.step(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= 4:
            print("Early stopping triggered.")
            break

# Save the trained model
joblib.dump(model.state_dict(), "task_matcher.pkl")
print("Model training complete. Model saved as task_matcher.pkl.")

import pandas as pd
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load base data (generated from generate_data.py)
df_tasks = pd.read_csv("task_list.csv")
df_team = pd.read_csv("team_profiles.csv")

# Initialize embedding function and ChromaDB client
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

# Access ChromaDB collections for tasks and team members
task_collection = client.get_collection(name="tasks", embedding_function=embedding_fn)
team_collection = client.get_collection(name="team", embedding_function=embedding_fn)

# Number of training samples to generate
NUM_SAMPLES = 40000

# Storage lists for features and labels
structured_features = []
task_embeddings = []
team_embeddings = []
labels = []

# Convert records to dictionaries for sampling
task_pool = df_tasks.to_dict("records")
team_pool = df_team.to_dict("records")

# Generate random task-member pairs and compute features
for _ in range(NUM_SAMPLES):
    task = random.choice(task_pool)
    person = random.choice(team_pool)

    # Structured feature extraction
    dept_match = int(task["department"] == person["department"])
    task_skills = set(task["required_skills"].split(";"))
    person_skills = set(person["skills"].split(";"))
    skill_overlap = len(task_skills & person_skills) / max(len(task_skills), 1)
    availability = person["availability_hrs"]
    success_rate = person["past_success_rate"]
    deadline = task["deadline_days"]
    complexity = task["complexity"]

    # Embedding similarity
    task_embed = embedding_fn(task["description"])[0]
    team_embed = embedding_fn(person["skills"])[0]
    similarity = cosine_similarity([task_embed], [team_embed])[0][0]

    # Label assignment (with noise injection)
    label = 1 if dept_match and skill_overlap >= 0.5 else 0
    if random.random() < 0.1:
        label = 1 - label  # Add label noise (10%)

    # Collect data for training
    structured_features.append([
        availability, success_rate, deadline, complexity, skill_overlap, similarity
    ])
    task_embeddings.append(task_embed)
    team_embeddings.append(team_embed)
    labels.append(label)

# Save processed arrays for model training
np.save("structured_features.npy", np.array(structured_features))
np.save("task_embeddings.npy", np.array(task_embeddings))
np.save("team_embeddings.npy", np.array(team_embeddings))
np.save("labels.npy", np.array(labels))

print(f"Generated and saved {NUM_SAMPLES} training samples.")

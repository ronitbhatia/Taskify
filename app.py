import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from task_match_model import TaskMatchModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load team profiles
df_team = pd.read_csv("team_profiles.csv")

# Load the trained model and embedding function
model = TaskMatchModel()
model.load_state_dict(joblib.load("task_matcher.pkl"))
model.eval()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit App Interface
st.title("Task-to-Team Member Match Predictor")

st.header("Enter Task Details")
task_description = st.text_area("Task Description", "Build a Flask API with Docker and deploy it to AWS")
task_skills = st.text_input("Required Skills (semicolon-separated)", "Flask;Docker;AWS")
deadline = st.slider("Deadline (days)", 1, 30, 7)
complexity = st.slider("Complexity (1–5)", 1, 5, 3)
department = st.selectbox("Department", ["Engineering", "Marketing", "Sales"])

# Prediction trigger
if st.button("Find Best Matches"):
    st.info("Processing predictions...")
    
    task_skill_set = set(task_skills.split(";"))
    task_embed = embedder.encode([task_description])

    results = []

    for _, person in df_team.iterrows():
        # Structured feature computation
        dept_match = int(department == person["department"])
        person_skills = set(person["skills"].split(";"))
        skill_overlap = len(task_skill_set & person_skills) / max(len(task_skill_set), 1)
        availability = person["availability_hrs"]
        success_rate = person["past_success_rate"]

        # Embedding similarity
        person_embed = embedder.encode([person["skills"]])
        similarity = cosine_similarity(task_embed, person_embed)[0][0]

        # Feature vector for model input
        structured = np.array([[availability, success_rate, deadline, complexity, skill_overlap, similarity]], dtype=np.float32)
        task_tensor = torch.tensor(task_embed, dtype=torch.float32).unsqueeze(1)
        team_tensor = torch.tensor(person_embed, dtype=torch.float32).unsqueeze(1)
        structured_tensor = torch.tensor(structured)

        # Model prediction
        with torch.no_grad():
            output = model(task_tensor, team_tensor, structured_tensor)
            prob = torch.sigmoid(output).item()

        # Reason generation for result explanation
        if skill_overlap >= 0.6 and availability >= 25:
            reason = "Strong skill match and availability"
        elif skill_overlap >= 0.5:
            reason = "Good skill match, but limited hours"
        elif similarity >= 0.6 and success_rate >= 0.9:
            reason = "High general match and proven success"
        else:
            reason = "Low skill overlap or availability"

        results.append({
            "Name": person["name"],
            "Department": person["department"],
            "Match Probability": round(prob, 4),
            "Availability": availability,
            "Success Rate": success_rate,
            "Reason": reason
        })

    # Display top 5 matches
    top_matches = sorted(results, key=lambda x: x["Match Probability"], reverse=True)[:5]
    st.subheader("Top Recommended Team Members")
    st.table(top_matches)

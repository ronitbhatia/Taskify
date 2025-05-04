import pandas as pd
import random
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Define skill pools for each department
engineering_skills = ["Python", "Flask", "SQL", "MongoDB", "Docker", "Kubernetes", "Git", "AWS", "React"]
marketing_skills = ["SEO", "Content Marketing", "Google Ads", "Figma", "Market Research"]
sales_skills = ["CRM", "Lead Generation", "Cold Calling", "Negotiation", "Salesforce"]

departments = [
    {"name": "Engineering", "skills": engineering_skills},
    {"name": "Marketing", "skills": marketing_skills},
    {"name": "Sales", "skills": sales_skills}
]

# Define task templates for each department
task_templates = {
    "Engineering": [
        "Build a {} microservice using {}",
        "Set up CI/CD with {}",
        "Write unit tests using {}"
    ],
    "Marketing": [
        "Launch a {} campaign with {}",
        "Optimize SEO using {}",
        "Design creatives with {}"
    ],
    "Sales": [
        "Generate leads using {}",
        "Create client pitch using {}",
        "Analyze pipeline via {}"
    ]
}

def generate_task(idx, dept):
    """Generates a synthetic task with randomized required skills and description."""
    max_skills = len(dept["skills"])
    num_skills = random.randint(2, min(4, max_skills))
    skills = random.sample(dept["skills"], num_skills)

    template = random.choice(task_templates[dept["name"]])
    num_placeholders = template.count("{}")
    techs = random.sample(dept["skills"], min(num_placeholders, len(dept["skills"])))
    description = template.format(*techs)

    return {
        "task_id": f"T{1000 + idx}",
        "description": description,
        "required_skills": ";".join(skills),
        "complexity": random.randint(1, 5),
        "deadline_days": random.randint(2, 14),
        "department": dept["name"]
    }

def generate_member(idx, dept):
    """Generates a synthetic team member profile with randomized skills and metrics."""
    max_skills = len(dept["skills"])
    num_skills = random.randint(2, min(6, max_skills))
    skills = random.sample(dept["skills"], num_skills)

    return {
        "name": f"{dept['name']}_M{idx}",
        "department": dept["name"],
        "skills": ";".join(skills),
        "availability_hrs": random.randint(10, 40),
        "past_success_rate": round(random.uniform(0.6, 0.98), 2)
    }

# Configuration for number of synthetic records
NUM_TASKS = 200
NUM_TEAM_MEMBERS = 100

# Generate datasets
tasks = []
team = []

task_count_per_dept = NUM_TASKS // len(departments)
team_count_per_dept = NUM_TEAM_MEMBERS // len(departments)

for dept in departments:
    tasks.extend([generate_task(i + len(tasks), dept) for i in range(task_count_per_dept)])
    team.extend([generate_member(i + len(team), dept) for i in range(team_count_per_dept)])

# Save datasets to CSV
df_tasks = pd.DataFrame(tasks)
df_team = pd.DataFrame(team)
df_tasks.to_csv("task_list.csv", index=False)
df_team.to_csv("team_profiles.csv", index=False)

# Initialize embedding function and ChromaDB client
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Add formatted semantic text for embedding
df_tasks["text"] = df_tasks.apply(
    lambda row: f"Task: {row['description']} | Skills: {row['required_skills']} | Dept: {row['department']}",
    axis=1
)
df_team["text"] = df_team.apply(
    lambda row: f"{row['name']} from {row['department']} knows {row['skills']} | Avail: {row['availability_hrs']} hrs",
    axis=1
)

# Create and populate ChromaDB collections
task_col = chroma_client.create_collection(name="tasks", embedding_function=embedding_fn)
task_col.add(
    documents=df_tasks["text"].tolist(),
    ids=df_tasks["task_id"].tolist(),
    metadatas=df_tasks.drop(columns="text").to_dict("records")
)

team_col = chroma_client.create_collection(name="team", embedding_function=embedding_fn)
team_col.add(
    documents=df_team["text"].tolist(),
    ids=df_team["name"].tolist(),
    metadatas=df_team.drop(columns="text").to_dict("records")
)

print("Base dataset and embeddings generated successfully.")

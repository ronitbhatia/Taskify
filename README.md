# Taskify- AI-Powered Task-to-Team Member Matching System

This project implements an end-to-end AI system that matches tasks to the most suitable team members using a Transformer-based deep learning model. It combines semantic embeddings, structured features, and cross-attention to predict task-member compatibility, and includes a local web interface for real-time recommendations.

The system processes task descriptions and team profiles, generates embeddings, builds a hybrid training dataset, and trains a custom neural network for binary classification. Users can interact with the model through a Streamlit UI to predict the best-fit team members for any task.

---

# Key Features

1. **Custom Transformer-Based Neural Network** for matching tasks to team members.
2. **Embedding + Structured Features Fusion** with cross-attention and multi-layer MLPs.
3. **Training on 40,000 Samples** using synthetic task/member data across departments.
4. **Validation and Evaluation Tools** with precision, recall, F1-score, and confusion matrix.
5. **Streamlit UI for Local Deployment** with real-time task input and top-N teammate recommendations.

---

# Dataset

The data is synthetically generated to simulate a realistic corporate environment with:
- 3 Departments: Engineering, Marketing, Sales
- 200 Tasks × 100 Team Members
- Each task has required skills, complexity, and deadlines
- Each team member has skills, availability, and historical success rates

All samples are embedded using `sentence-transformers` and labeled based on department and skill-match logic (with noise injected for realism).

---

# Project Structure

/Final
├── generate_data.py # Generates tasks, members, and stores embeddings in ChromaDB
├── build_training_set.py # Samples task-member pairs, computes features, generates labels
├── task_match_model.py # Custom Transformer model with cross-attention and MLP
├── train_model.py # Loads dataset, trains model, saves best checkpoint
├── test_model.py # Evaluates model on validation set (accuracy, F1, confusion matrix)
├── app.py # Streamlit interface for real-time team recommendation
├── team_profiles.csv # Synthetic team member profiles (skills, availability)
├── task_list.csv # Synthetic task descriptions with required skills and deadlines
├── task_matcher.pkl # Trained model weights
├── requirements.txt # Python dependencies for training and deployment

# File Descriptions

## 1. generate_data.py
- Creates 200 tasks and 100 team members across departments.
- Generates embeddings using ChromaDB with Sentence Transformers.
- Saves CSVs and stores data for downstream processing.

## 2. build_training_set.py
- Samples 40,000 task-member pairs.
- Computes structured features and semantic similarity.
- Adds label noise and saves NumPy arrays for training.

## 3. task_match_model.py
- Defines `TaskMatchModel`, a hybrid neural network using:
  - Transformer encoders
  - Multihead cross-attention
  - Deep MLP for structured data
  - Classifier head with dropout and layer normalization

## 4. train_model.py
- Trains the model with BCEWithLogitsLoss.
- Logs training and validation loss/accuracy.
- Uses early stopping and learning rate scheduler.
- Saves best model and validation set for testing.

## 5. test_model.py
- Loads validation data and trained model.
- Computes accuracy, precision, recall, F1, and confusion matrix.
- Exports predictions with confidence scores to CSV.

## 6. app.py
- Streamlit app where users can input a task.
- Automatically computes features and ranks team members.
- Shows top 5 team matches with explanations ("Why?").

---

# Example Use Case

A project manager wants to assign a task:  
> "Build a Flask API and deploy to AWS."  
Using the app, they input the task details, and the system recommends the top 5 engineers best suited for it, based on skill overlap, availability, past success rate, and embedding similarity.

# Model Performance (Validation Set)
Accuracy: 89.6%

Class 0 (Non-Match):

Precision: 89.7%, Recall: 96.0%, F1: 92.8%

Class 1 (Match):

Precision: 89.4%, Recall: 75.2%, F1: 81.7%

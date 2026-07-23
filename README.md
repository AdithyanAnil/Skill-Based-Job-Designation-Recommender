# Skill-Based Job Designation Recommender

A machine learning system that predicts the most suitable job designation for a candidate based on their listed skills. Given a free-text skill set (e.g. "Python, SQL, Machine Learning"), the model recommends the closest-matching role from a set of predefined designations.

## How it works
- **Input**: Raw skill text extracted from resumes/candidate profiles
- **Feature extraction**: TF-IDF vectorization converts skill text into numerical features
- **Model**: Random Forest Classifier trained to map skill patterns to job designations
- **Output**: Predicted job designation with associated confidence

## Dataset
Trained on a synthetic resume dataset covering 8 job designations (ML Engineer, Data Analyst, Backend Developer, Front End Developer, Cloud Engineer, Designer, HR Manager, Digital Marketing), with balanced class representation.

## Tech stack
- Python, pandas
- scikit-learn (TfidfVectorizer, RandomForestClassifier, train_test_split)
- Evaluation via accuracy, precision, recall, F1-score, and confusion matrix

## Usage
1. Load and clean the resume dataset
2. Vectorize the `Skills` column with TF-IDF
3. Train the Random Forest classifier on the vectorized features
4. Predict designation for new skill inputs

## Status
Currently a proof-of-concept on an 8-class dataset; being extended toward a larger 64-designation classification system.

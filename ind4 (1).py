import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import networkx as nx

# Define the Bayesian Network structure
model = BayesianNetwork([
    ('Hours_Studied', 'Exam_Score'),
    ('Attendance', 'Exam_Score'),
    ('Parental_Involvement', 'Exam_Score'),
    ('Access_to_Resources', 'Exam_Score'),
    ('Previous_Scores', 'Exam_Score'),
    ('Motivation_Level', 'Exam_Score'),
    ('Tutoring_Sessions', 'Exam_Score'),
    ('Family_Income', 'Access_to_Resources'),
    ('Family_Income', 'Parental_Involvement'),
    ('Teacher_Quality', 'Motivation_Level'),
    ('Teacher_Quality', 'Hours_Studied'),
    ('Teacher_Quality', 'Attendance'),
    ('School_Type', 'Teacher_Quality'),
    ('School_Type', 'Access_to_Resources'),
    ('School_Type', 'Peer_Influence'),
    ('Peer_Influence', 'Motivation_Level'),
    ('Peer_Influence', 'Parental_Involvement'),
    ('Physical_Activity', 'Sleep_Hours'),
    ('Physical_Activity', 'Motivation_Level'),
    ('Learning_Disabilities', 'Hours_Studied'),
    ('Learning_Disabilities', 'Attendance'),
    ('Learning_Disabilities', 'Previous_Scores'),
    ('Parental_Education_Level', 'Parental_Involvement'),
    ('Parental_Education_Level', 'Motivation_Level'),
    ('Distance_from_Home', 'Attendance'),
    ('Internet_Access', 'Access_to_Resources'),
    ('Internet_Access', 'Motivation_Level'),
    ('Extracurricular_Activities', 'Motivation_Level'),
    ('Hours_Studied', 'Sleep_Hours'),
    ('Extracurricular_Activities', 'Sleep_Hours'),
    ('Distance_from_Home', 'Sleep_Hours'),
    ('Motivation_Level', 'Sleep_Hours'),
    ('Learning_Disabilities', 'Sleep_Hours'),
    ('Gender', 'Parental_Involvement'),
    ('Parental_Education_Level', 'Access_to_Resources'),
    ('Parental_Education_Level', 'Motivation_Level'),
    ('School_Type', 'Parental_Involvement'),
    ('Internet_Access', 'Hours_Studied'),
    ('Learning_Disabilities', 'Motivation_Level')
])

# Load your dataset
data = pd.read_csv('cleaned_dataset.csv')


# Split the dataset into training (80%) and testing (20%)
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Fit data-driven CPTs for the remaining nodes using MLE
model.fit(train_data, estimator=MaximumLikelihoodEstimator)
# Add expert-defined CPDs to the model

# Check if the model is valid
if not model.check_model():
    raise ValueError("The Bayesian Network structure or CPTs are invalid!")

# Perform inference on the Bayesian Network
inference = VariableElimination(model)

# Evaluate the model on test data
predicted_scores = []
actual_scores = test_data['Exam_Score'].tolist()

for _, row in test_data.iterrows():
    evidence = row.drop('Exam_Score').to_dict()
    try:
        result = inference.map_query(variables=['Exam_Score'], evidence=evidence)
        predicted_scores.append(result['Exam_Score'])
    except Exception as e:
        predicted_scores.append(None)

# Calculate accuracy
valid_predictions = [(pred, actual) for pred, actual in zip(predicted_scores, actual_scores) if pred is not None]
if valid_predictions:
    predicted_scores, actual_scores = zip(*valid_predictions)
    accuracy = accuracy_score(actual_scores, predicted_scores)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
else:
    print("No valid predictions were made.")

# Function to save CPDs to a file
def save_cpds_to_file(model, filename):
    with open(filename, 'w') as file:
        for cpd in model.get_cpds():
            file.write(f"CPD for {cpd.variable}:\n")
            file.write(str(cpd))
            file.write("\n\n")  # Add spacing between CPDs
    print(f"CPDs saved to {filename}")

# Validation checks for the Bayesian Network
def validate_bayesian_network(model):
    """
    Perform a series of validation checks to ensure the Bayesian network is valid.
    """
    try:
        # Check if the model is a valid DAG
        if not model.check_model():
            raise ValueError("The model structure is not a valid DAG or some CPDs are missing.")
        print("Validation Passed: The model is a valid DAG.")

        # Check for missing CPDs
        nodes_without_cpds = [node for node in model.nodes() if not model.get_cpds(node)]
        if nodes_without_cpds:
            raise ValueError(f"Missing CPDs for the following nodes: {nodes_without_cpds}")
        print("Validation Passed: All nodes have CPDs.")

        # Check if all CPDs are consistent and sum to 1
        for cpd in model.get_cpds():
            if not np.allclose(np.sum(cpd.values, axis=0), 1):
                raise ValueError(f"CPD for {cpd.variable} does not sum to 1.")
        print("Validation Passed: All CPDs are consistent and valid.")

        # Run a sample inference sanity check
        inference = VariableElimination(model)
        test_query = inference.map_query(variables=['Exam_Score'], evidence={'Hours_Studied': 'High'})
        print("Sanity Check Passed: Inference query executed successfully.")
        print(f"Example inference result: {test_query}")

    except Exception as e:
        print(f"Validation Failed: {e}")

# Call validation function
validate_bayesian_network(model)

# Save CPDs to a file
save_cpds_to_file(model, "bayesian_network_cpds.txt")


def visualize_bayesian_network(model):
    """
    Visualizes the Bayesian Network structure.
    """
    plt.figure(figsize=(12, 8))
    nx_graph = nx.DiGraph(model.edges())
    pos = nx.circular_layout(nx_graph)  # Spring layout for better node positioning
    nx.draw(
        nx_graph,
        pos,
        with_labels=True,
        node_size=3000,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        edge_color="black",
        arrowsize=20,
    )
    plt.title("Bayesian Network Structure", fontsize=16)
    plt.show()

visualize_bayesian_network(model)
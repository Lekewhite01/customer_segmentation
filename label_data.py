import pickle as pkl
import pandas as pd
import numpy as np

data = pd.read_csv("./training_data/training_data.csv")

def load_model(filename):
    """
    Loads the machine learning model from a file using pickle.

    Parameters:
        filename (str): The filename from which to load the model.

    Returns:
        The loaded machine learning model.
    """
    # Open the file in binary read mode and load the model using pickle
    with open(filename, 'rb') as file:
        loaded_model = pkl.load(file)
    
    return loaded_model

def get_label(model):
    """
    Get the cluster labels assigned by a KMeans clustering model.

    Parameters:
        model (sklearn.pipeline.Pipeline): KMeans clustering model within a pipeline.

    Returns:
        numpy.ndarray: Array containing cluster labels assigned by the KMeans model.
    """
    # Extract cluster labels from the KMeans model within the pipeline
    labels = model.named_steps['kmeans'].labels_

    return labels

def main():
    """
    Main function to load a KMeans model, assign labels, and save the labeled data.

    Returns:
        None: The function saves the labeled data to a CSV file.
    """
    # Step 1: Load the KMeans model from a saved file
    model = load_model("./model/kmeans_model.pkl")

    # Step 2: Get cluster labels using the loaded model
    labels = get_label(model)

    # Step 3: Assign cluster labels to the input data
    data["clusters"] = labels

    # Step 4: Save the labeled data to a CSV file
    data.to_csv("./output/labelled_data.csv", index=False)
    
    print("Cluster Labels added to dataframe successfully!!!")
    
if __name__=="__main__":
    main()
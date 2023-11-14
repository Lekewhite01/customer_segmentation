import warnings
import pandas as pd
import numpy as np
import pickle as pkl
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from kneed import DataGenerator, KneeLocator
warnings.filterwarnings('ignore')

# Read training data into a pandas DataFrame
data = pd.read_csv("./training_data/training_data.csv")

def select_variance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select features with the highest trimmed variance from the input DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing numerical features.

    Returns:
        pd.DataFrame: Feature matrix containing the five columns with the highest trimmed variance.
    """
    # Calculate trimmed variance for numerical features in the DataFrame
    top_ten_trim_var = df.select_dtypes("number").apply(trimmed_var).sort_values()

    # Generate a list high_var_cols with the column names of the five features with the highest trimmed variance
    high_var_cols = top_ten_trim_var.tail(5).index.to_list()

    # Create the feature matrix X containing the five columns in high_var_cols
    X = df[high_var_cols]

    return X

def build_model(feature_matrix, num_clusters: int):
    """
    Build a KMeans clustering model using the input feature matrix.

    Parameters:
        feature_matrix (pd.DataFrame): Feature matrix for clustering.

    Returns:
        sklearn.pipeline.Pipeline: KMeans clustering model.
    """
    # Create a pipeline with StandardScaler for feature scaling and KMeans for clustering
    model = make_pipeline(StandardScaler(),
                          KMeans(n_clusters=num_clusters, random_state=42))

    # Fit the model to the input feature matrix
    model.fit(feature_matrix)
    
    return model

def save_model(model, filename):
    """
    Saves the machine learning model using pickle.

    Parameters:
        model: The machine learning model to be saved.
        filename (str): The filename to save the model.

    Returns:
        None
    """
    with open(filename, 'wb') as file:
        pkl.dump(model, file)

def main(k=5):
    """
    Main function to select features, build a KMeans model, and save the model.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing features for clustering.

    Returns:
        None: The function saves the KMeans model to a file.
    """
    # Step 1: Select features with the highest trimmed variance
    feature_matrix = select_variance(data)

    # Step 2: Build a KMeans clustering model using the selected features
    model = build_model(feature_matrix, num_clusters=k)

    # Step 3: Save the trained KMeans model to a file
    save_model(model, './model/kmeans_model.pkl')
    
    print("Model built and saved succesfully!!!")
    
if __name__=="__main__":
    main()


    
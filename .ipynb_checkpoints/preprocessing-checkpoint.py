import re
import glob
import pandas as pd
import numpy as np

def read_data(filepath: str) -> list:
    """
    Read multiple CSV files from a specified folder path.

    Parameters:
        filepath (str): The path to the folder containing CSV files.

    Returns:
        list: A list of DataFrames, each representing a CSV file.
    """
    # Specify the folder path and the regex pattern for file names
    folder_path = filepath
    file_pattern = '*.csv'  # Adjust the pattern as needed

    # Use glob to get a list of file paths that match the pattern
    file_paths = glob.glob(f'{folder_path}/{file_pattern}')

    # Create an empty list to store DataFrames
    dataframes = []

    # Loop through the file paths and read each CSV file into a DataFrame
    for file_path in file_paths:
        # Extract the file name using regex
        file_name = re.search(r'([^\/]+)\.csv', file_path).group(1)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Optionally, add a column with the file name to identify the source
        # df['source_file'] = file_name

        # Append the DataFrame to the list
        dataframes.append(df)

    return dataframes

def unpack_data_list(dataframes: list):
    """
    Unpack a list of DataFrames representing different tables in an e-commerce dataset.

    Parameters:
        dataframes (list): A list of DataFrames containing different tables.

    Returns:
        tuple: A tuple containing DataFrames for sellers, product categories, orders,
               order items, customers, geolocations, order payments, order reviews, and products.
    """
    # Unpack the list of DataFrames into individual variables
    sellers_df, product_category_df, orders_df,\
    order_items_df, customers_df, geolocation_df,\
    order_payments_df, order_reviews_df, products_df = dataframes

    return (
        sellers_df, product_category_df, orders_df,
        order_items_df, customers_df, geolocation_df,
        order_payments_df, order_reviews_df, products_df
    )

def merge_data(products_df: pd.DataFrame, 
               product_category_df: pd.DataFrame, 
               orders_df: pd.DataFrame,
               customers_df: pd.DataFrame, 
               order_items_df: pd.DataFrame, 
               order_payments_df: pd.DataFrame, 
               sellers_df: pd.DataFrame):
    """
    Merge multiple DataFrames to create a comprehensive dataset for analysis.

    Parameters:
        products_df (pd.DataFrame): DataFrame containing product information.
        product_category_df (pd.DataFrame): DataFrame containing product category information.
        orders_df (pd.DataFrame): DataFrame containing order information.
        customers_df (pd.DataFrame): DataFrame containing customer information.
        order_items_df (pd.DataFrame): DataFrame containing order items information.
        order_payments_df (pd.DataFrame): DataFrame containing order payments information.
        sellers_df (pd.DataFrame): DataFrame containing seller information.

    Returns:
        pd.DataFrame: A merged DataFrame containing information from all the input DataFrames.
    """
    # Merge products_df and product_category_df on 'product_category_name'
    products_merged = pd.merge(products_df, product_category_df, on='product_category_name', how='inner')

    # Drop unrequired columns from products_merged
    unrequired_columns = ["product_category_name"]
    products_merged.drop(unrequired_columns, axis=1, inplace=True)

    # Rename English product categories
    products_merged.rename({"product_category_name_english": "product_category"}, axis=1, inplace=True)

    # Merge all DataFrames to create a comprehensive dataset
    df = pd.merge(orders_df, customers_df, on='customer_id')
    df = df.merge(order_items_df, on='order_id')
    df = df.merge(order_payments_df, on='order_id')
    df = df.merge(products_merged, on='product_id')
    df = df.merge(sellers_df, on='seller_id')

    return df

def wrangle_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data wrangling on the input DataFrame.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame containing raw data.

    Returns:
        pd.DataFrame: Wrangled DataFrame with selected columns and corrected data types.
    """
    # Select required columns from the input DataFrame
    required_columns = ["customer_unique_id", "order_id", "order_status", "order_purchase_timestamp",
                        "customer_zip_code_prefix", "customer_city", "customer_state", "price", "freight_value",
                        "payment_sequential", "payment_type", "payment_installments", "payment_value", "product_category"]
    main_df = dataframe[required_columns]

    # Correcting the data types across the dataframe
    main_df["order_purchase_timestamp"] = pd.to_datetime(main_df["order_purchase_timestamp"])

    # Converting zipcode prefix from int to string since it won't be aggregated
    main_df["customer_zip_code_prefix"] = main_df["customer_zip_code_prefix"].astype(str)

    # Extracting year, month, and day from the order purchase timestamp
    main_df['order_purchase_year'] = main_df['order_purchase_timestamp'].dt.year.astype(str)
    main_df['order_purchase_month'] = main_df['order_purchase_timestamp'].dt.month.astype(str)
    main_df['order_purchase_day'] = main_df['order_purchase_timestamp'].dt.day.astype(str)

    return main_df

def final_wrangle(dataframe: pd.DataFrame):
    """
    Perform final data wrangling on the input DataFrame.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame containing raw data.

    Returns:
        None: The function modifies the input DataFrame in-place.
    """
    # Drop specified columns from the DataFrame
    training_df = dataframe.drop(["customer_unique_id", "order_id"], axis=1)
    return training_df

def main():
    """
    Main function to read, unpack, merge, wrangle, and save training data.

    Returns:
        None: The function saves the processed DataFrame as a CSV file.
    """
    # Step 1: Read data from the specified folder
    data_list = read_data("./input_data")

    # Step 2: Unpack the list of DataFrames
    sellers_df, product_category_df, orders_df,\
    order_items_df, customers_df, geolocation_df,\
    order_payments_df, order_reviews_df, products_df = unpack_data_list(data_list)

    # Step 3: Merge the unpacked DataFrames
    df = merge_data(products_df, 
               product_category_df, 
               orders_df,
               customers_df, 
               order_items_df, 
               order_payments_df, 
               sellers_df)

    # Step 4: Perform data wrangling on the merged DataFrame
    main_df = wrangle_data(df)
    
    # Step 5: Drop unrequired training fields on the wrangled DataFrame
    training_df = final_wrangle(main_df)
    
    # Step 6: Save the processed DataFrames as a CSV file
    main_df.to_csv("./training_data/EDA_data.csv", index=False) # This data is for EDA
    training_df.to_csv("./training_data/training_data.csv", index=False) # This data is for model building
    
    print("DataFrames have been saved to CSV successfully")



if __name__== "__main__":
    main()
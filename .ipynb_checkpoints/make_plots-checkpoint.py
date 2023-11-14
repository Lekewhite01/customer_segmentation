import pandas as pd
import plotly.express as px

def barplot(df: pd.DataFrame, x: str, y: str, title: str, xaxis: str, yaxis: str):
    """
    Create and display a bar plot using Plotly Express.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - x (str): The column in DataFrame to be used for the x-axis.
    - y (str): The column in DataFrame to be used for the y-axis.
    - title (str): The title of the plot.
    - xaxis (str): The label for the x-axis.
    - yaxis (str): The label for the y-axis.

    Returns:
    - None: The function displays the plot using Plotly Express.
    """

    # Create a bar plot using Plotly Express
    figure = px.bar(
        data_frame=df,
        x=x,
        y=y,
        title=title)

    # Customize the layout with axis titles and height
    figure.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)

    # Display the plot
    return figure.show()

def histogram(df: pd.DataFrame, x: str, title: str, xaxis: str):
    """
    Create and display a histogram using Plotly Express.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - x (str): The column in DataFrame to be used for the x-axis.
    - title (str): The title of the histogram.
    - xaxis (str): The label for the x-axis.

    Returns:
    - None: The function displays the histogram using Plotly Express.
    """

    # Create a histogram using Plotly Express
    figure = px.histogram(
        df,
        x=x,
        title=title)

    # Customize the layout with axis title and height
    figure.update_layout(xaxis_title=xaxis)

    # Display the histogram
    return figure.show()

def generate_analytics(df: pd.DataFrame, cluster_number: int):
    """
    Generate and display various analytics plots for a specific cluster in the given DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - cluster_number (int): The cluster number for which analytics should be generated.

    Returns:
    - None: The function generates and displays various analytics plots using Plotly Express.
    """

    # Extract data for the specified cluster
    cluster_df = df[df["clusters"]==cluster_number]
    
    # Compute number of customers in this cluster
    customers = cluster_df["customer_unique_id"].nunique()
    
    # Group by both 'order_year' and 'order_month' and count the number of orders in each month
    monthly_orders = cluster_df.groupby(['order_purchase_year', 'order_purchase_month']).size().reset_index(name='num_orders')

    # Group by 'order_year' and calculate the average number of orders per month to two decimal places
    average_monthly_orders = monthly_orders.groupby('order_purchase_year')['num_orders'].mean().reset_index(name='average_monthly_orders').round(2)

    # Compute customer distribution by state
    customers_per_state = cluster_df.groupby("customer_state").size().reset_index(name='num_customers').sort_values("num_customers", ascending=False)

    # Compute customer distribution by payment type
    customers_per_channel = cluster_df.groupby("payment_type").size().reset_index(name='num_customers').sort_values("num_customers", ascending=False)

    # Compute customer distribution by product category
    customers_per_product = cluster_df.groupby("product_category").size().reset_index(name='num_customers').sort_values("num_customers", ascending=False)
    
    print(f"Number of customers in the Cluster {cluster_number}:", customers)

    # Plot the customer distribution by product category
    barplot(df=customers_per_product[:10],
            x="product_category",
            y="num_customers",
            title="Customer distribution by Product Type",
            xaxis="Product Type", 
            yaxis="Number of Customers")

    # Plot the customer distribution by payment type
    barplot(df=customers_per_channel,
            x="payment_type",
            y="num_customers",
            title="Customer distribution by Payment Type",
            xaxis="Payment Type", 
            yaxis="Number of Customers")

    # Plot the customer distribution by state
    barplot(df=customers_per_state,
            x="customer_state",
            y="num_customers",
            title=f"Customer distribution by State for Cluster {cluster_number}",
            xaxis="Year", 
            yaxis="Number of Customers")

    # Plot Average Orders per month
    barplot(average_monthly_orders,
            x="order_purchase_year",
            y="average_monthly_orders",
            title=f"Average Orders per month for Cluster {cluster_number}",
            xaxis="Year", 
            yaxis="Average Monthly Orders")

    # Plot Price Distribution
    histogram(df=cluster_df,
              x="price", 
              title=f"Price Distribution for Cluster {cluster_number}",
              xaxis="Price")

    # Plot Payment Value Distribution
    histogram(df=cluster_df,
              x="payment_value", 
              title=f"Payment Value Distribution for Cluster {cluster_number}",
              xaxis="Payment Value")

    # Plot Payment Instalments Distribution
    histogram(df=cluster_df,
              x="payment_installments", 
              title=f"Payment Instalments Distribution for Cluster {cluster_number}",
              xaxis="Payment Instalments")
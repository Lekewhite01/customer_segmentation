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
    figure.update_layout(xaxis_title=xaxis, yaxis_title=yaxis, height=600)

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
    figure.update_layout(xaxis_title=xaxis, height=600)

    # Display the histogram
    return figure.show()

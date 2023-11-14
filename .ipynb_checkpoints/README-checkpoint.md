# Ecommerce Customer Segmentation
## About Dataset
This is a [Brazilian ecommerce public dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)of orders made at Olist Store. The dataset has information of 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allows viewing an order from multiple dimensions: from order status, price, payment and freight performance to customer location, product attributes and finally reviews written by customers. Also included, is a geolocation dataset that relates Brazilian zip codes to lat/lng coordinates.

This is real commercial data, it has been anonymised, and references to the companies and partners in the review text have been replaced with the names of Game of Thrones great houses.

The data used for this project is shared across nine separate CSV files containing `customer demographics`, `seller information`, `payment information` and `order information`.

## Approach
This project uses a kmeans algorithm to segment customers into a number of buckets as defined by the evaluation used. I started by importing all the required libraries and modules.
The following libraries were required for this project:
- re
- glob
- pandas
- numpy
- plotly express
- scipy
- scikit-learn
- kneed

The data was then read in from the destination path using regex (re) and glob. First the entire data is read into a list where each item is a pandas dataframe. Then tuple unpacking was used to assign variable names to each dataframe in the list for easier reference throughout the rest of the notebook.

All the dataframes were then merged using shared identifiers and some unrequired field(s) were dropped. Focus was then shifted to customer-centric, payment, and order information. This is because our focus is on customers to drive campaign efforts. 

The data is then explored for relationships between variables and wrangled to drop insignificant, high-cardinality features. Then the silhouette and elbow methods were used to determine an optimal number of clusters for model building. The clusters were expressed in terms of quantifiable (numeric) features to get a sense of their distinct attributes.

Lastly, to visualize the clusters, dimensionality reduction was applied to output the scatter plot in two dimensions.



**P.S. The clustering is based on high variance features for improved separation between clusters.**
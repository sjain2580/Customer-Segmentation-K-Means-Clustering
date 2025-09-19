# Customer Segmentation with K-Means Clustering

## Overview

This project demonstrates customer segmentation using the K-Means clustering algorithm. The goal is to group customers into distinct segments based on their demographic and spending habits, which can help businesses tailor marketing strategies and improve customer relationship management.

## Features

- Data Loading: Loads the customer data from a CSV file.

- Data Preprocessing: Handles data cleaning and scales numerical features.

- Elbow Method: Automatically determines the optimal number of clusters for the K-Means algorithm.Model Training: Trains a K-Means model on the prepared data.

- Clustering & Assignment: Assigns each customer to a specific cluster.

- Visualization: Generates an Elbow Plot and a scatter plot of the clustered data.

## Technologies Used

- Python: The core programming language for the project.

- Pandas: Used for data manipulation and analysis.

- Scikit-learn: A robust library for machine learning, used for K-Means clustering and data scaling.

- Matplotlib: A plotting library used for data visualization.

## Data Analysis & Processing

The project uses the Mall Customer Segmentation dataset. Key steps include:

- Loading the dataset into a Pandas DataFrame.

- Selecting features (Age, Annual Income (k$), and Spending Score (1-100)) for clustering.

- Scaling the numerical features using StandardScaler to ensure that no single feature dominates the clustering process.

## Model Used

The K-Means Clustering algorithm is used for unsupervised learning. It works by iteratively partitioning a dataset into a predefined number of clusters, aiming to minimize the sum of squared distances from each data point to its assigned cluster centroid.

## Model Training

The training process involves two key steps:

1. Finding Optimal Clusters: The Elbow Method is used to determine the best value for k (the number of clusters). This involves fitting the K-Means model for a range of k values and plotting the inertia (sum of squared distances). The point where the plot's curve "elbows" is considered the optimal number of clusters.

2. Training the Final Model: A KMeans model is then trained on the entire dataset using the optimal k value found in the previous step.

## How to Run the Project

1. Clone the repository:

```bash
git clone <https://github.com/sjain2580/Customer-Segmentation-K-Means-Clustering>
cd <repository_name>
```

2. Create and activate a virtual environment (optional but recommended):python -m venv venv

- On Windows:
  
```bash
.\venv\Scripts\activate
```

- On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Script:

```bash
python clustering.py
```

## Contributors

**<https://github.com/sjain2580>**
Feel free to fork this repository, submit issues, or pull requests to improve the project. Suggestions for model enhancement or additional visualizations are welcome!

## Connect with Me

Feel free to reach out if you have any questions or just want to connect!
**[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sjain04/)**
**[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/sjain2580)**
**[![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:sjain040395@gmail.com)**

---

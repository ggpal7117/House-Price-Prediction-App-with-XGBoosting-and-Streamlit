# House-Price-Prediction-App-with-XGBoosting-and-Streamlit
**App Link**: https://house-price-prediction-app-with-xgboosting-and-app-67y4ulhrykv.streamlit.app/

<img width="1420" height="620" alt="Image" src="https://github.com/user-attachments/assets/9e9c0b28-a18e-43ca-94eb-3d2138d90892" />

In this project, I took a sample housing dataset from Kaggle to predict house prices in cities in the United States. This data involved data preprocessing, splitting, training and testing, hyperparameter tuning, and the creation of the app using Streamlit. 

# Model Creation
The first part of the project was pre-processing the data for model training and testing. Features such as Zip Code, City, State, and County were in categorical form, so one-hot encoding was needed. The next step was reordering the dataset's features to build an inference pipeline that would accurately make predictions. After this, we could split our data to fit it to the XGBoost Model.


# Gradient Boosting
<img width="617" height="337" alt="Image" src="https://github.com/user-attachments/assets/e6d030ee-a784-4140-8d85-bd2908bf08f6" />

The XGBoost Model is built on top of the Gradient Boost Model algorithm, which uses a process called boosting. This means, results are aggregated throughout the algorithm's process instead of at the end. Each tree used in the algorithm is built on the previous tree's errors, and this gives it the power to learn complex patterns in the data.


# XGBoost Model
![Image](https://github.com/user-attachments/assets/0bec33a3-ca15-4432-b736-64311a4d7db1)

XGBoosting is an incredibly powerful machine learning algorithm built on gradient boosting, as stated earlier, with other optimization techniques. The XGBoost Model utilizes both L1 and L2 regularization to prevent overfitting. This helps with large datasets like the sample dataset with tens of thousands of rows. The algorithm also has built-in capabilities to impute and handle missing values aswell as built-in cross validation.


# Inference Pipeline
The .py file starts with many functions utilized for transforming inputs to make predictions. For example, getting the average population, income, and density of a certain area to fill in the input array. Also, creating functions to fill in the dummy variables(Zip Code, City, State, County) based on matches. 


# Streamlit App

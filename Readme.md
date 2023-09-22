# Telecommunications Customer Churn Prediction

## Overview

This project aims to predict customer churn in a telecommunications company using Machine Learning, specifically using a deep learning library: a TensorFlow Keras Artificial Neural Network. Customer churn, also known as customer attrition, refers to the loss of customers or subscribers.

## Dataset

The dataset used for this project is the ["Telco Customer Churn" dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) from Kaggle, which contains information about telecommunications customers. The dataset has been preprocessed to remove unnecessary columns and handle missing values.

## Important Variables

The following columns are considered important features for predicting customer churn:

- `Count`: Count of customer interactions.
- `Churn Score`: Churn score.
- `CLTV`: Customer Lifetime Value.
- `Phone Service`: Indicates if the customer has phone service (Yes/No).
- `Multiple Lines`: Indicates if the customer has multiple lines (Yes/No).
- `Gender`: Gender of the customer.
- `Senior Citizen`: Indicates if the customer is a senior citizen (Yes/No).
- `Tech Support`: Indicates if the customer has technical support (Yes/No).
- (Additional columns with customer information)

## Preprocessing

The preprocessing steps include:

1. Loading the dataset using Pandas.
2. Removing unnecessary columns.
3. Handling missing values and dropping rows with missing data.
4. Converting the "Total Charges" column to numeric.
5. Encoding categorical variables using LabelEncoder.
6. Replacing "No internet service" and "No phone service" with "No".

## Model Building

The machine learning model is built using TensorFlow and Keras. It's a binary classification model with the following architecture:

- Input layer with 26 neurons (features) and ReLU activation.
- Output layer with 1 neuron and sigmoid activation.

The model is compiled with the Adam optimizer and binary cross-entropy loss function. It's trained for 100 epochs.

## Model Evaluation

The model is evaluated on a validation dataset, and classification metrics such as accuracy, precision, recall, and F1-score are calculated. A confusion matrix is also generated to visualize the model's performance.

## Deployment

This project has been deployed on Render. You can access the deployed application at [https://customer-churn1.onrender.com/](https://customer-churn1.onrender.com/).

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required Python libraries using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook or Python script for data preprocessing and model training.
4. Deploy the trained model for predictions.

## Files and Structure

- `Telco_customer_churn.xlsx`: The dataset.
- `churn_model.h5`: The saved Keras model.
- `churn_model.pkl`: The pickled machine learning model.
- `new_scaler.pkl`: The pickled StandardScaler.

## Dependencies

- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Author

- Masixole Boya

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.



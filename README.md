# Airline Passenger Satisfaction Predictor

A machine learning web app that predicts airline passenger satisfaction based on flight details and service ratings.

**Live App Demo:** [**Click here to try the app!**](https://airline-satisfaction-predictor-8ygpsr2be2ear4jtuwxshu.streamlit.app*)

## Project Overview

This project aims to help airlines identify key factors that lead to passenger dissatisfaction. By predicting whether a passenger is likely to be dissatisfied, airlines can proactively address issues and improve service quality.

The model is a **Random Forest Classifier** trained on the "Airline Passenger Satisfaction" dataset from Kaggle. It achieves an **accuracy of ~96%** on the test set.

## Features
* **Interactive Web App:** Built with Streamlit to provide real-time predictions.
* **ML Pipeline:** Uses an `sklearn` Pipeline to handle data preprocessing (imputing missing values, scaling numeric features, and one-hot encoding categorical features).
* **Data Visualization:** The `model_training.ipynb` notebook includes exploratory data analysis (EDA) using Plotly to understand feature importance.

## Technologies Used
* **Python**
* **Pandas:** For data manipulation
* **Scikit-learn:** For building the ML pipeline and model
* **Joblib:** For saving and loading the model
* **Streamlit:** For building the interactive web app
* **Plotly:** For data visualization

## How to Run
1.  Clone the repository:
    `git clone https://github.com/your-username/airline-satisfaction-predictor.git`
2.  Install the required packages:
    `pip install -r requirements.txt`
3.  Run the Streamlit app:
    `streamlit run app.py`

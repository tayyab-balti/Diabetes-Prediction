# Diabetes Prediction

This repository contains an analysis and prediction model for the diabetes dataset. The project leverages popular data science libraries, including **pandas**, **numpy**, and **scikit-learn**. Additionally, a **Streamlit** web application has been developed to allow users to interact with the prediction model.

## Overview

The goal of this project is to predict whether a patient is diabetic based on diagnostic measurements from the [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set). The dataset contains various medical attributes such as glucose level, BMI, insulin levels, etc.

### Key Features:
- **Data Analysis**: In-depth exploration of the dataset, including descriptive statistics and data visualizations.
- **Modeling**: Machine learning models using **Random Forest**, **Logistic Regression**, and other classifiers.
- **Streamlit App**: A simple web app for real-time diabetes prediction based on user input.

---

## Project Structure

- **`app.py`**: Streamlit web application that enables user interaction and diabetes prediction.
- **`diabetes.csv`**: The dataset used for training and prediction.
- **`Diabeties_Prediction/`**: Jupyter notebooks containing data analysis and model building.
- **`README.md`**: This file, providing an overview of the project.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/tayyab-balti/Diabetes-Prediction.git
cd Diabetes-Prediction
```

### 2. Run the Streamlit App
```
streamlit run app.py
```

### 3. Data Analysis
Key findings from the dataset exploration:

- The dataset contains 768 instances and 9 attributes.
- Distribution of diabetic vs non-diabetic patients can be visualized in the bar chart below:

# Machine Learning Models
The following models have been trained and evaluated:

- Random Forest Classifier
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes

## Performance Metrics
The performance of each model was evaluated based on accuracy and confusion matrix.

# Streamlit Web Application
The Streamlit app allows users to input medical data manually via sliders, and it predicts the likelihood of diabetes. Key features of the app:

- **User-Friendly Interface**: Sliders for inputting features like glucose level, BMI, and age.
- **Real-time Prediction**: Instant prediction of diabetes based on user input using the pre-trained Random Forest model.

# Dataset
The dataset used in this project is sourced from Kaggle and can be found [here](https://www.kaggle.com/datasets/mathchi/diabetes-data-set). It contains the following features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (Target variable: 0 for non-diabetic, 1 for diabetic)

# Future Enhancements
Potential future improvements for this project include:

- **Model Tuning**: Further optimization of hyperparameters for better performance.
- **Additional Models**: Experimentation with other machine learning algorithms like SVM or DT.
- **Feature Engineering**: Testing additional feature selection and extraction techniques.

# Acknowledgements
- Kaggle for providing the Diabetes Dataset.
- Streamlit for making it easy to build interactive web apps for data science projects.

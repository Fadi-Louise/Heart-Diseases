Heart Disease Classification Project

Author: Fadi-Louise
Last Updated: 2025-03-31
Project Overview

This project implements a machine learning pipeline to predict heart disease based on patient medical data. The pipeline includes data cleaning, exploratory data analysis, feature selection, model training with regularization techniques, and model evaluation.
Table of Contents

    Introduction
    Dataset
    Implementation Details
    Installation
    Usage
    Results
    Future Work

Introduction

Heart disease is a leading cause of death globally. Early detection and prediction of heart disease can significantly improve patient outcomes. This project develops a classification model to predict the likelihood of heart disease based on patient attributes and medical measurements.

The project demonstrates several machine learning techniques including:

    Data cleaning and preprocessing
    Exploratory data analysis with visualizations
    Feature selection and engineering
    Implementation of multiple classification algorithms
    Regularization techniques (L1 and L2)
    Model evaluation and comparison

Dataset

The dataset contains various patient attributes and medical measurements, with a binary target variable indicating the presence of heart disease. Key features include:

    Demographic information (age, sex)
    Medical measurements (blood pressure, cholesterol levels)
    Test results (resting ECG, max heart rate)
    Symptomatic information (chest pain type, exercise angina)

Implementation Details

The project is implemented as a complete machine learning pipeline with the following components:
1. Data Preparation

    Handling missing values
    Identifying and removing duplicates
    Dealing with outliers
    Preparing data for analysis

2. Exploratory Data Analysis (EDA)

    Statistical summaries
    Distribution visualizations
    Correlation analysis
    Feature relationship exploration

3. Data Preprocessing

    Feature scaling using StandardScaler
    Categorical variable encoding
    Feature selection based on statistical tests

4. Model Training with Regularization

    Logistic Regression with L1 regularization (Lasso)
    Logistic Regression with L2 regularization (Ridge)
    Random Forest Classifier
    Support Vector Machine (where applicable)

5. Model Evaluation

    Accuracy, precision, recall, and F1-score metrics
    Confusion matrices
    ROC curves (for applicable models)
    Feature importance analysis

Installation

This project requires Python 3.5+ and several dependencies. To set up the environment:
bash

# Clone repository (if applicable)
git clone https://github.com/Fadi-Louise/heart-disease-classification.git
cd heart-disease-classification

# Install required packages
pip install -r requirements.txt

Dependencies

    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn

Usage

The project is implemented as a series of Jupyter notebooks that should be run in Google Colab:

    Upload the heart disease dataset to Google Colab
    Run the data preparation code to clean the dataset
    Execute the EDA section to understand feature distributions and relationships
    Proceed with model training and evaluation

Results

The models are evaluated based on multiple metrics including accuracy, precision, recall, and F1-score. Results typically show:

    Logistic Regression with L1/L2 regularization: Provides a good baseline with interpretable coefficients
    Random Forest: Often achieves higher accuracy by capturing non-linear relationships
    SVM: Can perform well on this dataset with proper parameter tuning

Feature importance analysis reveals the most predictive indicators of heart disease, which typically include:

    Exercise-induced angina
    Maximum heart rate
    Number of major vessels
    ST slope
    Chest pain type

Future Work

Potential enhancements for this project include:

    Implementing more advanced models like gradient boosting or neural networks
    Performing more extensive hyperparameter tuning
    Creating a simple web interface for model predictions
    Exploring additional feature engineering techniques
    Testing the model on external datasets for robustness

This project was created as part of a machine learning course assignment. The model is for educational purposes and should not be used for medical diagnosis.

# Heart Disease Classification Project

**Author**: Fadi-Louise  
**Last Updated**: 2025-03-31  

## Overview

This project implements a machine learning pipeline to predict heart disease based on medical data. The pipeline involves data cleaning, exploratory data analysis (EDA), feature selection, model training with regularization, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)

## Introduction

Heart disease is a major global health issue. Early prediction can improve outcomes significantly. This project develops a classification model that predicts the likelihood of heart disease based on various patient attributes and medical measurements, using several machine learning techniques.

Key features include:

- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Feature selection and engineering
- Model training using logistic regression, random forests, and SVM
- Regularization (L1/L2) techniques
- Model evaluation

## Dataset

The dataset includes medical and demographic attributes, with a target variable indicating the presence of heart disease. Notable features are:

- **Demographics**: age, sex
- **Medical Measurements**: blood pressure, cholesterol
- **Test Results**: ECG, max heart rate
- **Symptoms**: chest pain type, exercise-induced angina

## Implementation Details

The project follows a comprehensive pipeline:

### 1. Data Preparation
- Handle missing values
- Remove duplicates
- Address outliers
- Format data for analysis

### 2. Exploratory Data Analysis (EDA)
- Statistical summaries
- Visualizations of distributions and relationships
- Correlation analysis

### 3. Data Preprocessing
- Feature scaling using `StandardScaler`
- Encoding categorical variables
- Feature selection via statistical tests

### 4. Model Training
- Logistic Regression with L1 (Lasso) and L2 (Ridge) regularization
- Random Forest Classifier
- Support Vector Machine (SVM)

### 5. Model Evaluation
- Metrics: accuracy, precision, recall, F1-score
- Confusion matrix and ROC curves
- Feature importance analysis

## Installation

To set up the project environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/Fadi-Louise/heart-disease-classification.git
   cd heart-disease-classification
Install the required dependencies:

    pip install -r requirements.txt

Dependencies

    pandas

    numpy

    matplotlib

    seaborn

    scikit-learn

Usage

The project is organized as Jupyter notebooks and is intended to be run on Google Colab. The typical workflow is:

    Upload the heart disease dataset to Google Colab.

    Run the data cleaning and preprocessing code.

    Perform EDA to explore feature distributions and relationships.

    Train and evaluate the classification models.

Results

Models are evaluated using various metrics such as accuracy, precision, recall, and F1-score. Expected results:

    Logistic Regression: Provides a baseline model with interpretable coefficients.

    Random Forest: Captures non-linear relationships, typically achieving higher accuracy.

    SVM: Performs well when properly tuned.

Feature importance analysis typically highlights key indicators for heart disease prediction, including:

    Exercise-induced angina

    Maximum heart rate

    Number of major vessels

    ST slope

    Chest pain type

Future Work

Potential improvements include:

    Exploring advanced models like gradient boosting or neural networks

    Conducting more hyperparameter tuning

    Creating a web interface for easy model deployment

    Experimenting with additional feature engineering techniques

    Testing the model with external datasets to assess robustness

This project was created as part of a machine learning course assignment and is intended for educational purposes only. The model should not be used for medical diagnoses.


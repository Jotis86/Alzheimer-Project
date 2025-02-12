# 🧠 Alzheimer AI: Machine Learning Pipeline 🌟

This document provides a comprehensive overview of the machine learning pipeline used to predict Alzheimer's disease. The pipeline includes data preprocessing, feature selection, model training, hyperparameter tuning, and model evaluation. We have used various machine learning models and selected the best one based on performance metrics.

## Table of Contents 📑
1. Data Preprocessing 🧹
2. Feature Selection 🔍
3. Model Training and Hyperparameter Tuning 🤖
4. Model Evaluation 📊
5. Saving the Best Model 💾

## Data Preprocessing 🧹
Data preprocessing is a crucial step in the machine learning pipeline. It involves cleaning the data, handling missing values, and preparing the data for model training.

### Steps:
1. **Drop Unnecessary Columns**: We removed the `PatientID` and `DoctorInCharge` columns as they are not relevant for the prediction. This helps in reducing noise and focusing on the relevant features. 🗑️

2. **Split Data**: We split the data into features (`X`) and target variable (`y`). The features include various clinical and demographic information, while the target variable is the diagnosis of Alzheimer's disease. ✂️

3. **Train-Test Split**: We split the data into training and testing sets. The training set is used to train the models, while the testing set is used to evaluate their performance. We used an 80-20 split to ensure that the models have enough data to learn from while still having a separate set for evaluation. 📊

## Feature Selection 🔍
Feature selection helps in selecting the most relevant features for the model, reducing dimensionality, and improving model performance. By selecting the top features, we can improve the model's accuracy and reduce overfitting.

### Steps:
1. **SelectKBest**: We used `SelectKBest` with `f_classif` to select the top 10 features. This method selects features based on their statistical significance in predicting the target variable. 🔝

2. **Selected Features**: We printed the names of the selected features to understand which features are most important for the prediction. This helps in interpreting the model and understanding the underlying factors that contribute to Alzheimer's disease. 📝

## Model Training and Hyperparameter Tuning 🤖
We trained multiple machine learning models and performed hyperparameter tuning using `GridSearchCV` to find the best parameters for each model. Hyperparameter tuning is essential for optimizing the model's performance and ensuring that it generalizes well to new data.

### Models and Hyperparameters:
1. **Logistic Regression**:
   - `C`: Regularization strength. We tested values [0.1, 1, 10]. 🔧
   - `solver`: Algorithm to use in the optimization problem. We tested ['liblinear']. 🧩

2. **Support Vector Machine (SVM)**:
   - `C`: Regularization parameter. We tested values [0.1, 1]. 🔧
   - `gamma`: Kernel coefficient. We tested values [0.1, 0.01]. 🧩
   - `kernel`: Specifies the kernel type to be used in the algorithm. We tested ['rbf']. ⚙️

3. **K-Nearest Neighbors (KNN)**:
   - `n_neighbors`: Number of neighbors to use. We tested values [3, 5]. 🔧
   - `weights`: Weight function used in prediction. We tested ['uniform', 'distance']. ⚖️

4. **Gradient Boosting**:
   - `n_estimators`: Number of boosting stages to be run. We tested values [100, 200]. 🌲
   - `learning_rate`: Shrinks the contribution of each tree. We tested values [0.1, 0.2]. 📉
   - `max_depth`: Maximum depth of the individual regression estimators. We tested values [3, 4]. 🌳

5. **Random Forest**:
   - `n_estimators`: Number of trees in the forest. We tested values [100, 200]. 🌲
   - `max_depth`: Maximum depth of the tree. We tested values [10, 20]. 🌳
   - `min_samples_split`: Minimum number of samples required to split an internal node. We tested values [2, 5]. ✂️

### Training and Tuning:
We used `GridSearchCV` to perform hyperparameter tuning and selected the best model based on accuracy. `GridSearchCV` performs an exhaustive search over the specified parameter grid for each model, using cross-validation to evaluate the performance of each combination of parameters. 🔍

## Model Evaluation 📊
We evaluated each model using various metrics such as accuracy, classification report, confusion matrix, and ROC curve. These metrics provide a comprehensive view of the model's performance and help in selecting the best model.

### Evaluation Metrics:
1. **Accuracy**: Measures the proportion of correctly classified instances. It is a simple and intuitive metric but can be misleading if the classes are imbalanced. 📈

2. **Classification Report**: Provides precision, recall, and F1-score for each class. Precision measures the proportion of true positives among the predicted positives, recall measures the proportion of true positives among the actual positives, and F1-score is the harmonic mean of precision and recall. 📋

3. **Confusion Matrix**: Shows the number of true positives, true negatives, false positives, and false negatives. It provides a detailed breakdown of the model's performance and helps in identifying specific areas where the model may be making errors. 🔄

4. **ROC Curve and AUC**: Plots the true positive rate against the false positive rate and calculates the area under the curve. The ROC curve provides a visual representation of the model's performance across different thresholds, and the AUC summarizes the overall performance. 📉

## Best Model 🏆
After evaluating all the models, we selected the best one based on accuracy and other performance metrics. The best model is crucial for making accurate predictions and providing reliable results.

### Best Model Details:
1. **Model Name**: Gradient Boosting Classifier
2. **Hyperparameters**: 
   - `learning_rate`: 0.1
   - `max_depth`: 4
   - `n_estimators`: 200
3. **Performance Metrics**:
   - **Accuracy**: 0.944
   - **Precision**: 
     - Class 0: 0.95
     - Class 1: 0.93
   - **Recall**: 
     - Class 0: 0.96
     - Class 1: 0.92
   - **F1-Score**: 
     - Class 0: 0.96
     - Class 1: 0.92
   - **AUC**: 0.94
4. **Feature Importance**: The importance of each selected feature in the best model.

### Why This Model?
The Gradient Boosting Classifier was selected because it provided the highest accuracy and balanced performance across all evaluation metrics. It effectively handles the complexity of the data and generalizes well to new, unseen data. This model will be used for making predictions on new patient data to assist in the early detection of Alzheimer's disease.

## Saving the Best Model 💾
After evaluating all models, we selected the best one based on accuracy and saved it along with the feature selector using `joblib`. Saving the model and the feature selector allows us to easily load and use them for future predictions without retraining.

### Steps:
1. **Select Best Model**: We selected the model with the highest accuracy. This model is considered the best because it has the highest proportion of correctly classified instances on the test set. 🏆

2. **Save Model and Feature Selector**: We saved the best model and the feature selector as `.pkl` files for future use. This ensures that we can reproduce the same results and use the model for making predictions on new data. 💾

---

This README provides a detailed overview of the machine learning pipeline used in the Alzheimer's disease prediction project. Each step is crucial for building an accurate and reliable model. By following this pipeline, we ensure that the data is properly preprocessed, the most relevant features are selected, the models are optimized, and the best model is saved for future use. Feel free to reach out if you have any questions or need further assistance. 📞
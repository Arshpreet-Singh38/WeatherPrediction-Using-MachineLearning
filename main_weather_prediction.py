# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:58:13 2020

@author: singh
"""
# @file main_weather_prediction: main file from where all other functionalities will be called


#Importing classes from different files
from data_visualization import variables_trend_visualization
from data_visualization import variables_relationship_visualization
from machine_learning_prediction_models import machine_learning_models
from analyzing_feature_importance import feature_importance
from analyzing_feature_importance import visualizing_feature_importance
from feature_reduction_prediction import feature_reduction
from time_analysis import time_computation


# Importing pandas for data manipulation
import pandas as pd
# Importing numpy for data computing
import numpy as np
# Importing matplotlib for plotting
import matplotlib.pyplot as plt
# Chosing the style for matplotlib
plt.style.use('fivethirtyeight')
# Importing simple filter from warnings to suppress future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
# Importing training and testing model
from sklearn.model_selection import train_test_split

# read data from the .csv file as pandas dataframe
features=pd.read_csv('weather_record.csv')

# Retrieving years,months and days from the dataset
years=features['year']
months=features['month']
days=features['day']

# Creating an object for analyzing the variables in the dataset
# Creating an object for visualizing the trends of features
trend_obj=variables_trend_visualization(features)
trend_obj.variables_trend(years,months,days)
# Creating an object for visualizing the relationship among features and with the target value
relationship_obj=variables_relationship_visualization(features)
relationship_obj.variables_relationship()


# Data Preparations
# One-Hot encoding-to create binary column for all features
features=pd.get_dummies(features)
# Extracting the Target Values
labels=features['actual']
# Extracting the features after removing the Target Values 
features=features.drop('actual',axis=1)
# Listing the feature columns
feature_list_column_names=list(features.columns)
# Converting to numpy arrays
features=np.array(features)
labels=np.array(labels)


# Creating Training and Testing Sets-Training (70%),Testing(30%)
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.3,random_state=42)


# The baseline predictions are when model always predicts the 
# actual max temperature to be the historical average max temp of the day
baseline_predictions=x_test[:,feature_list_column_names.index('average')]
# Baseline error is baseline predictions with respect to the actual maximun temp value of the day 
baseline_errors=abs(baseline_predictions-y_test)
Average_baseline_error=round(np.mean(baseline_errors), 2)
# Printing the average baseline error,this error range is the reference for the model
# the predictions made by the model must have lesser error than baseline error
print("\n")
print('Average Baseline Error: ',Average_baseline_error)
print('\n')


# Creating object to formulate machine learning models
ml_obj=machine_learning_models(x_train,x_test,y_train,y_test)
# For RandomForest Regression
model_randomforest=ml_obj.random_forest(500,0)
# For Support Vector Regression
ml_obj.support_vector("rbf",1e3,0.1)
# For Gradient Boosting Regression
ml_obj.gradient_boosting(500)
# For Logistic Regression
ml_obj.logistic_regression()


# Creating object for analyzing features importance
feature_imp_obj=feature_importance(model_randomforest,feature_list_column_names)
# For computing features importance analytically 
importance_f=feature_imp_obj.analysis_feature_imp()
# Creating object for visualizing the features importance
visual_imp_obj=visualizing_feature_importance(model_randomforest,feature_list_column_names,importance_f)
# Plotting the cummulative graph of feature importance
features_imp=visual_imp_obj.cummulative_feature_importance()


# Creating the object to reduce the features as calculated from features importance
feature_red_obj=feature_reduction(features_imp,feature_list_column_names,x_train,x_test,y_train,y_test,model_randomforest)
accuracy_r=feature_red_obj.feature_reduction_model()
# Creating the object for computing and visualizing the time efficiency attained with feature reduction
time_comp_obj=time_computation(features_imp,feature_list_column_names,model_randomforest,x_train,x_test,y_train,y_test,accuracy_r)
time_comp_obj.computing_time()
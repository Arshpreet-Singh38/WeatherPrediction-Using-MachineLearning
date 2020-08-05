# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:39:42 2020

@author: singh
"""
# @file time_analysis- computing the run time between training and testing
# phase of machine learning models when fed with total features and reduced number of features

# Importing the class machine_learning_models from machine_learning_prediction_models file
from machine_learning_prediction_models import machine_learning_models

# Importing the time library for run-time evaluations
import time
# Importing nunpy for data computing
import numpy as np
# Importing pandas for data manipulation
import pandas as pd
# Importing matplotlib for plotting
import matplotlib.pyplot as plt

# Defining the class for computing the run-time
class time_computation:
    # The class constructor
    def __init__(self,imp_features,feature_names,rf_model,x_train,x_test,y_train,y_test,accuracy_red):
        self.features=imp_features
        self.feature_names=feature_names
        self.rf_model=rf_model
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test
        self.accuracy_red=accuracy_red
    
    # Creating the method for computing run time between training and testing
    # phase of machine learning model when fed with total features and reduced number of features
    # and plotting the trade off between accuracy and run-time
    def computing_time(self):
        
        # All features training and testing run time
        all_features_time = []
        # Iterating 5 times and taking average for the model run time with total features
        for i in range(5):
            start_time=time.time()
            self.rf_model.fit(self.x_train,self.y_train)
            all_features_predictions=self.rf_model.predict(self.x_test)
            end_time=time.time()
            all_features_time.append(end_time-start_time)

        all_features_time=np.mean(all_features_time)
        print("\n")
        print('All features total training and testing time:',round(all_features_time, 2),'seconds.')

        # Run Time training and testing for reduced number of features
        reduced_features_time=[]

        # Extract the names of the most important features
        important_feature_names=[feature[0] for feature in self.features[0:5]]
        # Find the columns of the most important features
        important_indices=[self.feature_names.index(feature) for feature in important_feature_names]
                
        # Create training and testing sets with only the important features
        important_train_features=self.x_train[:,important_indices]
        important_test_features=self.x_test[:,important_indices]
        
        # Iterating 5 times and taking average of the run time for reduced number of features
        for i in range(5):
            start_time=time.time()
            self.rf_model.fit(important_train_features,self.y_train)
            reduced_features_predictions=self.rf_model.predict(important_test_features)
            end_time=time.time()
            reduced_features_time.append(end_time-start_time)

        reduced_features_time = np.mean(reduced_features_time)
        print("\n")
        print('Reduced features total training and testing time:', round(reduced_features_time, 2), 'seconds.')
        print("\n")
        
        # Calling for model accuracy with total features
        obj=machine_learning_models(self.x_train,self.x_test,self.y_train,self.y_test)
        accuracy_tot=obj.random_forest(500,1)
        
        model_comparison = pd.DataFrame({'model': ['Total_Features', 'Reduced_Features'], 
                                 'accuracy': [round(accuracy_tot,2),round(self.accuracy_red,2)],
                                 'run_time (s)': [round(all_features_time,2),round(reduced_features_time,2)]})
        
        # Order the dataframe
        print(model_comparison[['model','accuracy', 'run_time (s)']])
        
        # Making the  plots
        
        # Setting up the plotting layout
        figure,(x1,x2)=plt.subplots(nrows=2,ncols=1,figsize=(4,8),sharex=True)

        # Seting up the x-axis
        x_values=[0,1]
        labels=list(model_comparison['model'])
        plt.xticks(x_values,labels)
        
        # Seting up the fonts
        fontdict = {'fontsize': 10}
        fontdict_yaxis = {'fontsize': 14}

        # Comparison for Accuracy in Total Features and Reduced Features
        x1.bar(x_values, model_comparison['accuracy'], color = ['r', 'g'], edgecolor = 'k', linewidth = 1.5)
        x1.set_ylim(bottom = 92, top = 94)
        x1.set_ylabel('Accuracy (%)', fontdict = fontdict_yaxis); 
        x1.set_title('Model Accuracy Comparison', fontdict= fontdict)

        # Comparison for Run-Time in Total Features and Reduced Features
        x2.bar(x_values, model_comparison['run_time (s)'], color = ['r', 'g'], edgecolor = 'k', linewidth = 1.5)
        x2.set_ylim(bottom = 0, top = 6)
        x2.set_ylabel('Run Time (sec)', fontdict = fontdict_yaxis)
        x2.set_title('Model Run-Time Comparison', fontdict= fontdict)
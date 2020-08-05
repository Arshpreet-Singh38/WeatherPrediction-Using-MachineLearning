# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:17:21 2020

@author: singh
"""
# @file feature_reduction_prediction- The file for reducing the total number of features
# to only the important features as computed earlier.

# Importing numpy for data computing
import numpy as np

# Creating the class for feature reduction
class feature_reduction:
    # The class constructor
    def __init__(self,imp_features,feature_names,x_train,x_test,y_train,y_test,rf_model):
        self.features=imp_features
        self.feature_names=feature_names
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test
        self.rf_model=rf_model
    
    # Defining the method for feature reduction and producing
    # prediction results using the reduced features
    def feature_reduction_model(self):
        # Extracting the names of the most important features
        important_feature_names=[feature[0] for feature in self.features[0:5]]
        # Finding the columns of the most important features
        important_indices=[self.feature_names.index(feature) for feature in important_feature_names]
                
        # Create training and testing sets with only the important features
        important_train_features=self.x_train[:,important_indices]
        important_test_features=self.x_test[:,important_indices]
        
        # Training the model on reduced features
        self.rf_model.fit(important_train_features,self.y_train)
        # Making predictions on the test data
        predictions=self.rf_model.predict(important_test_features)
        
        # Performance Metrics
        errors=abs(predictions-self.y_test)
        print("\n")
        print('Average error for Random Forest after feature Reduction:',round(np.mean(errors),4),'degrees.')
        
        # Calculating Mean Absolute Percentage Error (MAPE)
        mape=100*(errors /self.y_test)
        
        # Calculating and displaying accuracy for the reduced featues
        accuracy_reduced_features=100-np.mean(mape)
        print('Accuracy for Random Forest after feature reduction:', round(accuracy_reduced_features, 2), '%.')
        return accuracy_reduced_features
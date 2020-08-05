# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:49:34 2020

@author: singh
"""
#@file machine_learning_prediction_models- the file consists of the machine learning models for prediction


# Importing simple filter from warnings to suppress future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
# Importing numpy for data computing
import numpy as np

# Importing random forest for regression
from sklearn.ensemble import RandomForestRegressor
# Importing Support Vector for regression
from sklearn.svm import SVR
# Importing Gradient Boosting for Regression
from sklearn.ensemble import GradientBoostingRegressor
# Importing for Logistic Regression
from sklearn.linear_model import LogisticRegression

# Creating the class for training and testing various machine learning models
class machine_learning_models:
    # The class constructor
    def __init__(self,x_train,x_test,y_train,y_test):
        # The instance variables
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test
    
    # Method for the Random Forest Regression  
    def random_forest(self,estimators,check):
        # Defining the model for Random Forest Regression
        rf= RandomForestRegressor(n_estimators=estimators,random_state=42)

        #Training the Random Forest Model
        rf.fit(self.x_train,self.y_train);
        
        # Making predictions on the test data
        predictions=rf.predict(self.x_test)
        
        # Performance metrics
        errors=abs(predictions-self.y_test)
        
        # Mean Absolute Percentage Error(MAPE)
        mape=np.mean(100*(errors/self.y_test))
        
        # Displaying Accuracy results
        accuracy=100-mape
        
        # Check is used to distimguish when it is called from main file(check=0)
        # and when from some other class in any file except main (check=1)
        if(check==1):
            return accuracy
        else:
            print('Average Error for Random Forest Regression:',round(np.mean(errors),4),'degrees.')
            print('Accuracy for Random Forest Regression:', round(accuracy, 2), '%.')
            return rf
    
    # Method for the Support Vector Regression
    def support_vector(self,kernel_value,cost,gamma_value):
        # Defining the model for Support Vector Regression
        svr_model=SVR(kernel=kernel_value,C=cost,gamma=gamma_value)
        
        #Training the SVR Model
        svr_model.fit(self.x_train,self.y_train)
        
        # Making predictions on the test data
        svr_predictions=svr_model.predict(self.x_test)

        # Performance metrics
        errors=abs(svr_predictions-self.y_test)
        print("\n")
        print('Average Error for  Support Vector Regression:',round(np.mean(errors),4),'degrees.')

        # Mean Absolute Percentage Error(MAPE)
        mape=np.mean(100*(errors/self.y_test))
        
        #Displaying the acuracy results
        accuracy=100-mape
        print('Accuracy for Support Vector Regression:',round(accuracy, 2), '%.')
        
    # Method for the Gradient Boosting Regression
    def gradient_boosting(self,estimators):
        # Defining the model for gradient boosting regression
        gbr_model = GradientBoostingRegressor(n_estimators=estimators)
        
        # Training the GBR Model
        gbr_model.fit(self.x_train,self.y_train)
        
        # Making predictions on test data
        gbr_predictions=gbr_model.predict(self.x_test)

        # Performance Metrics
        errors=abs(gbr_predictions-self.y_test)
        print("\n")
        print('Average error for Gradient Boosting Regression:',round(np.mean(errors), 4),'degrees')

        # Mean Absolute Percentage Error(MAPE)
        mape=np.mean(100*(errors/self.y_test))
        
        # Displaying the accuracy results
        accuracy=100-mape
        print('Accuracy for Gradient Boosting Regression:',round(accuracy, 2),'%.')
    
    # Method for Logistic Regression
    def logistic_regression(self):
        # Defining the model for Logistic Regression
        logistic_rmodel=LogisticRegression()
        
        # Training the Logistic Regression Model
        logistic_rmodel.fit(self.x_train,self.y_train)
        
        # Making predictions on test data
        logistic_predictions=logistic_rmodel.predict(self.x_test)
        
        #Performance Metrics
        errors=abs(logistic_predictions-self.y_test)
        print("\n")
        print('Average error for Logistic Regression:',round(np.mean(errors), 4),'degrees.')

        # Mean Absolute Percentage Error(MAPE)
        mape=np.mean(100*(errors/self.y_test))
        
        # Displaying the accuracy results
        accuracy=100-mape
        print('Accuracy for Logistic Regression:',round(accuracy,2),'%.')
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:49:31 2020

@author: singh
"""
# @file analyzing_feature_importance: analyzing the important features that were considered by
# the random forest regression model to produce comparitively better results than other ML Models.

# Importing matplotlib for plotting
import matplotlib.pyplot as plt
# Importing numpy for data computing
import numpy as np

# Creating the class to compute features importance analytically
class feature_importance:
    # Defining the class variable
    features=()
    # The class constructor
    def __init__(self,model_rf,feature_names):
        # The instance variables
        self.model_rf=model_rf
        self.feature_names=feature_names
    
    # Method to compute the tuple with feature name and its corresponding importance value 
    def analysis_feature_imp(self):
        # Analyzing the importance of the features
        imp=list(self.model_rf.feature_importances_)
        
        # Tuples with feature and its corresponding importance
        feature_importance.features=[(feature,round(importance,2)) for feature,importance in zip(self.feature_names,imp)]

        # Sorting the feature importances by most important first
        feature_importance.features=sorted(feature_importance.features,key=lambda x:x[1],reverse=True)
        return imp

# Creating the class to visualize features importance
class visualizing_feature_importance(feature_importance):
    # The constructor
    def __init__(self,model_rf,feature_names,importance):
        # Inheriting the feature_importance class
        super().__init__(model_rf,feature_names)
        self.importance=importance
    
    # Plotting the bar graph to show percentage of the feature importance in making predictions by the model 
    def visual_feature_imp(self):
        # Defining the layout for the bar graph
        fig,(x1)=plt.subplots(nrows=1,ncols=1,figsize=(5,5))
        fig.autofmt_xdate(rotation = 45)
        
        # Total no of importances (features) to be plotted 
        no_x_values=list(range(len(self.importance)))

        # Creating a Bar Graph
        plt.bar(no_x_values,self.importance,orientation='vertical',color = 'b',edgecolor='k',linewidth=1.0)
        
        # Labeling ticks for the x-axis
        plt.xticks(no_x_values,self.feature_names,rotation='vertical')
        
        # Defining the Axes labels and the title
        x1.set_xlabel('VARIABLE');x1.set_ylabel('Importance');x1.set_title('Feature Importances')
    
    # Plotting a cummulative graph to visualize the total markup importance of the features (adding to each other)
    def cummulative_feature_importance(self):
        # Defining the layout for the cummulative graph
        fig,(x1)=plt.subplots(nrows=1,ncols=1,figsize=(5,5))
        fig.autofmt_xdate(rotation = 45)
        
        # Total no of features to be plotted
        no_x_values=list(range(len(self.importance)))
        
        # Sorting importances and features independently
        sorted_importances=[importance[1] for importance in feature_importance.features]
        sorted_features=[importance[0] for importance in feature_importance.features]
        
        # Cumulative importances
        cumulative_importances=np.cumsum(sorted_importances)
        
        # Making a line graph to show cummulative importances
        plt.plot(no_x_values,cumulative_importances,'b')
        
        # Drawing a line at 95% of importance to filter
        # the featues that together have 95% of importance in making predictions
        plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
        
        # Formating the x ticks and labeling the axes and title
        plt.xticks(no_x_values,sorted_features,rotation = 'vertical')
        x1.set_xlabel('VARIABLES');x1.set_ylabel('Cummulative Importance');x1.set_title('Cummulative Importances')
        
        # Calling the function to plot bar graph after the cummulative graph has been plotted
        visualizing_feature_importance.visual_feature_imp(self)
        return feature_importance.features
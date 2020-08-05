# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:58:49 2020

@author: singh
"""
#@file data_visualization- this file contains functionalities for visualizing the dataset variables


# Importing datetime to deal with the dates
import datetime
# Importing matplotlib for plotting
import matplotlib.pyplot as plt
# Importing seaborn for pair plots 
import seaborn as sns


# Class for visually analysing the trends of the features and target value with time
class variables_trend_visualization:
    # The class constructor
    def __init__(self,features):
        # Instance Variable
        self.features=features
    
    # Method for visualizing the variables trend with time through multiple plots
    def variables_trend(self,years,months,days):
        # Listing the date and then converting to datetime object
        dates=[str(int(year))+'-'+str(int(month))+'-'+str(int(day))for year, month, day in zip(years, months, days)]
        dates=[datetime.datetime.strptime(date,'%Y-%m-%d') for date in dates]        
        
        # Plotting the trends of the variables with time
        
        # Defining the plotting layout
        fig,((x1,x2),(x3,x4))=plt.subplots(nrows=2,ncols=2,figsize = (8,8))
        fig.autofmt_xdate(rotation = 45)
        
        # Plotting for Actual max temperature
        x1.plot(dates,self.features['actual'])
        x1.set_xlabel('');x1.set_ylabel('Temperature (F)');x1.set_title('Actual Max Temp')
        
        # Plotting for Max Temperature 1 day ago
        x2.plot(dates,self.features['temp_1'])
        x2.set_xlabel('');x2.set_ylabel('Temperature (F)');x2.set_title('Max Temp 1 day ago')
        
        # Plotting for Max Temperature 2 days ago
        x3.plot(dates,self.features['temp_2'])
        x3.set_xlabel('DATE');x3.set_ylabel('Temperature (F)');x3.set_title('Max Temp 2 Days Ago')
        
        # Plotting for the Estimate of Max Temperature
        x4.plot(dates,self.features['estimate'])
        x4.set_xlabel('DATE');x4.set_ylabel('Temperature (F)');x4.set_title('Estimate')
        plt.tight_layout(pad=2)
        
        # Defining the plotting layout
        figure,((x1,x2),(x3,x4))=plt.subplots(nrows=2,ncols=2,figsize = (8,8))
        figure.autofmt_xdate(rotation = 45)
        
        # Plotting for Historical Average Max Temp for the day
        x1.plot(dates,self.features['average'],'red')
        x1.set_xlabel('');x1.set_ylabel('Temperature (F)');x1.set_title('Historical Avg Max Temp')
        
        # Plotting for Avg Wind Speed 1 day ago
        x2.plot(dates,self.features['ws_1'],'red')
        x2.set_xlabel('');x2.set_ylabel('Wind Speed (mph)');x2.set_title('Wind Speed 1 day ago')
        
        # Plotting for Precipitation 1 day ago
        x3.plot(dates,self.features['prcp_1'], 'red')
        x3.set_xlabel('DATE');x3.set_ylabel('Precipitation (in)');x3.set_title('Precipitation 1 day ago')
        
        # Plotting for Snowdepth 1 day ago
        x4.plot(dates,self.features['snwd_1'], 'red')
        x4.set_xlabel('DATE');x4.set_ylabel('Snow Depth (in)');x4.set_title('Snow Depth 1 day ago')
        plt.tight_layout(pad=2)

# Class for visualizing the relationships between features and with target value using pairplot
class variables_relationship_visualization(variables_trend_visualization):
    # The class constructor
    def __init__(self,features):
        #Inheriting the variables_trend_visualization class
        super().__init__(features)
    
    #Method for plotting the pair-plots using seaborn to analyze relationships among variables    
    def variables_relationship(self):
        # Initializing the seasons list to classify months into seasons
        season=[]
        for month in self.features['month']:
            if month==1 or month==2 or month==12:
                season.append("Winter")
            elif month==3 or month==4 or month==5:
                season.append("Spring")
            elif month==6 or month==7 or month==8:
                season.append("Summer")
            else:
                season.append("Fall")
        # Chosing the variables of which we want to analyze the relationships        
        relationship_features=self.features[['temp_1','prcp_1','actual']]
        # To analyze the relationships with a clear picture through seasons
        relationship_features['season']=season
        
        #Plotting the pair-plots using seaborn
        
        sns.set(style="ticks",color_codes=True);
        # Creating a customized colour pallette using seaborn for pairplot
        palette=sns.xkcd_palette(['dark blue','dark green','gold','orange'])
        # Plotting the pair plot
        sns.pairplot(relationship_features,hue='season',diag_kind='kde',palette=palette,plot_kws=dict(alpha = 0.7),diag_kws=dict(shade=True));
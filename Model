#import libraries

import numpy as np
import matplotlib.pyplot as py
import pandas as pd
#import tensorflow as tf
import seaborn as sns

#from tensorflow.keras import layers
from matplotlib import colors

dataset = pd.read_csv('heart.csv')

print(dataset.shape)

dataset.head()

dataset.info()

dataset.describe()

dataset.corr()

sns.histplot(data=dataset, x="Age", binwidth=5, kde=True).set(title='Count of Age to Target')

# Using groupby() and count()
sex_group = dataset.groupby(['Sex'])['Sex'].count()
print(sex_group)

sns.histplot(data=dataset, x="Sex", binwidth=5, kde=True).set(title='Count of Sex to Target')

# Using groupby() and count()
chestpain = dataset.groupby(['ChestPainType'])['ChestPainType'].count()
print(chestpain)

sns.histplot(data=dataset, x="ChestPainType", binwidth=5, kde=True).set(title='Count of Chest Pain Type to Target')

sns.histplot(data = dataset,x = 'Age', hue='Sex' ,kde = True).set(title='Distribution of age in correlation gender')    

sns.histplot(data = dataset,x = 'Age', hue='ChestPainType' ,kde = True).set(title='Distribution of chest pain type between age groups')

py.figure(figsize =(15,10))
for i,col in enumerate(dataset.columns,1):
    py.subplot(4,3,i)
    py.title(f"Distribution of {col} Data")
    sns.histplot(dataset[col],kde = True)
    py.tight_layout()
    py.plot()

correlation_matrix = (dataset).corr()
py.figure(figsize=(15, 8))
sns.heatmap(correlation_matrix, annot=True, linewidths=1.0)
py.xticks(rotation=45, ha='right')
py.yticks(rotation=0)
py.show()

py.figure(figsize = (15,8))
sns.pairplot(dataset, hue = "HeartDisease")

# Finding outlierss with interquartile 

# Outliers for RestingBP
IQR = dataset.RestingBP.quantile(0.75) - dataset.RestingBP.quantile(0.25)
Lower_limit = dataset.RestingBP.quantile(0.25) - (IQR*3)
Upper_upper = dataset.RestingBP.quantile(0.75) + (IQR*3)
print(f"RestingBP outliers are values < {Lower_limit} or > {Upper_upper}")

# Outliers for Cholesterol
IQR = dataset.Cholesterol.quantile(0.75) - dataset.Cholesterol.quantile(0.25)
Lower_limit = dataset.Cholesterol.quantile(0.25) - (IQR*3)
Upper_upper = dataset.Cholesterol.quantile(0.75) + (IQR*3)
print(f"Cholesterol outliers are values < {Lower_limit} or > {Upper_upper}")

# Outliers for FastingBS
IQR = dataset.FastingBS.quantile(0.75) - dataset.FastingBS.quantile(0.25)
Lower_limit = dataset.FastingBS.quantile(0.25) - (IQR*3)
Upper_upper = dataset.FastingBS.quantile(0.75) + (IQR*3)
print(f"FastingBS outliers are values < {Lower_limit} or > {Upper_upper}")

# Outliers for MaxHR
IQR = dataset.MaxHR.quantile(0.75) - dataset.MaxHR.quantile(0.25)
Lower_limit = dataset.MaxHR.quantile(0.25) - (IQR*3)
Upper_upper = dataset.MaxHR.quantile(0.75) + (IQR*3)
print(f"MaxHR outliers are values < {Lower_limit} or > {Upper_upper}"

# Outliers for Oldpeak
IQR = dataset.Oldpeak.quantile(0.75) - dataset.Oldpeak.quantile(0.25)
Lower_limit = dataset.Oldpeak.quantile(0.25) - (IQR*3)
Upper_upper = dataset.Oldpeak.quantile(0.75) + (IQR*3)
print(f"MaxHR outliers are values < {Lower_limit} or > {Upper_upper}")

# Handeling Outliers 

def Max_value(dataset_engineered,variable,top):
    return np.where(dataset_engineered[variable] > top, top,dataset_engineered[variable])

dataset['RestingBP'].head(), 
dataset['Cholesterol'].shape, 
dataset['FastingBS'].shape, 
dataset['MaxHR'].shape, 
dataset['Oldpeak'].head()

dataset['RestingBP'] = Max_value(dataset,'RestingBP',200)
dataset['Cholesterol'] = Max_value(dataset,'Cholesterol',200)
dataset['FastingBS'] = Max_value(dataset,'FastingBS',200)
dataset['MaxHR'] = Max_value(dataset,'MaxHR',200)
dataset['Oldpeak'] = Max_value(dataset,'Oldpeak',200.0)

dataset.describe()

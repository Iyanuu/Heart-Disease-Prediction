#import libraries

import numpy as np
import matplotlib.pyplot as py
import pandas as pd
#import tensorflow as tf
import seaborn as sns

#from tensorflow.keras import layers
from matplotlib import colors
from sklearn.preprocessing import StandardScaler

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
print(f"MaxHR outliers are values < {Lower_limit} or > {Upper_upper}")

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

dataset = pd.get_dummies(dataset, columns=["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope"])
scaler = StandardScaler()
numerical_features = ["Age","RestingBP","Cholesterol","FastingBS","MaxHR","Oldpeak"]
dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])
py.figure(figsize=(20, 16))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f')
py.title('Correlation Matrix')
py.show()


######################################################### Model 1 ###############################################
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, average_precision_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense


# Reset indices of the dataset
dataset.reset_index(drop=True, inplace=True)

# split into input (X) and output (Y) variables
X = dataset.drop(['HeartDisease'], axis=1)
y = dataset['HeartDisease']

# Initialize 10-fold StratifiedKFold cross-validation
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store evaluation metrics for each fold
accuracy_scores = []
confusion_matrices = []
auc_pr_scores = []
cohen_kappa_scores = []
f1_scores = []
recall_scores = []
precision_scores = []

# Perform nested cross-validation
for train_index, val_index in cv_outer.split(X, y):
    X_train_fold, X_val_fold = X.loc[train_index], X.loc[val_index]
    y_train_fold, y_val_fold = y.loc[train_index], y.loc[val_index]

    # Create a function for Model 1
    def create_model():
        model = Sequential()
        model.add(Dense(units=128, activation='relu', input_shape=(20,), kernel_regularizer=l2(0.01)))
        model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(units=32, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Create the model
    model = create_model()

    # Fit the model on the training data
    model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)

    # Predict on validation data
    y_pred = model.predict(X_val_fold)

    # Convert probabilities to binary predictions
    y_pred_binary = np.round(y_pred)

    # Calculate evaluation metrics
    accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
    confusion_matrix_val = confusion_matrix(y_val_fold, y_pred_binary)
    auc_pr = average_precision_score(y_val_fold, y_pred)
    cohen_kappa = cohen_kappa_score(y_val_fold, y_pred_binary)
    f1 = f1_score(y_val_fold, y_pred_binary)
    recall = recall_score(y_val_fold, y_pred_binary)
    precision = precision_score(y_val_fold, y_pred_binary)

    accuracy_scores.append(accuracy)
    confusion_matrices.append(confusion_matrix_val)
    auc_pr_scores.append(auc_pr)
    cohen_kappa_scores.append(cohen_kappa)
    f1_scores.append(f1)
    recall_scores.append(recall)
    precision_scores.append(precision)

# Calculate mean values for evaluation metrics across all folds
mean_accuracy = np.mean(accuracy_scores)
mean_auc_pr = np.mean(auc_pr_scores)
mean_cohen_kappa = np.mean(cohen_kappa_scores)
mean_f1 = np.mean(f1_scores)
mean_recall = np.mean(recall_scores)
mean_precision = np.mean(precision_scores)

print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean AUC-PR: {mean_auc_pr:.4f}")
print(f"Mean Cohen's Kappa: {mean_cohen_kappa:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")

# Print Confusion Matrix for Model 1
print("\nConfusion Matrix - Model 1:")
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
print(mean_confusion_matrix)

# Visualise Confusion matrix - Define the confusion matrix values 
conf_matrix = mean_confusion_matrix

# Define class labels
class_labels = ['Negative', 'Positive']

# Create a heatmap
py.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
py.xlabel('Predicted Labels')
py.ylabel('True Labels')
py.title('Model 1 Confusion Matrix')
py.show()

######################################################### Model 2 ###############################################
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, average_precision_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense

# Reset indices of the dataset
dataset.reset_index(drop=True, inplace=True)

# split into input (X) and output (Y) variables
X = dataset.drop(['HeartDisease'], axis=1)
y = dataset['HeartDisease']

# Initialize 10-fold StratifiedKFold cross-validation
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store evaluation metrics for each fold
accuracy_scores = []
confusion_matrices = []
auc_pr_scores = []
cohen_kappa_scores = []
f1_scores = []
recall_scores = []
precision_scores = []

# Perform nested cross-validation
for train_index, val_index in cv_outer.split(X, y):
    X_train_fold, X_val_fold = X.loc[train_index], X.loc[val_index]
    y_train_fold, y_val_fold = y.loc[train_index], y.loc[val_index]

    # Create a function for Model 2
    def create_model():
        model = Sequential()
        model.add(Dense(units=64, activation='tanh', input_shape=(20,), kernel_regularizer=l2(0.01)))
        model.add(Dense(units=64, activation='tanh', kernel_regularizer=l2(0.01)))
        model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=l2(0.01)))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Create the model
    model = create_model()

    # Fit the model on the training data
    model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)

    # Predict on validation data
    y_pred = model.predict(X_val_fold)

    # Convert probabilities to binary predictions
    y_pred_binary = np.round(y_pred)

    # Calculate evaluation metrics
    accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
    confusion_matrix_val = confusion_matrix(y_val_fold, y_pred_binary)
    auc_pr = average_precision_score(y_val_fold, y_pred)
    cohen_kappa = cohen_kappa_score(y_val_fold, y_pred_binary)
    f1 = f1_score(y_val_fold, y_pred_binary)
    recall = recall_score(y_val_fold, y_pred_binary)
    precision = precision_score(y_val_fold, y_pred_binary)

    accuracy_scores.append(accuracy)
    confusion_matrices.append(confusion_matrix_val)
    auc_pr_scores.append(auc_pr)
    cohen_kappa_scores.append(cohen_kappa)
    f1_scores.append(f1)
    recall_scores.append(recall)
    precision_scores.append(precision)

# Calculate mean values for evaluation metrics across all folds
mean_accuracy = np.mean(accuracy_scores)
mean_auc_pr = np.mean(auc_pr_scores)
mean_cohen_kappa = np.mean(cohen_kappa_scores)
mean_f1 = np.mean(f1_scores)
mean_recall = np.mean(recall_scores)
mean_precision = np.mean(precision_scores)

print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean AUC-PR: {mean_auc_pr:.4f}")
print(f"Mean Cohen's Kappa: {mean_cohen_kappa:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")

# Print Confusion Matrix for Model 2
print("\nConfusion Matrix - Model 2:")
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
print(mean_confusion_matrix)

# Visualise Confusion matrix - Define the confusion matrix values 
conf_matrix = mean_confusion_matrix

# Define class labels
class_labels = ['Negative', 'Positive']

# Create a heatmap
py.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
py.xlabel('Predicted Labels')
py.ylabel('True Labels')
py.title('Model 2 Confusion Matrix')
py.show()

######################################################### Model 3 ###############################################
# Reset indices of the dataset
dataset.reset_index(drop=True, inplace=True)

# split into input (X) and output (Y) variables
X = dataset.drop(['HeartDisease'], axis=1)
y = dataset['HeartDisease']

# Initialize 10-fold StratifiedKFold cross-validation
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store evaluation metrics for each fold
accuracy_scores = []
confusion_matrices = []
auc_pr_scores = []
cohen_kappa_scores = []
f1_scores = []
recall_scores = []
precision_scores = []

# Perform nested cross-validation
for train_index, val_index in cv_outer.split(X, y):
    X_train_fold, X_val_fold = X.loc[train_index], X.loc[val_index]
    y_train_fold, y_val_fold = y.loc[train_index], y.loc[val_index]

    # Create a function for Model 3
    def create_model():
        model = Sequential()
        model.add(Dense(units=256, activation='relu', input_shape=(20,)))
        model.add(Dense(units=128, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=l2(0.01)))
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Create the model
    model = create_model()

    # Fit the model on the training data
    model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)

    # Predict on validation data
    y_pred = model.predict(X_val_fold)

    # Convert probabilities to binary predictions
    y_pred_binary = np.round(y_pred)

    # Calculate evaluation metrics
    accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
    confusion_matrix_val = confusion_matrix(y_val_fold, y_pred_binary)
    auc_pr = average_precision_score(y_val_fold, y_pred)
    cohen_kappa = cohen_kappa_score(y_val_fold, y_pred_binary)
    f1 = f1_score(y_val_fold, y_pred_binary)
    recall = recall_score(y_val_fold, y_pred_binary)
    precision = precision_score(y_val_fold, y_pred_binary)

    accuracy_scores.append(accuracy)
    confusion_matrices.append(confusion_matrix_val)
    auc_pr_scores.append(auc_pr)
    cohen_kappa_scores.append(cohen_kappa)
    f1_scores.append(f1)
    recall_scores.append(recall)
    precision_scores.append(precision)

# Calculate mean values for evaluation metrics across all folds
mean_accuracy = np.mean(accuracy_scores)
mean_auc_pr = np.mean(auc_pr_scores)
mean_cohen_kappa = np.mean(cohen_kappa_scores)
mean_f1 = np.mean(f1_scores)
mean_recall = np.mean(recall_scores)
mean_precision = np.mean(precision_scores)

print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean AUC-PR: {mean_auc_pr:.4f}")
print(f"Mean Cohen's Kappa: {mean_cohen_kappa:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")

# Print Confusion Matrix for Model 3
print("\nConfusion Matrix - Model 3:")
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
print(mean_confusion_matrix)

# Visualise Confusion matrix - Define the confusion matrix values 
conf_matrix = mean_confusion_matrix

# Define class labels
class_labels = ['Negative', 'Positive']

# Create a heatmap
py.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
py.xlabel('Predicted Labels')
py.ylabel('True Labels')
py.title('Model 3 Confusion Matrix')
py.show()


######################################################### Model 4 ###############################################
# Reset indices of the dataset
dataset.reset_index(drop=True, inplace=True)

# split into input (X) and output (Y) variables
X = dataset.drop(['HeartDisease'], axis=1)
y = dataset['HeartDisease']

# Initialize 10-fold StratifiedKFold cross-validation
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store evaluation metrics for each fold
accuracy_scores = []
confusion_matrices = []
auc_pr_scores = []
cohen_kappa_scores = []
f1_scores = []
recall_scores = []
precision_scores = []

# Perform nested cross-validation
for train_index, val_index in cv_outer.split(X, y):
    X_train_fold, X_val_fold = X.loc[train_index], X.loc[val_index]
    y_train_fold, y_val_fold = y.loc[train_index], y.loc[val_index]

    # Create a function for Model 4
    def create_model():
        model = Sequential()
        model.add(Dense(units=128, activation='elu', input_shape=(20,)))
        model.add(Dense(units=64, activation='elu', kernel_regularizer=l2(0.01)))
        model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=l2(0.01)))
        model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Create the model
    model = create_model()

   # Create the model
    model = create_model()

    # Fit the model on the training data
    model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)

    # Predict on validation data
    y_pred = model.predict(X_val_fold)

    # Convert probabilities to binary predictions
    y_pred_binary = np.round(y_pred)

    # Calculate evaluation metrics
    accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
    confusion_matrix_val = confusion_matrix(y_val_fold, y_pred_binary)
    auc_pr = average_precision_score(y_val_fold, y_pred)
    cohen_kappa = cohen_kappa_score(y_val_fold, y_pred_binary)
    f1 = f1_score(y_val_fold, y_pred_binary)
    recall = recall_score(y_val_fold, y_pred_binary)
    precision = precision_score(y_val_fold, y_pred_binary)

    accuracy_scores.append(accuracy)
    confusion_matrices.append(confusion_matrix_val)
    auc_pr_scores.append(auc_pr)
    cohen_kappa_scores.append(cohen_kappa)
    f1_scores.append(f1)
    recall_scores.append(recall)
    precision_scores.append(precision)

# Calculate mean values for evaluation metrics across all folds
mean_accuracy = np.mean(accuracy_scores)
mean_auc_pr = np.mean(auc_pr_scores)
mean_cohen_kappa = np.mean(cohen_kappa_scores)
mean_f1 = np.mean(f1_scores)
mean_recall = np.mean(recall_scores)
mean_precision = np.mean(precision_scores)

print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean AUC-PR: {mean_auc_pr:.4f}")
print(f"Mean Cohen's Kappa: {mean_cohen_kappa:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")

# Print Confusion Matrix for Model 4
print("\nConfusion Matrix - Model 4:")
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
print(mean_confusion_matrix)

# Visualise Confusion matrix - Define the confusion matrix values 
conf_matrix = mean_confusion_matrix

# Define class labels
class_labels = ['Negative', 'Positive']

# Create a heatmap
py.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
py.xlabel('Predicted Labels')
py.ylabel('True Labels')
py.title('Model 4 Confusion Matrix')
py.show()


######################################################### Model 5 ###############################################
from sklearn.metrics import confusion_matrix

# Reset indices of the dataset
dataset.reset_index(drop=True, inplace=True)

# split into input (X) and output (Y) variables
X = dataset.drop(['HeartDisease'], axis=1)
y = dataset['HeartDisease']

# Initialize 10-fold StratifiedKFold cross-validation
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store evaluation metrics for each fold
accuracy_scores = []
confusion_matrices = []
auc_pr_scores = []
cohen_kappa_scores = []
f1_scores = []
recall_scores = []
precision_scores = []

# Perform nested cross-validation
for train_index, val_index in cv_outer.split(X, y):
    X_train_fold, X_val_fold = X.loc[train_index], X.loc[val_index]
    y_train_fold, y_val_fold = y.loc[train_index], y.loc[val_index]

    # Create a function for Model 5
    def create_model():
        model = Sequential()
        model.add(Dense(units=128, activation='softmax', input_shape=(20,)))
        model.add(Dense(units=64, activation='softmax', kernel_regularizer=l2(0.01)))
        model.add(Dense(units=32, activation='softmax', kernel_regularizer=l2(0.01)))
        model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=l2(0.01)))
        model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Create the model
    model = create_model()

    # Create the model
    model = create_model()

    # Fit the model on the training data
    model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)

    # Predict on validation data
    y_pred = model.predict(X_val_fold)

    # Convert probabilities to binary predictions
    y_pred_binary = np.round(y_pred)

    # Calculate evaluation metrics
    accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
    confusion_matrix_val = confusion_matrix(y_val_fold, y_pred_binary)
    auc_pr = average_precision_score(y_val_fold, y_pred)
    cohen_kappa = cohen_kappa_score(y_val_fold, y_pred_binary)
    f1 = f1_score(y_val_fold, y_pred_binary)
    recall = recall_score(y_val_fold, y_pred_binary)
    precision = precision_score(y_val_fold, y_pred_binary)

    accuracy_scores.append(accuracy)
    confusion_matrices.append(confusion_matrix_val)
    auc_pr_scores.append(auc_pr)
    cohen_kappa_scores.append(cohen_kappa)
    f1_scores.append(f1)
    recall_scores.append(recall)
    precision_scores.append(precision)

# Calculate mean values for evaluation metrics across all folds
mean_accuracy = np.mean(accuracy_scores)
mean_auc_pr = np.mean(auc_pr_scores)
mean_cohen_kappa = np.mean(cohen_kappa_scores)
mean_f1 = np.mean(f1_scores)
mean_recall = np.mean(recall_scores)
mean_precision = np.mean(precision_scores)

print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean AUC-PR: {mean_auc_pr:.4f}")
print(f"Mean Cohen's Kappa: {mean_cohen_kappa:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")

# Print Confusion Matrix for Model 5
print("\nConfusion Matrix - Model 5:")
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
print(mean_confusion_matrix)

# Visualise Confusion matrix - Define the confusion matrix values 
conf_matrix = mean_confusion_matrix

# Define class labels
class_labels = ['Negative', 'Positive']

# Create a heatmap
py.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
py.xlabel('Predicted Labels')
py.ylabel('True Labels')
py.title('Model 5 Confusion Matrix')
py.show()

################################################################################# Hyper Paramter Tuning #################################################################################
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, average_precision_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch

# Reset indices of the dataset
dataset.reset_index(drop=True, inplace=True)

# Split into input (X) and output (y) variables
X = dataset.drop(['HeartDisease'], axis=1)
y = dataset['HeartDisease']

# Initialize 10-fold StratifiedKFold cross-validation
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store evaluation metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
cohen_kappa_scores = []
roc_auc_scores = []
auc_pr_scores = []
confusion_matrices = []

# Perform nested cross-validation
for train_index, val_index in cv_outer.split(X, y):
    X_train_fold, X_val_fold = X.loc[train_index], X.loc[val_index]
    y_train_fold, y_val_fold = y.loc[train_index], y.loc[val_index]

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Create a function for Model 1
    def create_model(hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=256, step=32), activation=hp.Choice('activation_1', values=['relu', 'tanh']), input_shape=(20,), kernel_regularizer=l2(0.01)))
        model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=256, step=32), activation=hp.Choice('activation_2', values=['relu', 'tanh']), kernel_regularizer=l2(0.01)))
        model.add(Dense(units=hp.Int('units_3', min_value=32, max_value=256, step=32), activation=hp.Choice('activation_3', values=['relu', 'tanh']), kernel_regularizer=l2(0.01)))
        model.add(Dense(units=1, activation='sigmoid'))

        optimizer_choice = hp.Choice('optimizer', values=['RMSprop', 'adam'])
        if optimizer_choice == 'RMSprop':
            optimizer = RMSprop(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
        else:
            optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Implement Early Stopping and Learning Rate Scheduling
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler_cb = LearningRateScheduler(lambda epoch, lr: lr * 0.95 if epoch % 2 == 0 else lr)

    # Hyperparameter tuning using RandomSearch from keras-tuner
    tuner = RandomSearch(
        create_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='my_dir',
        project_name='heart_disease_tuning'
    )

    tuner.search(X_train_fold, y_train_fold, epochs=20, validation_data=(X_val_fold, y_val_fold),
                 callbacks=[early_stopping, lr_scheduler_cb], class_weight=class_weight_dict)

    best_model = tuner.get_best_models(num_models=1)[0]

    # Print details of the best model
    print("Best Model Summary:")
    best_model.summary()

    # Print the best hyperparameters chosen by the tuner
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best Hyperparameters:")
    print(best_hp)

    # Predict on validation data
    y_pred = best_model.predict(X_val_fold)

    # Convert probabilities to binary predictions
    y_pred_binary = np.round(y_pred)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_val_fold, y_pred_binary)
    precision = precision_score(y_val_fold, y_pred_binary)
    recall = recall_score(y_val_fold, y_pred_binary)
    f1 = f1_score(y_val_fold, y_pred_binary)
    cohen_kappa = cohen_kappa_score(y_val_fold, y_pred_binary)
    roc_auc = roc_auc_score(y_val_fold, y_pred)
    auc_pr = average_precision_score(y_val_fold, y_pred)

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    cohen_kappa_scores.append(cohen_kappa)
    roc_auc_scores.append(roc_auc)
    auc_pr_scores.append(auc_pr)
    confusion_matrices.append(confusion_matrix(y_val_fold, y_pred_binary))

# Calculate mean evaluation metrics across all folds
mean_accuracy = np.mean(accuracy_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1 = np.mean(f1_scores)
mean_cohen_kappa = np.mean(cohen_kappa_scores)
mean_roc_auc = np.mean(roc_auc_scores)
mean_auc_pr = np.mean(auc_pr_scores)

print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean Cohen's Kappa: {mean_cohen_kappa:.4f}")
print(f"Mean ROC AUC: {mean_roc_auc:.4f}")
print(f"Mean AUC-PR: {mean_auc_pr:.4f}")

# Print Confusion Matrix for Best Model
print("\nConfusion Matrix - Best Model:")
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
print(mean_confusion_matrix)

# Visualize Confusion matrix
class_labels = ['Negative', 'Positive']
plt.figure(figsize=(8, 6))
sns.heatmap(mean_confusion_matrix, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Best Model Confusion Matrix')
plt.show()



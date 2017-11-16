# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:48:16 2017

@author: Kevin

TODO: More visualizations
TODO: SVM, Random forest
TODO: General Cleaning

Great resource for visualisations
https://www.kaggle.com/ash316/learn-pandas-with-pokemons
http://thelillysblog.com/2017/08/18/machine-learning-k-fold-validation/
https://machinelearningmastery.com/evaluate-performance-machine-learning-algorithms-python-using-resampling/
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns

import matplotlib.pyplot as plt
%pylab inline

pylab.rcParams['figure.figsize'] = (8.0, 7.0)

#os.chdir('C:/Users/Kevin/Desktop/Dropbox/Dropbox/Brainstation - Project/')
os.chdir('C:/Users/knguyen/Dropbox/Brainstation - Project/')

# Import data
df = pd.read_csv('Pokemon.csv')
df2 = pd.read_csv('pokemon_stats.csv')

''' Data Cleaning and Merging '''

# To ensure consistency between datasets
df['Name'] = df['Name'].str.lower()

# Fix name issues between datasets
df = df.replace({'Name': {'nidoran♀': 'nidoran-f', 'nidoran♂': 'nidoran-m', "farfetch'd": 'farfetchd',
                                'mr. mime': 'mr-mime', 'deoxysnormal forme': 'deoxys-normal', 'wormadamplant cloak': 'wormadam-plant',
                                'mime jr.': 'mime-jr', 'giratinaaltered forme': 'giratina-altered', 'shayminland forme': 'shaymin-land',
                                'basculin': 'basculin-red-striped', 'darmanitanstandard mode': 'darmanitan-standard',
                                'tornadusincarnate forme': 'tornadus-incarnate', 'thundurusincarnate forme': 'thundurus-incarnate',
                                'landorusincarnate forme': 'landorus-incarnate', 'keldeoordinary forme': 'keldeo-ordinary',
                                'meloettaaria forme': 'meloetta-aria', 'flabébé': 'flabebe', 'meowsticmale': 'meowstic-male',
                                'aegislashshield forme': 'aegislash-shield', 'pumpkabooaverage size': 'pumpkaboo-average',
                                'gourgeistaverage size': 'gourgeist-average', 'hoopahoopa confined': 'hoopa'}})

# Merge both datasets
df_merged = pd.merge(df, df2, left_on='Name', right_on='identifier', how='left')

# Drop unnecessary rows and columns
df_merged = df_merged[df_merged['is_default'] == 1]
cols_to_keep = ['Name', 'Type 1', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed',
                'Generation', 'Legendary', 'height', 'weight', 'base_experience']
df_merged = df_merged[cols_to_keep]

# View the data
df_merged.head(5)

# How much are of each type (we are only using type 1)
df_merged['Type 1'].value_counts().plot(kind='barh', color='lightblue')

# Whoa, there are only 4 flying pokemon! Who are they?
df_merged[df['Type 1'] == 'Flying']


# Domain knowledge tells us Noibat and its evolution are the only Pokemon with flying as their main type
# In addition, they are actually the weakest pokemon with "dragon" types
# lets remove all 4 of these pokemon
# https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
df_merged = df_merged[df_merged['Type 1'] != 'Flying']


''' Data Visualizations '''

# Who has the strongest def?
print("Defense:", df_merged['Defense'].argmax())

# How powerful is each generation
plt.title('Total Power by Generation')
sns.violinplot(x = 'Generation', y='Total', data=df_merged)
#sns.plt.show()

# Where do legendaries fall within the power level in each type?
plt.figure(figsize=(12,6))
sns.swarmplot(x='Type 1', y='Total', data=df_merged, hue='Legendary')
plt.axhline(df_merged['Total'].mean(), color='red', linestyle='dashed')
plt.show()

# Visualize the distribution of each type in a pie chart (to do)


''' Preparing training and testing data for algorithms '''


# Preparing our train and test data
# Set legendary False = 0, True = 1
df_merged['Legendary'] = np.where(df_merged['Legendary'] == True, 1, 0)


# Take out a random sample of 5% of EACH TYPE of pokemon to be used as the test data
type_list = list(df_merged['Type 1'].unique())

df_test = pd.DataFrame(columns = df_merged.columns)
for i in type_list:
    df_test = df_test.append((df_merged[df_merged['Type 1'] == i]).sample(frac=0.05))

# Now remove the sample rows from the main dataset
# https://stackoverflow.com/questions/16704782/python-pandas-removing-rows-from-a-dataframe-based-on-a-previously-obtained-su
df_train = df_merged[~df_merged['Name'].isin(df_test['Name'])]


# Standardize and scale our numeric data
# Scale our data from [-1,1]
from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
max_abs_scaler = preprocessing.MaxAbsScaler()


#X_train = min_max_scaler.fit_transform(df_train.iloc[:,2:13])
#X_test = min_max_scaler.fit_transform(df_test.iloc[:,2:13])

X_train = max_abs_scaler.fit_transform(df_train.iloc[:,2:13])
X_test = max_abs_scaler.fit_transform(df_test.iloc[:,2:13])

# Create an arrays for our labels
array_train = df_train.values
array_test = df_test.values


Y_train = array_train[:,1]
Y_test = array_test[:,1]

''' Algorithm - Naive Bayes with K-Folds Cross Validation '''


import nltk
#from sklearn.model_selection import KFold

# 10-fold cross validation
from sklearn import model_selection

# Prepare the kfold model
kfold = model_selection.KFold(n_splits=10, shuffle=True)

# Leave one out cross validation model
loocv = model_selection.LeaveOneOut()

# Attempt naives bayes
# http://scikit-learn.org/stable/modules/cross_validation.html
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)

#pred = gnb.predict(X_test)
# Calculate accuracy
#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(pred, Y_test)

# lets try naive bayes with kfolds
results = model_selection.cross_val_score(gnb, X_test, Y_test, cv=kfold)
results.mean()

# What about leave one out cv?
results2 = model_selection.cross_val_score(gnb, X_test, Y_test, cv=loocv)
results2.mean()


''' Algorithm - Support Vector Machines (SVM) with K-Folds Cross Validation '''
# Start up SVM



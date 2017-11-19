# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:48:16 2017

@author: Kevin

SKlearn cheatsheets:
https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf
https://www.datacamp.com/community/blog/scikit-learn-cheat-sheet

Resources Used:
http://www.tradinggeeks.net/2015/08/calculating-correlation-in-python/
https://stackoverflow.com/questions/16704782/python-pandas-removing-rows-from-a-dataframe-based-on-a-previously-obtained-su
http://scikit-learn.org/stable/modules/cross_validation.html
https://chrisalbon.com/machine-learning/cross_validation_parameter_tuning_grid_search.html
https://stats.stackexchange.com/questions/95797/how-to-split-the-dataset-for-cross-validation-learning-curve-and-final-evaluat
https://stackoverflow.com/questions/41407451/rbf-svm-parameters-using-gridsearchcv-in-scikit-learn-typeerror-kfold-o
https://www.kaggle.com/abcsds/pokemon
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


''' Data Visualizations '''


# What is the proportion of each type in our dataset?
df_merged['Type 1'].value_counts().plot(kind='barh', color='lightblue')

# Lets view the proportion by %'s instead
labels = df_merged['Type 1'].value_counts().index
df_merged['Type 1'].value_counts().plot(kind='pie', labels=labels, autopct='%1.0f%%')

# There is 0% flying types! (rounded) - who are they?
df_merged[df['Type 1'] == 'Flying']


'''
Domain knowledge tells us Noibat and its evolution are the only Pokemon with flying as their main type.
In addition, they are actually the weakest pokemon with "dragon" types.
Let us remove the flying types
''' 
df_merged = df_merged[df_merged['Type 1'] != 'Flying']

# Which Pokemon has the highest value for each stat?
variables_list = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'height', 'weight', 'base_experience']

for i in variables_list:    
    print(df_merged.groupby(['Name'], as_index=False)['%s' % i].max().sort_values(by=['%s' % i], ascending=False).head(3))

# Who is the 'strongest' Pokemon of each type?
strongest = df_merged.sort_values(by='Total', ascending=False)
strongest.drop_duplicates(subset=['Type 1'], keep='first')
    
# How powerful is each generation
plt.title('Total Power by Generation')
sns.violinplot(x = 'Generation', y='Total', data=df_merged)
#sns.plt.show()

# Where do legendaries fall within the power level in each type?
plt.figure(figsize=(12,6))
sns.swarmplot(x='Type 1', y='Total', data=df_merged, hue='Legendary')
plt.axhline(df_merged['Total'].mean(), color='red', linestyle='dashed')

# Are the variables correlated to each other?
corr_df = df_merged.corr(method='pearson')

mask = np.zeros_like(corr_df)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr_df, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=4.5, annot=True)


''' Preparing training and testing data for algorithms '''


# Preparing our train and test data
# Set legendary False = 0, True = 1
df_merged['Legendary'] = np.where(df_merged['Legendary'] == True, 1, 0)


# Remove out a random sample of 5% of EACH TYPE of pokemon to be used as the test data
type_list = list(df_merged['Type 1'].unique())

df_test = pd.DataFrame(columns = df_merged.columns)
for i in type_list:
    df_test = df_test.append((df_merged[df_merged['Type 1'] == i]).sample(frac=0.05))

# Now remove the sample rows from the main dataset
df_train = df_merged[~df_merged['Name'].isin(df_test['Name'])]


# Standardize and scale our numeric data
from sklearn import preprocessing
max_abs_scaler = preprocessing.MaxAbsScaler()

X_train = max_abs_scaler.fit_transform(df_train.iloc[:,2:13])
X_test = max_abs_scaler.fit_transform(df_test.iloc[:,2:13])

# Create an arrays for our labels
array_train = df_train.values
array_test = df_test.values

Y_train = array_train[:,1]
Y_test = array_test[:,1]


''' Algorithm One - Naive Bayes '''


#from sklearn.model_selection import KFold

# 10-fold cross validation
from sklearn import model_selection

# Prepare the kfold model
kfold = model_selection.KFold(n_splits=10, shuffle=True)

# Leave one out cross validation model
loocv = model_selection.LeaveOneOut()

# Train the Naive Bayes model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)

#pred = gnb.predict(X_test)
# Calculate accuracy
#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(pred, Y_test)

# Naives Bayes with Kfolds on our Test Data
results = model_selection.cross_val_score(gnb, X_test, Y_test, cv=kfold)
results.mean()

# Naive Bayes with LOOCV on our Test Data
results2 = model_selection.cross_val_score(gnb, X_test, Y_test, cv=loocv)
results2.mean()


''' Algorithm Two  - Support Vector Machines (SVM) '''


# Import SVM
from sklearn.svm import SVC
from sklearn import svm
estimator = SVC(kernel='linear')

# Let us conduct parameter tuning with GridSearchCV to optimize our results
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Prep up parameters to tune
svm_parameters = [
                  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
                  ]
                  
# Train our SVM classifier                  
svm_classifier = GridSearchCV(estimator=estimator, cv=kfold, param_grid=svm_parameters)
svm_classifier.fit(X_train, Y_train)

# The best parameters with the score:
print('Best score for data:', svm_classifier.best_score_)
print('Best C:',svm_classifier.best_estimator_.C) 
print('Best Kernel:',svm_classifier.best_estimator_.kernel)
print('Best Gamma:',svm_classifier.best_estimator_.gamma)

# Test our trained SVM classifier with our test data
svm_classifier.score(X_test, Y_test)

#svm.SVC(C=1000, kernel='linear', gamma='auto').fit(X_train, Y_train).score(X_test, Y_test)

''' Algorithm - Random Forest '''
from sklearn.ensemble import RandomForestClassifier

# Prep parameters and the Random Forest classifier
forest_clf = RandomForestClassifier()
forest_parameters = {
                        "max_features": [2, 5, 10, 'auto'],
                        "n_estimators": [50, 100, 200],
                        "criterion": ["gini", "entropy"],
                        "min_samples_leaf": [1,2,4,6],
                    }
                    
forest_classifier = GridSearchCV(estimator=forest_clf, param_grid=forest_parameters)

# Train our Random Forest Classifier - This will take very long to run!
forest_classifier.fit(X_train, Y_train)

# The best parameters with the score:
print('Best score:', forest_classifier.best_score_)
print('Best Max_Features:', forest_classifier.best_estimator_.max_features)
print('Best N_estimators:', forest_classifier.best_estimator_.n_estimators)
print('Best criterion:', forest_classifier.best_estimator_.criterion)
print('Best Min_samples_leaf:', forest_classifier.best_estimator_.min_samples_leaf)

# Test our trained Random Forest classifier with our test data
forest_classifier.score(X_test, Y_test)

# Compare algorithms computational difficulty
import time
start_time = time.time()
gnb.fit(X_train, Y_train)
print("Naive Bayes took", time.time() - start_time, "seconds to train.")

start_time = time.time()
svm_classifier.fit(X_train, Y_train)
print("SVM took ", time.time() - start_time, "seconds to train.")

start_time = time.time()
forest_classifier.fit(X_train, Y_train)
print("Random Forest took", time.time() - start_time, "seconds to train.")

# Visualize results
labels = ['Naive Bayes', 'SVM', 'Random Forest']
scores = [0.092, 0.235, 0.147]
train_time = [0.002, 8.379, 62.117]

fig, axs = plt.subplots(nrows=1, ncols=2)
sns.set(style="darkgrid", context="talk")
sns.barplot(x=labels, y=scores, ax=axs[0])
sns.barplot(x=labels, y=train_time, ax=axs[1])
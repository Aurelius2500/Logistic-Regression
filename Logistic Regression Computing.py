# -*- coding: utf-8 -*-
"""
Data Analytics Computing Follow Along: Logistic Regression With Python And Introduction to Classification
Spyder version 5.1.5
"""

# Import required packages. We need these packages for the script to run
# Think of a package as a ready-to-use piece of software 

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# We will perform both linear and logistic regression this time and compare them
# If you do not have the required packages, run the following lines

# pip install pandas
# pip install sklearn
# pip install seaborn

print("Packages imported")

# The lines below import the dataset that we will use in this script
# The link in kaggle is in the description, this is a salaries dataset

# We will be using pandas.read_csv to import our data

income_df = pd.read_csv('C:/Videos/archive/salaries_clean.csv')

# We can get the shape of the dataframe with the shape method

income_df.shape

# The print function just shows in the console the text that you give it    
    
print("Dataset imported")

# Get relevant columns

income_df_sub = income_df[['employer_name', 'job_title', 'job_title_category',
                           'total_experience_years', 'annual_base_pay', 
                           'employer_experience_years']]

# We can see the null values below

income_df_sub.isna().sum()

# We need to drop the null values for Logistic Regression to work

income_df_clean = income_df_sub.dropna()

# For this classification problem, a High Income variable will be created
# We will say that if the salary is more than 120000, then it will be high income

income_df_clean['High Income?'] = pd.cut(income_df_clean['annual_base_pay'], 
                                         bins = [-1, 120000, float('Inf')], 
                                         labels = [0, 1])

# See the columns and the first rows of the dataset

print(income_df_clean.columns)

print(income_df_clean.head())

# We will be using employer_experience_years as our predictor
# High Income? will be our prediction
# The relationship is that the number of years with the current employer can be used to predict high income
# We want to know if the number of years with an employer can be used to predict if a position is high income

X = income_df_clean[['employer_experience_years']]

y = income_df_clean['High Income?']

# Why not use a linear regression model like before?
# Import the linear regression model from sklearn

linear_model = LinearRegression()

# Now we can fit the model

linear_model.fit(X, y)

linear_model.score(X, y)

linear_predictions = linear_model.predict(X)

# See how the R-squared value is pretty bad
# Let's inspect our graph for Simple Linear Regression

plt.ticklabel_format(style='plain')
plt.xlabel("Years at Company")
plt.ylabel("High Income Indicator Prediction")
plt.title("Linear Regression on High Income Indicator vs Years at Company")

# The lines above are optional, merely for aesthetic purposes

plt.scatter(X, y, color = "Blue", s = 10)
plt.plot(X["employer_experience_years"], linear_predictions, color = "black", linewidth = 2)
plt.show()

# The problem is noticiable, the regression line very off, and goes to more than 1
# This is because this is not a regression problem, but a classification one
# We will need to use Logistic Regression for this, a similar approach to Linear Regression
# Logistic Regression is somewhat of a misnomer, because it is not a regression model

# We can confirm this by checking at the maximum of the predictions

max(linear_predictions)

# Because the outcome is more than 1, we cannot use this model to predict the class
# Use Logistic Regression instead

log_model = LogisticRegression()

# Train the Logistic Regression Model in the data

log_model.fit(X, y)

# Make sure that the order is correct, as we are predicting the probability of the classes

log_model.classes_

# For classification, instead of score we will use a confusion matrix

log_predictions = log_model.predict(X)

# Use the confusion matrix function
# For the confusion matrix, we will be looking at the top left and bottom right values
# The top left value is the true positives
# The bottom right value is the true negatives

confusion_matrix(y, log_predictions)

# See how the predict_proba function brings two probabilities that always add up to one

print(log_model.predict_proba(X))

# Get the first class probabilities for Logistic Regression

log_predictions_prob = log_model.predict_proba(X)[:, 1]

# Plot the model, the probabilities are in orange

plt.ticklabel_format(style='plain')
plt.xlabel("Years at Company")
plt.ylabel("High Income Indicator")
plt.title("Logistic Regression Probabilities on High Income Indicator vs Years at Company")

plt.scatter(X, y)
plt.scatter(X, log_predictions_prob)
plt.show()

# A value is classified as 1 if the probability is more than 0.5, and 0 if not
# This is know as a cutoff value or a threshold
# Compare this plot by putting the predictions on the y axis instead of the actual values

plt.ticklabel_format(style='plain')
plt.xlabel("Years at Company")
plt.ylabel("High Income Indicator (Predicted)")
plt.title("Logistic Regression Probabilities on High Income Indicator vs Years at Company")

plt.scatter(X, log_predictions)
plt.scatter(X, log_predictions_prob)
plt.show()

# The cutoff is pretty visible at around 15 years at the company
# Notice how this is still a linear model, but for classification
# As with linear regression, we can get the intercept and coeficient, but we cannot interpret them as easily

log_model.intercept_

log_model.coef_

# We can also get the score, although it is not the R-squared for Linear Regression
# Instead, this is the number of correct predictions over the total number of observation

log_model.score(X, y)

# Looking at the confusion matrix, we can calculate this ourselves

1076/(1076 + 13 + 486 + 9)

# This metric is also called the accuracy
# Finally, we can get a more detailed report with the classification report function

print(classification_report(y, log_predictions))

# Let's check our assumption on Linear Regression about predicting an X value outside of the range
out_of_range_prediction = log_model.predict(np.array(70).reshape(-1, 1))
float(out_of_range_prediction)

# We can use seaborn, a powerful visualization library, to see the famous logit function
sns.regplot(x = X, y = y, logistic = True, ci = None, line_kws = {"color" : "orange"})
plt.xlabel("Years at Company")
plt.ylabel("High Income Indicator (Predicted)")
plt.title("Logistic Regression Visualization")
plt.show()
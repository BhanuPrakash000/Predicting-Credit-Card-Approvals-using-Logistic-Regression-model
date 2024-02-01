# Predicting Credit-Card Approval Using Logistic Regression Model
## _Description_

Banks receive many applications for credit cards. Many get rejected due to many reasons, like low-income levels or based on individula credit report. As time passes this can be done by the use of machine learning, In this project, the automatic credit card approval predictor using machine learning technique. The dataset used in this project is the crdit card dataset from UCI Machine Learning Repository.
## Import python libraries
Let's import the primary pacakges like pandas to work with data, Numpy works with array, Scikit learn to split data, build and evaluate the classification models and matplotlib to visualise the data. In this project i have provided the code snippets in google colab environment or else you can try this code in jupyter notebook.
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
```
## Instructions to run the code
- At first you can find the dataset from [UCI Machine Learning Repository.](https://archive.ics.uci.edu/dataset/27/credit+approval)
- Next, upload the dataset to colab files and that is read to df variable.
``` python.py
df = pd.read_csv('/content/crx.data',header=None)
header_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
df = pd.read_csv('/content/crx.data',names=header_names)
```
- Upon the code works good as dataset is uploaded to colab. 


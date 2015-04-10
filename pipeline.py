__author__ = 'nickroth'


# coding: utf-8

# In[2]:

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# ## Scikit-Learn

# In[6]:

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


# ## Pre-Processed Data

# In[19]:

from datasets import get_data

X, X_0, Y = get_data()


# ### Principle Components Analysis

# In[21]:

n_components = [3, 4, 5]

pca = PCA()
logistic = LogisticRegression()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

estimator = GridSearchCV(pipe,
                         dict(
                             pca__n_components=n_components,
                             ))
estimator.fit(X, Y)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()
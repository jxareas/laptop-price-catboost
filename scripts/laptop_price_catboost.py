#!/usr/bin/env python
# coding: utf-8

# # eBay Laptops & Notebooks - Modeling
# 

# ### Loading Libraries
# 
# 

# In[ ]:


# Importing libraries and setting constants

# Libraries
import polars as pl

# Constants
RANDOM_SEED = 287
DATA_SOURCE_PATH = '../data/ebay_laptops_and_notebooks_cleansed.csv'



# ## Data Preparation

# ### Loading the dataset

# In[ ]:


df = pl.read_csv(DATA_SOURCE_PATH)
df.head(n=10).to_pandas()


# ### Dropping duplicates

# In[ ]:


df = df.unique()
df.head(n=10).to_pandas()


# ## Exploratory Data Analysis

# In[ ]:


# TODO : Exploratory Data Analysis


# ## Feature Engineering

# In[ ]:


# TODO : Feature Engineering


# ## Machine Learning

# In[ ]:


# TODO : Machine Learning


# ## Hyperparameter Tuning

# In[ ]:


# TODO : Hyperparameter Tuning


# ## Explainable AI - SHAP

# In[ ]:


# TODO: Explainable AI - SHAP


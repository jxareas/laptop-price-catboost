#!/usr/bin/env python
# coding: utf-8

# # eBay Laptops & Netbooks - Modeling
# 
# In this project, we aim to build a predictive model for **estimating laptop prices** using a dataset which contains cleaned information from eBay's Laptops & Netbooks category, originally obtained via web scraping, which includes product attributes such as brand, specifications, and other listing details.
# 
# The model utilizes **CatBoost** (*Categorical Boosting*), a state-of-the-art gradient boosting library [developed by Yandex][Yandex CatBoost], renowned for its efficiency in handling categorical features and its strong performance in regression, classification & ranking.
# 
# <div align="center">
# <img src="../assets/logos/catboost_logo.png" height="200" width="200"/>
# </div>
# 
# [Yandex Catboost]: https://yandex.com/dev/catboost/

# ## Links & Information
# 
# **Project Repository** - GitHub: [Laptop Price Prediction with CatBoost][Project Code]
# 
# [Project Code]: https://github.com/jxareas/laptop-price-catboost
# 

# ### Loading Libraries
# 
# 

# In[1]:


# Importing libraries and setting constants

# Libraries
import polars as pl

# Constants
RANDOM_SEED = 287
DATA_SOURCE_PATH = '../data/ebay_laptops_and_netbooks_cleansed.csv'



# ## Data Preparation

# ### Loading the dataset

# In[2]:


df = pl.read_csv(DATA_SOURCE_PATH)
df.head(n=10).to_pandas()


# ### Dropping duplicates

# In[3]:


df = df.unique()
df.head(n=10).to_pandas()


# ## Exploratory Data Analysis

# In[4]:


# TODO : Exploratory Data Analysis


# ## Feature Engineering

# In[5]:


# TODO : Feature Engineering


# ## Machine Learning

# In[6]:


# TODO : Machine Learning


# ## Hyperparameter Tuning

# In[7]:


# TODO : Hyperparameter Tuning


# ## Explainable AI - SHAP

# In[8]:


# TODO: Explainable AI - SHAP


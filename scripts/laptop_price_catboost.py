#!/usr/bin/env python
# coding: utf-8

# # eBay Laptops & Netbooks - Modeling
# 
# In this project, we aim to build a predictive model for **estimating laptop prices** using a dataset which contains cleaned information from eBay's Laptops & Netbooks category, originally obtained via web scraping, which includes product attributes such as brand, specifications, and other listing details.
# 
# The model utilizes **CatBoost** (*Categorical Boosting*), a state-of-the-art gradient boosting library [developed by Yandex][Yandex CatBoost], renowned for its efficiency in handling categorical features and its strong performance in regression, classification & ranking.
# 
# <div align="center">
# <img src="../assets/logos/catboost_logo.png" height="225" width="225"/>
# </div>
# 
# [Yandex Catboost]: https://yandex.com/dev/catboost/

# ## Links & Information
# 
# **Project Repository** - GitHub: [Laptop Price Prediction with CatBoost][Project Code]
# 
# [Project Code]: https://github.com/jxareas/laptop-price-catboost
# 

# ### Importing Libraries
# 
# In this project, we will leverage several powerful libraries for efficient data manipulation, feature engineering, modeling, and hyperparameter tuning.
# 
# - **Scikit-learn**: Used for **training the model**, performing model evaluation, and splitting the dataset in train, validation & test sets.
# - **CatBoost**: The **core library** for modeling, used to design a gradient boosting model to handle categorical features effectively and provide good performance in regression tasks.
# - **Optuna**: Utilized for **hyperparameter tuning**, automating the process of finding the best model parameters to improve performance.
# - **Feature Engine**: A library for **feature engineering** that provides various techniques for transforming and selecting features, ensuring that our model has the most relevant data.
# - **Seaborn**: Used for **exploratory data analysis (EDA)** and plotting, providing intuitive and high-level visualizations to understand the dataset and its relationships.
# - **SHAP**: Used for **model interpretability**, providing insights into how each feature influences the model's predictions, helping to explain the decisions made by the model & empowering **Explainable AI**.
# 
# These libraries will work together to ensure a streamlined and efficient manner for building and optimizing the predictive model.
# 

# In[1]:


# Importing libraries and setting constants

# Libraries
import polars as pl
import seaborn as sns
import seaborn.objects as so
from feature_engine.encoding import RareLabelEncoder
from scipy import stats
import numpy as np
from sklearn.preprocessing import PowerTransformer
from textblob import TextBlob
from typing import Tuple

# Constants
RANDOM_SEED = 287
DATA_SOURCE_PATH = '../data/ebay_laptops_and_netbooks_cleansed.csv'


# ## Data Preparation

# ### Loading the dataset

# In[2]:


df = pl.read_csv(DATA_SOURCE_PATH)
df.describe()


# ## Exploratory Data Analysis
# 

# In[3]:


# TODO : Exploratory Data Analysis


# In[4]:


(
    so.Plot(df, x='min_price')
    .add(so.Bars(color='royalblue'), so.Hist('density'))
    .add(so.Area(color='red', alpha=.15), so.KDE())
    .label(title='Minimum Price KDE')
)


# In[5]:


## TODO : More EDA - AutoEDA? (SweetViz, AutoViz, pandas-profiling) or manual? TBD


# ## Feature Engineering
# 

# ### Feature Creation

# #### **Sentiment Analysis** for `seller_note`
# 
# In this section, we perform sentiment analysis on the `seller_note` feature from the eBay dataset using [TextBlob][TextBlob], a simple library that helps analyze and process textual data, built upon python's [Natural Language Toolkit][NLTK]. The goal is to evaluate the sentiment expressed in the seller's notes, identifying whether the comments are **positive**, **negative**, or **neutral**.
# 
# We calculate the *polarity* and *subjectivity* of each note to understand the sentiment's intensity and whether it reflects personal opinions or objective facts. This simple analysis can help **assess the overall tone of the seller's descriptions**.
# 
# Notes:
# - **Polarity** measures the sentiment's intensity, ranging from **-1** (negative) to **+1** (positive).
#   A score close to **0** indicates neutral sentiment.
# - **Subjectivity** measures how subjective or opinionated the text is, ranging from **0** (very objective) to **1** (very subjective). Higher subjectivity indicates that the text reflects personal opinions or feelings, while lower subjectivity suggests it is more factual.
# 
# <div align="center">
# <img src="../assets/images/textblob_working.png" height="100" width="550"/>
# <br>
# <a href="https://www.researchgate.net/figure/Working-of-the-TextBlob-method_fig7_350191934"><i>Working of the TextBlob method</i></a>
# </div>
# 
# [TextBlob]: https://github.com/sloria/TextBlob
# [NLTK]: https://github.com/nltk/nltk

# In[6]:


df.select(
    'seller_note'
).filter(
    pl.col('seller_note').is_not_null()
).head(n=10).to_pandas()


# In[7]:


def get_sentiment_label_by_polarity(polarity: float, threshold: float = 0.1) -> str:
    """
    Assigns a sentiment label based on polarity.

    Parameters:
    -----------
    polarity : float
        The polarity score (-1 to 1).

    threshold : float, optional, default=0.1
        The threshold for determining the label: 'positive', 'negative', or 'neutral'.

    Returns:
    --------
    str
        A sentiment label ('positive', 'negative', or 'neutral').
    """
    if polarity > threshold:
        return "positive"
    elif polarity < -threshold:
        return "negative"
    return "neutral"


def get_sentiment_features(text: str, threshold: float = 0.1) -> Tuple[float, float, str]:
    """
    Analyzes sentiment of the given text using TextBlob.

    Parameters:
    -----------
    text : str
        The text to analyze (e.g., seller note).

    threshold : float, optional, default=0.1
        The threshold for determining sentiment labels.

    Returns:
    --------
    Tuple[float, float, str]
        - polarity (float): The polarity score (-1 to 1).
        - subjectivity (float): The subjectivity score (0 to 1).
        - sentiment_label (str): 'positive', 'negative', or 'neutral'.
    """
    blob = TextBlob(text)

    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    sentiment_label = get_sentiment_label_by_polarity(polarity, threshold)

    return polarity, subjectivity, sentiment_label


# In[8]:


def get_polarity(text: str, threshold: float = 0.1) -> float:
    """
    Extracts the polarity score from the sentiment of the given text.

    Parameters:
    -----------
    text : str
        The text to analyze for sentiment polarity.

    threshold : float, optional, default=0.1
        The threshold for determining sentiment polarity, passed to
        the `get_sentiment_features` function.

    Returns:
    --------
    float
        The polarity score, ranging from -1 (negative) to 1 (positive).
    """
    polarity, _, _ = get_sentiment_features(text, threshold)
    return polarity


def get_subjectivity(text: str, threshold: float = 0.1) -> float:
    """
    Extracts the subjectivity score from the sentiment of the given text.

    Parameters:
    -----------
    text : str
        The text to analyze for sentiment subjectivity.

    threshold : float, optional, default=0.1
        The threshold for determining sentiment polarity, passed to
        the `get_sentiment_features` function.

    Returns:
    --------
    float
        The subjectivity score, ranging from 0 (objective) to 1 (subjective).
    """
    _, subjectivity, _ = get_sentiment_features(text, threshold)
    return subjectivity


def get_sentiment_label(text: str, threshold: float = 0.1) -> str:
    """
    Extracts the sentiment label ('positive', 'negative', or 'neutral')
    based on the polarity of the given text.

    Parameters:
    -----------
    text : str
        The text to analyze for sentiment and label.

    threshold : float, optional, default=0.1
        The threshold for determining sentiment polarity, passed to
        the `get_sentiment_features` function.

    Returns:
    --------
    str
        The sentiment label: 'positive', 'negative', or 'neutral'.
    """
    _, _, sentiment_label = get_sentiment_features(text, threshold)
    return sentiment_label


# In[9]:


df = df.with_columns(
    # Polarity
    pl.col('seller_note')
    .map_elements(function=get_polarity, return_dtype=pl.Float64)
    .alias('seller_note_polarity'),
    # Subjectivity
    pl.col('seller_note')
    .map_elements(function=get_subjectivity, return_dtype=pl.Float64)
    .alias('seller_note_subjectivity'),
    # Sentiment Label
    pl.col('seller_note')
    .map_elements(function=get_sentiment_label, return_dtype=pl.Utf8)
    .alias('seller_note_sentiment_label'),
)


# Visualizing the new features for the seller notes:

# In[10]:


df.select(
    'seller_note', 'seller_note_polarity', 'seller_note_subjectivity', 'seller_note_sentiment_label'
).filter(
    pl.col('seller_note').is_not_null()
).with_columns(
    pl.col('seller_note_polarity').round(2).alias('seller_note_polarity'),
    pl.col('seller_note_subjectivity').round(2).alias('seller_note_subjectivity'),
)


# #### **Visualization** - Seller Note Polarity

# In[11]:


(
    so.Plot(df, x='seller_note_polarity')
    .add(so.Bars(color='skyblue', alpha=.8), so.Hist('density'))
    .add(so.Area(color='red', alpha=.15), so.KDE())
    .label(title='Seller Note Polarity KDE', x='Polarity', y='Density')
)


# In[12]:


(
    so.Plot(df, x='seller_note_polarity', color='seller_note_sentiment_label')
    .add(so.Bars(alpha=.8), so.Hist('probability'))
    .label(title='Seller Note Polarity by Sentiment', x='Polarity', y='Probability', color='Sentiment')
)


# #### **Visualization** - Seller Note Subjectivity

# In[13]:


(
    so.Plot(df, x='seller_note_subjectivity')
    .add(so.Bars(color='purple', alpha=.8), so.Hist('density'))
    .add(so.Area(color='yellow', alpha=.3), so.KDE())
    .label(title='Seller Note Subjectivity KDE', x='Subjectivity', y='Density')
)


# In[14]:


(
    so.Plot(df, x='seller_note_subjectivity', color='seller_note_sentiment_label')
    .add(so.Bars(alpha=.8), so.Hist('probability'))
    .label(title='Seller Note Subjectivity KDE', x='Subjectivity', y='Density')
)


# #### **Visualization** - Seller Note Sentiment

# In[15]:


(
    so.Plot(df, x='seller_note_sentiment_label', color='seller_note_sentiment_label')
    .add(so.Bars(), so.Hist(), legend=False)
    .label(title='Seller Notes by Sentiment', x='sentiment', y='count')
)


# In[16]:


sns.catplot(data=df, x='seller_note_sentiment_label', kind='box', y='min_price', hue='seller_note_sentiment_label')

(
    so.Plot(df, x="min_price", color='seller_note_sentiment_label')
    .facet("seller_note_sentiment_label")
    .add(so.Area(), so.KDE())
    .label(x='minimum price', y='probability density', color='sentiment')
)


# In[17]:


(
    so.Plot(df, x='seller_note_polarity', y='seller_note_subjectivity', color='seller_note_sentiment_label')
    .add(so.Dots())
    .limit(x=(-1, 1), y=(0, 1))
    .label(title='Seller Note Polarity vs Subjectivity by Sentiment', x='Polarity', y='Subjectivity', color='sentiment')
)


# In[18]:


(
    so.Plot(df, x="seller_note_polarity", y='seller_note_subjectivity', color='seller_note_sentiment_label')
    .facet("seller_note_sentiment_label", order=['negative', 'neutral', 'positive'])
    .add(so.Dots())
    .limit(x=(-1, 1))
    .label(x='polarity', y='subjectivity', color='sentiment')
)


# #### Standardizing hard drive, RAM & SSD size to their size in gigabytes

# In[19]:


df.select([
    pl.col(col).unique().alias(col)
    for col in df.columns if col.endswith('_unit') and not col.startswith('processor')
])


# In[20]:


def convert_to_gb(df_to_convert: pl.DataFrame, size_column: str, unit_column: str):
    """
    Convert size values to gigabytes based on the unit column.

    Args:
        df_to_convert (pl.DataFrame): DataFrame with size and unit columns.
        size_column (str): Column containing size values.
        unit_column (str): Column containing units (e.g., 'gigabytes', 'megabytes', 'terabytes').

    Returns:
        pl.DataFrame: DataFrame with a new column `<size_column>_in_gb` containing size in gigabytes.
    """
    return df_to_convert.with_columns(
        pl.when(df_to_convert[unit_column] == "gigabytes").then(df_to_convert[size_column])
        .when(df_to_convert[unit_column] == "megabytes").then(df_to_convert[size_column] / 1024)
        .when(df_to_convert[unit_column] == "terabytes").then(df_to_convert[size_column] * 1024)
        .otherwise(None).alias(f"{size_column}_in_gb")
    )


# In[21]:


size_columns = [col for col in df.columns if col.endswith('_size')]

for size_col in size_columns:
    unit_col = f"{size_col}_unit"
    df = convert_to_gb(df, size_col, unit_col)


# In[22]:


for size in size_columns:
    print(f'\nNon-null values for {size}: {df[size].drop_nulls().len()}')
    size_in_gb = f'{size}_in_gb'
    print(f'Non-null values for {size_in_gb}: {df[size_in_gb].drop_nulls().len()}')


# ### Feature Selection

# In[23]:


# The target variable : the minimum pricer required to purchase the item (laptop/netbook)
target_var = 'min_price'
# A polars expression for feature selection
feature_selection_expr = pl.all().exclude(target_var, 'currency', 'condition_description', 'seller_note',
                                          'seller_note_sentiment_label', 'hard_drive_size', 'hard_drive_size_unit',
                                          'ram_size', 'ram_size_unit', 'ssd_size', 'ssd_size_unit')

# Dataframe holding all feature variables
df_features = df.select(
    feature_selection_expr
)
# Series holding the target vector
series_target = df[target_var]


# ### Categorical Features

# In[24]:


feature_names = df_features.columns
cat_feature_names = df_features.select(
    pl.col(pl.String),
).columns

print(f'{feature_names=}')


# ### Encoding rare labels
# 
# In this project, we leverage the [**Feature-engine**][Feature-Engine] package to handle categorical variables efficiently, specifically using `RareLabelEncoder` to group infrequent categories under a common label.
# 
# Rare categories in a dataset can introduce **high cardinality**, making statistical analysis and model generalization more challenging. By setting a **minimum frequency threshold**, we ensure that only sufficiently common categories remain distinct, while rare ones are grouped into an "other" category.
# 
# This reduces noise, prevents overfitting, and enhances model interpretability without significantly losing information.
# 
# <div align="center">
# <img src="../assets/logos/feature_engine.png" height="150" width="150"/>
# <img src="../assets/images/rare_label_encoding.png" height="300" width="500"/>
# </div>
# 
# [Feature-Engine]: https://github.com/feature-engine/feature_engine

# In[25]:


# Sets the minimum count for a category to be kept separately, categories with fewer than 20 occurrences will be grouped.
MIN_COUNT_FOR_LABEL = 20
# Sets the tolerance for rare categories based on the minimum count and total number of rows in the dataset.
TOLERANCE_FOR_LABEL = MIN_COUNT_FOR_LABEL / df_features.height

encoder = RareLabelEncoder(n_categories=1, replace_with='other', tol=TOLERANCE_FOR_LABEL)
for col in cat_feature_names:
    encoder_transform = encoder.fit_transform(df_features[[col]].fill_null('unknown').to_pandas())
    df_features = df_features.with_columns(
        pl.Series(encoder_transform[col])
    )


# ## Machine Learning

# In[26]:


# TODO : Machine Learning


# ### Splitting the data - Train-Validation-Test Split
# 
# <div align="center">
# <img src="../assets/images/train_test_validation_split.png" height="326" width="500"/>
# </div>

# ### Categorical Boosting
# 
# **CatBoost** is a gradient boosting algorithm that builds decision trees sequentially, improving predictions at each step (*Ensemble learning*).
# 
# A key feature is how CatBoost handles **categorical** data (hence its name). Instead of one-hot encoding or label encoding, it uses **target encoding**, where categories are replaced by smoothed target statistics, essentially *converting categorical values into numerical values*.
# 
# Additionally, trees CatBoost builds are **oblivious trees**, meaning that every split at a given depth applies the same condition across all nodes. This structure helps prevent overfitting, improves generalization, and makes computation highly efficient, especially on GPUs.
# 
# Overall, CatBoost refines the traditional gradient boosting approach by handling categorical data naturally, and optimizing computation, making it one of the best choices for tabular data problems with a considerable amount of categorical variables.
# 
# <div align="center">
# <img src="../assets/images/categorical_boosting.png" height="375" width="375"/>
# <br>
# <a href="https://www.researchgate.net/figure/The-flow-diagram-of-the-CatBoost-model_fig3_370695897"><i>The flow diagram of the CatBoost model</i></a>
# </div>

# ## Hyperparameter Tuning

# In[27]:


# TODO : Hyperparameter Tuning


# ## Explainable AI - SHAP

# In[28]:


# TODO: Explainable AI - SHAP


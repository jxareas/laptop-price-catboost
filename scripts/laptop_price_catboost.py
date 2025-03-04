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

# In[23]:


# Importing libraries and setting constants

# Libraries
import polars as pl
import seaborn.objects as sns
from feature_engine.encoding import RareLabelEncoder
from textblob import TextBlob
from typing import Tuple

# Constants
RANDOM_SEED = 287
DATA_SOURCE_PATH = '../data/ebay_laptops_and_netbooks_cleansed.csv'


# ## Data Preparation

# ### Loading the dataset

# In[ ]:


df = pl.read_csv(DATA_SOURCE_PATH)
df.head(n=10)


# ## Exploratory Data Analysis

# In[ ]:


# TODO : Exploratory Data Analysis


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

# In[ ]:


df.select(
    'seller_note'
).filter(
    pl.col('seller_note').is_not_null()
).head(n=10).to_pandas()


# In[ ]:


def get_sentiment_label_by_polarity(polarity: float, threshold: float = 0.1) -> str:
    """
    Determines the sentiment label based on polarity.

    This function assigns a sentiment label ('positive', 'negative', or 'neutral')
    based on the polarity value and a given threshold.

    Parameters:
    -----------
    polarity : float
        The polarity score of the text, ranging from -1 (negative) to 1 (positive).

    threshold : float, optional, default=0.1
        The threshold for determining sentiment labels. If the polarity is greater
        than the threshold, the label will be 'positive'; if less than the negative
        threshold, the label will be 'negative'; otherwise, the label will be 'neutral'.

    Returns:
    --------
    str
        A sentiment label ('positive', 'negative', or 'neutral') based on the polarity score.
    """
    if polarity > threshold:
        return "positive"
    elif polarity < -threshold:
        return "negative"
    else:
        return "neutral"


def get_sentiment_features(text: str, threshold: float = 0.1) -> Tuple[float, float, str]:
    """
    Analyzes the sentiment of a given text using TextBlob.

    This function calculates the polarity and subjectivity of the input text
    and assigns a sentiment label using the `get_sentiment_label` function.

    Parameters:
    -----------
    text : str
        The input text (e.g., seller note) to analyze for sentiment.

    threshold : float, optional, default=0.1
        The threshold for determining sentiment labels. If the polarity is greater
        than the threshold, the label will be 'positive'; if less than the negative
        threshold, the label will be 'negative'; otherwise, the label will be 'neutral'.

    Returns:
    --------
    Tuple[float, float, str]
        A tuple containing:
        - polarity (float): The polarity score of the text, ranging from -1 (negative)
          to 1 (positive).
        - subjectivity (float): The subjectivity score of the text, ranging from 0 (objective)
          to 1 (subjective).
        - sentiment_label (str): A sentiment label ('positive', 'negative', or 'neutral')
          based on the polarity score.
    """
    blob = TextBlob(text)

    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Get the sentiment label based on polarity
    sentiment_label = get_sentiment_label_by_polarity(polarity, threshold)

    return polarity, subjectivity, sentiment_label


# In[ ]:


def get_polarity(text: str, threshold: float = 0.1) -> float:
    """
    Extracts the polarity from the sentiment of the given text.

    Parameters:
    -----------
    text : str
        The input text to analyze for sentiment polarity.

    threshold : float, optional, default=0.1
        The threshold for determining sentiment polarity. This value is passed
        to the `get_sentiment_features` function, but is not used in this
        function directly.

    Returns:
    --------
    float
        The polarity score of the text, ranging from -1 (negative) to 1 (positive).
    """
    polarity, _, _ = get_sentiment_features(text, threshold)
    return polarity


def get_subjectivity(text: str, threshold: float = 0.1) -> float:
    """
    Extracts the subjectivity from the sentiment of the given text.

    Parameters:
    -----------
    text : str
        The input text to analyze for sentiment subjectivity.

    threshold : float, optional, default=0.1
        The threshold for determining sentiment polarity. This value is passed
        to the `get_sentiment_features` function, but is not used in this
        function directly.

    Returns:
    --------
    float
        The subjectivity score of the text, ranging from 0 (objective) to 1 (subjective).
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
        The input text to analyze for sentiment and label.

    threshold : float, optional, default=0.1
        The threshold for determining sentiment polarity. This value is passed
        to the `get_sentiment_features` function, but is not used in this
        function directly.

    Returns:
    --------
    str
        The sentiment label of the text: 'positive', 'negative', or 'neutral'.
    """
    _, _, sentiment_label = get_sentiment_features(text, threshold)
    return sentiment_label


# In[ ]:


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

# In[ ]:


df.select(
    'seller_note', 'seller_note_polarity', 'seller_note_subjectivity', 'seller_note_sentiment_label'
).filter(
    pl.col('seller_note').is_not_null()
).with_columns(
    pl.col('seller_note_polarity').round(2).alias('seller_note_polarity'),
    pl.col('seller_note_subjectivity').round(2).alias('seller_note_subjectivity'),
)


# #### **Visualization** - Seller Note Polarity

# In[ ]:


## TODO : Add Seller Note Polarity Plots


# #### **Visualization** - Seller Note Subjectivity

# In[ ]:


## TODO : Add Seller Note Subjectivity Plots


# #### **Visualization** - Seller Note Sentiment

# In[ ]:


## TODO : Add Seller Note Sentiment Plots


# #### Standardizing hard drive, RAM & SSD size to their size in gigabytes

# In[ ]:


df.select([
    pl.col(col).unique().alias(col)
    for col in df.columns if col.endswith('_unit') and not col.startswith('processor')
])


# In[ ]:


def convert_to_gb(df_to_convert: pl.DataFrame, size_column: str, unit_column: str):
    return df_to_convert.with_columns(
        pl.when(df_to_convert[unit_column] == "gigabytes").then(df_to_convert[size_column])
        .when(df_to_convert[unit_column] == "megabytes").then(df_to_convert[size_column] / 1024)
        .when(df_to_convert[unit_column] == "terabytes").then(df_to_convert[size_column] * 1024)
        .otherwise(None).alias(f"{size_column}_in_gb")
    )


# In[ ]:


size_columns = [col for col in df.columns if col.endswith('_size')]

for size_col in size_columns:
    unit_col = f"{size_col}_unit"
    df = convert_to_gb(df, size_col, unit_col)


# In[ ]:


print(f'Hard Drive Size - Non-null values: {df['hard_drive_size'].drop_nulls().len()}')
print(f'RAM Size - Non-null values: {df['ram_size'].drop_nulls().len()}')
print(f'SSD Size - Non-null values: {df['ssd_size'].drop_nulls().len()}')

print(f'Hard Drive Size in Gigabytes - Non-null values: {df['hard_drive_size_in_gb'].drop_nulls().len()}')
print(f'RAM Size in Gigabytes - Non-null values: {df['ram_size_in_gb'].drop_nulls().len()}')
print(f'SSD Size in Gigabytes - Non-null values: {df['ssd_size_in_gb'].drop_nulls().len()}')


# ### Feature Selection

# In[ ]:


# The target variable : the minimum pricer required to purchase the item (laptop/netbook)
target_var = 'min_price'
# A polars expression for feature selection -> selecting every column except for the target variable & currency (currency cardinality is one)
feature_selection_expr = pl.all().exclude(target_var, 'currency', 'seller_note')

# Dataframe holding all feature variables
df_features = df.select(
    feature_selection_expr
)
# Series holding the target vector
series_target = df[target_var]


# ### Categorical Features

# In[ ]:


feature_names = df_features.columns
cat_feature_names = df_features.select(
    pl.col(pl.String),
).columns


# ### Encoding rare labels
# 
# In this project, we leverage the [**Feature-engine**][Feature-Engine] package to handle categorical variables efficiently, specifically using `RareLabelEncoder` to group infrequent categories under a common label.
# 
# Rare categories in a dataset can introduce **high cardinality**, making statistical analysis and model generalization more challenging. By setting a **minimum frequency threshold**, we ensure that only sufficiently common categories remain distinct, while rare ones are grouped into an "other" category.
# 
# This reduces noise, prevents overfitting, and enhances model interpretability without significantly losing information.
# 
# <div align="center">
# <img src="../assets/logos/feature_engine.png" height="125" width="125"/>
# </div>
# 
# [Feature-Engine]: https://github.com/feature-engine/feature_engine

# In[ ]:


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

# In[ ]:


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

# In[ ]:


# TODO : Hyperparameter Tuning


# ## Explainable AI - SHAP

# In[ ]:


# TODO: Explainable AI - SHAP


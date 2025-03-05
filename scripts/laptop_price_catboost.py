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
# [Yandex CatBoost]: https://yandex.com/dev/catboost/

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

# In[ ]:


# Importing libraries and setting constants

# Libraries
import polars as pl
import seaborn as sns
import seaborn.objects as so
import plotly
import catboost as cb
import shap
import optuna
import optuna.visualization as vis
import matplotlib.style as style
from feature_engine.encoding import RareLabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from textblob import TextBlob
import os
from typing import Tuple

# Constants
RANDOM_SEED = 287  # Random Seed for reproducibility
N_CORES = os.cpu_count() / 2  # Half the cores
DATA_SOURCE_PATH = '../data/ebay_laptops_and_netbooks_cleansed.csv'  # Path where the data is stored


# ## Data Preparation

# ### Loading the dataset

# In[ ]:


df = pl.read_csv(DATA_SOURCE_PATH)
df.describe()


# ## Exploratory Data Analysis
# 

# In[ ]:


# TODO : Exploratory Data Analysis


# In[ ]:


(
    so.Plot(df, x='min_price')
    .add(so.Bars(color='royalblue'), so.Hist('density'))
    .add(so.Area(color='red', alpha=.15), so.KDE())
    .label(title='Minimum Price KDE')
)


# In[ ]:


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

# In[ ]:


df.select(
    'seller_note'
).filter(
    pl.col('seller_note').is_not_null()
).head(n=10).to_pandas()


# In[ ]:


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


# In[ ]:


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


(
    so.Plot(df, x='seller_note_polarity')
    .add(so.Bars(color='skyblue', alpha=.8), so.Hist('density'))
    .add(so.Area(color='red', alpha=.15), so.KDE())
    .label(title='Seller Note Polarity KDE', x='Polarity', y='Density')
)


# In[ ]:


(
    so.Plot(df, x='seller_note_polarity', color='seller_note_sentiment_label')
    .add(so.Bars(alpha=.8), so.Hist('probability'))
    .label(title='Seller Note Polarity by Sentiment', x='Polarity', y='Probability', color='Sentiment')
)


# #### **Visualization** - Seller Note Subjectivity

# In[ ]:


(
    so.Plot(df, x='seller_note_subjectivity')
    .add(so.Bars(color='purple', alpha=.8), so.Hist('density'))
    .add(so.Area(color='yellow', alpha=.3), so.KDE())
    .label(title='Seller Note Subjectivity KDE', x='Subjectivity', y='Density')
)


# In[ ]:


(
    so.Plot(df, x='seller_note_subjectivity', color='seller_note_sentiment_label')
    .add(so.Bars(alpha=.8), so.Hist('probability'))
    .label(title='Seller Note Subjectivity KDE', x='Subjectivity', y='Density')
)


# #### **Visualization** - Seller Note Sentiment

# In[ ]:


(
    so.Plot(df, x='seller_note_sentiment_label', color='seller_note_sentiment_label')
    .add(so.Bars(), so.Hist(), legend=False)
    .label(title='Seller Notes by Sentiment', x='sentiment', y='count')
)


# In[ ]:


sns.catplot(data=df, x='seller_note_sentiment_label', kind='box', y='min_price', hue='seller_note_sentiment_label')

(
    so.Plot(df, x="min_price", color='seller_note_sentiment_label')
    .facet("seller_note_sentiment_label")
    .add(so.Area(), so.KDE())
    .label(x='minimum price', y='probability density', color='sentiment')
)


# In[ ]:


(
    so.Plot(df, x='seller_note_polarity', y='seller_note_subjectivity', color='seller_note_sentiment_label')
    .add(so.Dots())
    .limit(x=(-1, 1), y=(0, 1))
    .label(title='Seller Note Polarity vs Subjectivity by Sentiment', x='Polarity', y='Subjectivity', color='sentiment')
)


# In[ ]:


(
    so.Plot(df, x="seller_note_polarity", y='seller_note_subjectivity', color='seller_note_sentiment_label')
    .facet("seller_note_sentiment_label", order=['negative', 'neutral', 'positive'])
    .add(so.Dots())
    .limit(x=(-1, 1))
    .label(x='polarity', y='subjectivity', color='sentiment')
)


# #### Standardizing hard drive, RAM & SSD size to their size in gigabytes

# In[ ]:


df.select([
    pl.col(col).unique().alias(col)
    for col in df.columns if col.endswith('_unit') and not col.startswith('processor')
])


# In[ ]:


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


# In[ ]:


size_columns = [col for col in df.columns if col.endswith('_size')]

for size_col in size_columns:
    unit_col = f"{size_col}_unit"
    df = convert_to_gb(df, size_col, unit_col)


# In[ ]:


for size in size_columns:
    print(f'Non-null values for {size}: {df[size].drop_nulls().len()}')
    size_in_gb = f'{size}_in_gb'
    print(f'Non-null values for {size_in_gb}: {df[size_in_gb].drop_nulls().len()}\n')


# ### Feature Selection

# In[ ]:


# The target variable : the minimum pricer required to purchase the item (laptop/netbook)
target_var = 'min_price'
# A polars expression for feature selection
feature_selection_expr = pl.all().exclude(target_var, 'currency', 'condition_description', 'seller_note',
                                          'seller_note_polarity' 'hard_drive_size', 'hard_drive_size_unit',
                                          'ram_size', 'ram_size_unit', 'ssd_size', 'ssd_size_unit')

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

print(f'All Features: \n{feature_names}')
print(f'Categorical Features: \n{cat_feature_names}')


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

# In[ ]:


stratify_col = 'brand'

df_train, df_val_and_test, series_train, series_val_and_test = train_test_split(
    df_features,
    series_target,
    stratify=df_features[stratify_col],
    test_size=.3,
    random_state=RANDOM_SEED,
)

df_val, df_test, series_val, series_test = train_test_split(
    df_val_and_test,
    series_val_and_test,
    test_size=.5,
    stratify=df_val_and_test[stratify_col],
    random_state=RANDOM_SEED, )


# In[ ]:


X_train = df_train.to_numpy()
y_train = series_train.to_numpy()
X_val = df_val.to_numpy()
y_val = series_val.to_numpy()
X_test = df_test.to_numpy()
y_test = series_test.to_numpy()


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

# In[ ]:


train_pool = cb.Pool(data=X_train, label=y_train, cat_features=cat_feature_names, feature_names=feature_names)
val_pool = cb.Pool(data=X_val, label=y_val, cat_features=cat_feature_names, feature_names=feature_names)


# In[ ]:


catboost = cb.CatBoostRegressor(loss_function='RMSE')
catboost.fit(X=train_pool, eval_set=val_pool, logging_level='Silent')


# In[ ]:


catboost_feature_importance = pl.DataFrame({
    'feature': catboost.feature_names_,
    'importance': catboost.feature_importances_,
}).sort(by='importance', descending=True)

(
    so.Plot(catboost_feature_importance, x='importance', y='feature')
    .add(so.Bar(), color='feature')
    .label(title='CatBoost Feature Importance')
    .theme(style.library['fast'])
)


# In[ ]:


# Make predictions using the trained model on both the training and validation data
y_train_pred = catboost.predict(train_pool)
y_val_pred = catboost.predict(val_pool)

# Calculate the Root Mean Squared Error (RMSE) scores for training and validation data
rmse_train = root_mean_squared_error(series_train, y_train_pred)
rmse_val = root_mean_squared_error(series_val, y_val_pred)

# Calculate the Mean Absolute Error (MAE) scores for training and validation data
mae_train = mean_absolute_error(series_train, y_train_pred)
mae_val = mean_absolute_error(series_val, y_val_pred)

# Print the rounded RMSE scores
print(f"RMSE score for train {round(rmse_train)} USD & for validation {round(rmse_val)} USD")
print(f"MAE score for train {round(mae_train)} USD & for validation {round(mae_val)} USD")


# ## Hyperparameter Tuning
# 
# <div align="center">
# <img src="../assets/logos/optuna.png" height="375" width="375"/>
# </div>

# In[ ]:


# TODO : Hyperparameter Tuning


# In[ ]:


def objective(trial):
    # Define hyperparameter search space for optimization
    params = {
        'iterations': trial.suggest_int('iterations', 400, 500),  # Number of iterations
        'depth': trial.suggest_int('depth', 4, 6),  # Max depth of trees
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),  # Learning rate
        'loss_function': 'RMSE',  # Loss function to optimize
        'verbose': 0,  # Set verbosity level to 0
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 20, 100),  # Early stopping
    }

    # Train CatBoostRegressor with suggested parameters
    model = cb.CatBoostRegressor(**params, thread_count=N_CORES)
    model.fit(train_pool, eval_set=val_pool)

    # Make predictions and calculate RMSE on the validation set
    pred = model.predict(val_pool)
    trial_rmse = root_mean_squared_error(series_val, pred)

    return trial_rmse


# In[ ]:


tpe_sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
hyperband_pruner = optuna.pruners.HyperbandPruner()

study = optuna.create_study(study_name='catboost-hyperopt', direction='minimize')
study.optimize(objective, n_trials=50, n_jobs=N_CORES)


# In[ ]:


print(f'Best Trial: #{study.best_trial.number}')
print(f'Best Trial Value: {study.best_trial.value}')
print(f'Best Trial Params: {study.best_trial.params}')


# In[ ]:


vis.plot_param_importances(study)


# In[ ]:


fig = vis.plot_parallel_coordinate(study)
fig.update_layout(
    title="Hyperparameter Tuning for CatBoost - Parallel Coordinate Plot",
    title_font=dict(family="Arial", size=18, color="black", weight="bold"),  # Bold title
    title_x=0.5,
    font=dict(family="Arial", size=14, color="black")
)

# Access the traces to modify the line colorscale
fig.data[0].line.colorscale = plotly.colors.sequential.Reds
fig


# ## Explainable AI - SHAP

# ### Understanding Model Decisions

# - **SHAP (SHapley Additive exPlanations)** is a method used to explain machine learning models by showing how much each input factor (feature) contributes to a prediction.
# - It helps us see which features are **driving the model’s decisions**—whether they increase or decrease the predicted outcome.
# - The **summary plot** provides a visual representation of feature importance, showing:
#   - Which features **have the most impact** overall.
#   - Whether a feature **positively or negatively influences** predictions.
#   - How the values of these features interact with the model.
# 
# <div align="center">
# <img src="../assets/logos/shap.png" height="375" width="375"/>
# </div>

# In[ ]:


# Run SHAP
shap.initjs()
tree_explainer = shap.TreeExplainer(catboost)
shap_values = tree_explainer.shap_values(X_val)

# Pass feature names manually
shap.summary_plot(shap_values, X_val, feature_names=feature_names)


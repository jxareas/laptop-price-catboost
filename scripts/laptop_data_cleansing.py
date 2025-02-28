#!/usr/bin/env python
# coding: utf-8

# # eBay Laptops & Notebooks - Data Cleansing
# 
# The following dataset contains information about laptops and notebooks, by web scraping the popular e-commerce website
# **eBay**.
# 
# The dataset contains information about the laptop prices (possible variable of interest), brand, ratings, condition, as well as hardware information (processor, screen size, ram, etc).
# 
# <div style="text-align: center;">
# <img src="../assets/logos/ebay.png" width="250"/>
# </div>

# ## Loading Libraries
# 
# We are using **Polars** for data cleansing. Polars is a **fast**, ***parallel**, and **memory-efficient** DataFrame library written in *Rust*, designed for handling large datasets.
# 
# It provides intuitive API operations for data manipulation, such as filtering, grouping, joining, and transforming data, all in a very efficient manner. Unlike other popular libraries like Pandas, Polars is optimized for performance and can handle larger datasets with significantly faster execution times.
# During the data cleaning process, we will leverage Polars to:
# - Load and transform the data
# - Clean and map columns based on custom logic
# - Perform various aggregations and groupings
# - Handle null values and ensure consistency across the dataset
# 
# Polars supports both eager and lazy execution modes, making it flexible for a variety of use cases.
# 
# <div style="text-align: center;">
# <img src="../assets/logos/polars.png"/>
# </div>

# In[1]:


# Importing libraries and setting constants

import re
from typing import Optional

import polars as pl
from polars import LazyFrame

# Data source location
DATA_SOURCE_PATH = '../data/ebay_laptops_and_notebooks.csv'


# ## Data Exploration

# In[2]:


# Loading the dataset
df = pl.read_csv(DATA_SOURCE_PATH)

# Top 10 rows from the dataframe
df.head(n=10)


# Observations:
# 
# * `Price` variable contains information about the price itself, but also the currency. Also, some prices are in a range format, such as `$399.99 to $634.99` which needs to be handled before converting `Price` into a numerical variable.
# * **Several** columns are being treated as `String` (such as `Price`), which might not be adequate.
# * **Several** columns have **missing values**, which needs to be treated during further data preprocessing.

# ### Summary Statistics
# 
# We compute the summary statistics for our dataset, but we instantly notice most of our columns are nullable, which makes statistics, such as the mean and standard deviation, to output `null` too.

# In[3]:


# Summary statistics
df.describe()


# ### Null Count and Null Proportion
# 
# Now, we take a look at the `null count` and `null proportion` per each dataframe column:

# In[4]:


# Nulls per each column in the dataset
df.null_count().unpivot(
    variable_name='variable',
    value_name='null_count',
).sort(
    by='null_count', descending=True,
).with_columns(
    (pl.col('null_count') / df.height).round(2).alias('null_proportion'),
)


# Observations:
# 
# - Extremely high percentage (85%+) of nulls for columns such as `country_region_of_manufacturer`, `ratings_count` or `release_year`.
# - `Price` column has no null values.

# ### Data Types
# 
# We look at each column data type in the data frame

# In[5]:


# Data Types
pl.DataFrame(data={
    'column': df.columns,
    'dtype': df.dtypes
})


# As aforementioned, some variables which are currently being treated as `String`, such as `Price`, are actually numerical. It's very important to find out whether this is a single column, or there are more occurrences within the dataframe.

# ## Data Preparation

# ## Column Renaming
# 
# We kick things off by *reformatting* the column names: lowercasing and removing spaces for underscores. These convention for string data will be **STANDARD** during our data cleansing process, so that most categorical data is transformed into similar formatting.

# In[6]:


# Renaming columns: replacing blank spaces for underscores and lowercasing columns
df = df.rename({col: col.strip().lower().replace(' ', '_') for col in df.columns})
print(df.columns)


# ## Cleaning `brand`
# - **Top categories**: The top three brands—Dell (19%), Lenovo (11%), and HP (7%)—represent a combined 37% of the total brands in the dataset. The null values account for 39% of the data, with 2596 entries marked as null. This is quite a large portion of the dataset. The top 4 categories (`null`, `Dell`, `Lenovo` & `HP`) account for 76% of the dataset.
# - **Unknown values**: There's an abundance of brands with very low frequencies, or unknown values (like `?` or `Does not apply`).
# - **Inconsistent formatting**: Some entries like `Dell Inc` are not captured under the main brand `Dell` due to inconsistent formatting. Similar things occur to other top brands.
# - **Multibrand categories**: Some entries represent several brands, like `Dell / HP / Lenovo` or `Apple / LG`

# In[7]:


# Taking a look at the frequency of each brand: total count and proportion
df['brand'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# ### **Reformatting `brand`:**
# 
# In this step of our analysis, we're focusing on cleaning up and standardizing the `brand` data to ensure consistency and accuracy. Brands can be listed in many different ways, which can lead to confusion or misinterpretation during analysis. So, we're transforming the brand names into a uniform format that makes it easier to work with.
# 
# - **Consistent Formatting:** We ensure all brand names are in lowercase, and we remove any unnecessary spaces or punctuation that could create inconsistencies. (transforming `Dell Inc` into `dell_inc`).
# - **Replacing Uncertain Values:** We also handle special cases where the data might have uncertain or irrelevant values (e.g., a "?" for unknown brands), replacing them with more meaningful terms like "unbranded" or "unknown."
# 

# In[8]:


# Define a mapping to replace certain placeholder values with more meaningful labels.
brand_replace_map = {
    '?': 'unbranded',  # Replace '?' with 'unbranded' for unknown brands.
    'does_not_apply': 'not_applicable',  # Replace 'does_not_apply' with 'not_applicable' to handle irrelevant data.
}

# Clean and normalize the brand names.
df = df.with_columns(
    pl.col('brand')
    .str.to_lowercase()  # Convert brand names to lowercase to ensure uniformity.
    .str.strip_chars()  # Remove any leading whitespace or special characters.
    .str.strip_chars_end('.')  # Specifically remove any trailing dots at the end of brand names.
    .str.replace_all(' / |\\/', ' or ')  # Replace slashes ("/" or " / ") with ' or ' to standardize multi-brand names.
    .str.replace_all(' ', '_')  # Replace spaces with underscores to ensure no spaces in brand names.
    .replace(brand_replace_map)  # Replace values like '?' or 'does_not_apply' based on the predefined map.
    .fill_null('unknown')  # Fill any missing brand values with 'unknown' to handle missing data.
    .alias('brand_clean')
)


# ### **Identifying and Mapping Top Brands:**
# 
# In this step, we're focusing on identifying the most popular brands in the dataset and ensuring that any variations in the brand names are mapped to the correct top brand.
# 
# - **Identifying Top Brands:** We start by filtering out any unknown brand names and then count the occurrences of each brand. From this, we identify the top 5 brands with the highest frequency in the dataset.
# - **Pattern Matching:** We create a pattern to search for these top brands in the brand names. If the brand name matches one of the top brands (e.g., includes "dell", "hp", etc.), we recognize it as a valid entry for that brand.
# - **Mapping Variations to Top Brands:** We then apply a function to map variations of the brand names (e.g., "dell_epsilon" or "hp_dell") to the main brand names (e.g., "dell" or "hp"). This ensures that similar or ambiguous entries are standardized to the correct top brand.
# 
# By applying these steps, we ensure that any variations or inconsistencies in the brand names are correctly mapped to the top brands, improving the consistency and accuracy of our analysis.
# 

# In[9]:


# Identify the top 5 most frequent brands in the dataset.
top_brands = (
    df.filter(pl.col('brand_clean').ne('unknown'))
    .group_by('brand_clean')
    .agg(pl.len().alias('count'))
    .sort('count', descending=True)
    .head(5)
    .select('brand_clean')
    .to_series()
)

# Create a regex pattern to match brand names followed by an underscore (_).
is_a_top_brand_string_pattern = "|".join(re.escape(color) for color in top_brands + '_')


# Function to extract the actual brand from a string.
def fetch_first_matching_brand_from_str(string: str, brand_list: list[str] = top_brands) -> str:
    """
    Extracts the first matching brand from a given string.

    Args:
        string (str): The input string containing the brand.
        brand_list (list): List of top brands to check against.

    Returns:
        str: The first matched brand if found; otherwise, the original string.
    """
    for brand in brand_list:
        if brand + '_' in string:
            return brand
    return string


# Apply the brand extraction function to standardize brand names.
df = df.with_columns(
    pl
    # Check if the brand contains any of the top brands and isn't a mixed brand like "dell_or_hp".
    .when(
        pl.col('brand_clean').str.contains(is_a_top_brand_string_pattern) &
        ~pl.col('brand_clean').str.contains('_or_')
    )
    # If the condition is met, map the brand name to its standardized version.
    .then(pl.col('brand_clean').map_elements(fetch_first_matching_brand_from_str, return_dtype=pl.Utf8))
    # Otherwise, keep the original brand name.
    .otherwise(pl.col('brand_clean'))
    # Rename the transformed column as 'brand_clean'.
    .alias('brand_clean')
)


# In[10]:


# Taking another look at the frequency of each brand: total count and proportion
df['brand_clean'].value_counts().with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
).sort(by='count', descending=True)


# Observations:
# - **More records assigned to `dell`**: `dell` has more than 20 new records. (from `1273` to `1300`). Likely due to previously fragmented brand variations (e.g., `dell_inc`, `dell_xps`) now being mapped correctly to `dell`.
# - **Consistent top brands**: The overall ranking of top brands did not change. The counts for `acer` (150), `lg` (125), `fujitsu_siemens` (123), `samsung` (122), `microsoft` (119), and AUO (114) remained stable, indicating that these brands were either already well-represented or had fewer naming inconsistencies.

# ## Cleaning `price`
# 
# - **Multiple info**: By analyzing the `price` variable, we realize it also has information about the currency (via symbols like `$`).
# - **Inconsistent format**: The cost of a laptop might be fixed (e.g: `880`) or a range (e.g: `from 500 to 999`)
# - **Bad typing**: `price` is currently a categorical (`String`) variable, due to commas, whitespaces and symbols. To use this data effectively for statistical models, we **MUST CONVERT** it into **NUMERICAL**, removing currency symbols and handling text-based variations like ranges.

# In[11]:


# Taking a look at the first 5 records from the `price` column, as is
df['price'].head(n=10)


# Observations:
# - **Variable format**: Cost of a laptop might be either a range (`$399.99 to $634.99`) or a fixed price (`$399.99 to $634.99`).
# - **Wrong typing**: Data is currently of `String` data type, instead of a numerical type, due to the presence of currency symbols and words ('to').

# ### **Reformatting `price`:**
# 
# In this step, we focus on cleaning and standardizing the `price` variable to make it usable for further analysis. Currently, the `price` column contains multiple inconsistencies that need to be addressed.
# 
# - **Extracting Currency Symbols:** The dataset contains various currency symbols (such as `$`), which are separated from the numerical values, and assigned to a new variable `currency_clean`.
# - **Handling Inconsistent Formats:** Prices appear in different formats, sometimes as a single fixed value (`880`) and other times as a range (`from 500 to 999`). We need to extract the lowest value from these ranges (`min_price_clean`) to ensure uniformity.
# - **Ensuring Proper Data Types:** After cleaning, we cast the extracted price values to `Float64`, making them compatible with statistical models and numerical computations.
# 

# In[12]:


df = df.with_columns(
    # Extract the currency symbol from the price column
    pl.col('price').str.slice(0, 1).alias('currency_clean'),
    # Removing the currency symbol, commas from the price and trimming whitespace
    pl.col('price')
    .str.slice(1)  # Removing the currency symbol
    .str.replace(',', '')  # Removing commas
    .str.split("to")  # Separating the price into min_price and max_price range
    .list.first()  # Obtaining the min_price, the first element of the list
    .str.strip_chars()  # Remove leading and trailing characters
    .cast(pl.Float64)  # Casting the min_price to float
    .alias('min_price_clean')  # Renaming to reflect that it's the min price in case of ranges
)

# Preview the transformed data by displaying the top 5 rows
df.select(
    'price', 'currency_clean', 'min_price_clean'
).head(n=10)


# ## Cleaning `rating` & `ratings_count`

# In[13]:


# Taking a look at the frequency of each rating: total count and proportion
df['rating'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# Observations:
# - **Extremely high null proportion**: 96% of entries have missing (`null`) ratings, significantly reducing the available data for analysis. Almost the entire column is composed of `null` values.
# - **Strong positive bias**: Ratings are overwhelmingly positive, with **5 out of 5 stars (2%)** and **4.5 out of 5 stars (2%)** making up nearly all non-null values. Only **5 instances** have ratings of **3 stars or lower**, making it difficult to assess dissatisfaction trends.
# - **Possible numerical conversion**: The `rating` column is currently a String but can be converted into an `Int` **five_star_scale** variable for analysis.

# In[14]:


# Taking a look at the frequency of each rating count: total count and proportion
df['ratings_count'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# Observations:
# - **High Null Proportion**: Similarly to `rating`, 96% of entries lack a `ratings_count`, making it almost impossible to gauge customer engagement.
# - **Majority Single Reviews**: Most rated products have **1 to 6 reviews**, suggesting limited user feedback. A few products have significantly higher review counts (e.g., **1,533 reviews**), but these are rare.

# ### Reformatting `rating`
# 
# In this step, we transform the `rating` variable to variable `five_star_scale_rating` ( which, as its name implies, is a numerical variable ranging from 1-5; representing the product rating on a five-star scale) as well as clean its format (replacing spaces / decimal points with underscores) and assign the result to `rating_clean`.

# In[15]:


# Define the regex pattern for extracting numeric values (including decimals)
rating_pattern = r'(\d+(\.\d+)?)'  # Match whole numbers and decimal numbers (e.g., 4, 4.5)

# Transforming the rating variable
df = df.with_columns(
    # Cleaning the rating column
    pl.col('rating')
    .str.replace_all(r' ', '_')  # Replace spaces with underscores
    .str.replace_all(r'\.', '_')  # Replace decimal point with underscore for values like 3.5
    .alias('rating_clean'),
    # Using the rating pattern to match numbers
    pl.when(pl.col('rating').str.contains(rating_pattern))
    # If matched, convert to a numerical variable (e.g., 4, 4.5)
    .then(
        pl.col('rating')
        # Extract the numeric value, including decimals
        .str.extract(rating_pattern)
        # Cast to float to handle half-star ratings
        .cast(pl.Float32)
    )
    # Set null for non-matching or missing values
    .otherwise(pl.lit(None).cast(pl.Float32).shrink_dtype())
    .alias('five_star_scale_rating_clean')
)


# Now, we inspect our newly created `rating_clean` & `five_star_scale_rating_clean` variables, and compare them to the original `rating` variable

# In[16]:


rating_columns = ('rating', 'rating_clean', 'five_star_scale_rating_clean')

df.select(
    rating_columns
).unique(
    subset=rating_columns,
).sort(
    by='five_star_scale_rating_clean',
    descending=True,
)


# As evidenced, data was successfully reformatted (and assigned to `rating_clean`) as well as parsed to `Float` (and assigned to`five_star_scale_rating_clean`).

# ### Shrinking `ratings_count`
# 
# The `ratings_count` column is correctly parsed as a numerical variable, but we can shrink it, as its range is very small (from `1` to `1533`), as we can see in the summary statistics below.

# In[17]:


# Summary statistics the `ratings_count` variable
df.select(
    'ratings_count'
).describe()


# In[18]:


# Shrinking the datatype for the `ratings_count` column
df = df.with_columns(
    pl.col('ratings_count')
    .shrink_dtype()  # Shrink numeric columns to the minimal required datatype.
    .alias('ratings_count_clean')
)


# Now, we can see the result of the shrinkage, which transformed the data type from `Int64` to `Int16`:

# In[19]:


ratings_count_columns = ['ratings_count', 'ratings_count_clean']
ratings_count_columns_dtypes = zip(
    [f'{x}_dtype' for x in ratings_count_columns],
    df.select(ratings_count_columns).dtypes,
)

dict(ratings_count_columns_dtypes)


# ## Cleaning `condition`
# 
# By analyzing the `condition` variable, we realize it is composed of two separate items, label and description. It is important to create such columns within the dataset:
# - **`condition_label`**: A categorical label which represents the condition of a laptop (e.g: `new`, `used`, `certified_refurbished`)
# - **`condition_description`**: A description for the `condition_label` (e.g: `A brand-new, unused, unopened laptop...`)

# In[20]:


# Taking a look at the values counts for the `condition` column
df['condition'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# Observations:
# - **Similar categories**: Labels `UsedAn item that has been used previously` and `Used: An item that has been used previously` seem to contain duplicate or nearly identical information but are represented differently. Similar things occur for other labels, such as `For parts or not working`. These discrepancies may need to be cleaned and consolidated into one label.
# - **Bad formatting**: A considerable amount of labels show formatting issues (e.g., "UsedAn item...") or seem to be concatenated with additional descriptions. This would require text processing to standardize the condition labels and improve consistency.

# In[21]:


# Creating a dictionary which represents the condition labels and descriptions

condition_replace_map = {
    'New': """A brand-new, unused, unopened, undamaged item in its original packaging.
    Packaging should be the same as what is found in a retail store, unless the item is handmade or was packaged by the manufacturer in non-retail packaging, such as an unprinted box or plastic bag.""",

    'Open box': """An item in excellent, new condition with no wear.
    The item may be missing the original packaging or protective wrapping, or may be in the original packaging but not sealed.
    The item includes original accessories and may be a factory second.""",

    'Certified - Refurbished': """The item is in pristine, like-new condition.
    It has been professionally inspected, cleaned, and refurbished by the manufacturer or a manufacturer-approved vendor to meet manufacturer specifications.
    The item will be in new packaging with original or new accessories.""",

    'Excellent - Refurbished': """The item is in like-new condition, backed by a one-year warranty.
    It has been professionally refurbished, inspected, and cleaned to excellent condition by qualified sellers.
    The item includes original or new accessories and will come in new generic packaging.""",

    'Very Good - Refurbished': """The item shows minimal wear and is backed by a one-year warranty.
    It is fully functional and has been professionally refurbished, inspected, and cleaned to very good condition by qualified sellers.
    The item includes original or new accessories and will come in new generic packaging.""",

    'Good - Refurbished': """The item shows moderate wear and is backed by a one-year warranty.
    It is fully functional and has been professionally refurbished, inspected, and cleaned to good condition by qualified sellers.
    The item includes original or new accessories and will come in a new generic packaging.""",

    'Seller refurbished': """The item has been restored to working order by the eBay seller or a third party.
    This means the item was inspected, cleaned, and repaired to full working order and is in excellent condition.
    This item may or may not be in original packaging.""",

    'Used': """An item that has been used previously.
    The item may have some signs of cosmetic wear but is fully operational and functions as intended.
    This item may be a floor model or store return that has been used.""",

    'For parts or not working': """An item that does not function as intended and is not fully operational.
    This includes items that are defective in ways that render them difficult to use, items that require service or repair, or items missing essential components."""
}


# In[22]:


# Creating a function to map condition values to a condition dictionary, which represents labels (e.g: `New`) as keys and description as values (e.g: `A brand-new, unused item...`).
def map_condition(value: str, condition_dict: dict[str, str] = condition_replace_map) -> tuple[str, str]:
    """
    Maps a given condition string to a predefined label and its corresponding description.

    Args:
        value (str): The condition value to be matched.
        condition_dict (dict, optional): A dictionary where keys are condition labels and
                                         values are descriptions. Defaults to `condition_replace_map`.

    Returns:
        tuple: A tuple containing the matched label (str) and its corresponding description (str).
               If no match is found, returns ('No label', 'No description').

    Example:
        >>> map_condition("Excellent - Refurbished device")
        ('Excellent - Refurbished', 'The item is in like-new condition, backed by a one-year warranty. ...')

        >>> map_condition("Unknown condition")
        ('No label', 'No description')
    """
    if isinstance(value, str):
        for label, description in condition_dict.items():
            if label in value:
                return label, description
    return 'No label', 'No description'


# ### Separating condition label and condition description
# 
# - **Splitting data**: We assign values from the `condition` column (like `New`, `Used`, or `Refurbished`) to a standardized label and description. This ensures that any variation in wording across the data is unified, so we can easily categorize and compare items.
#   - For example, if we have a condition like `Excellent - Refurbished`, it gets split into two parts: a **label** (`Excellent - Refurbished`) and a **description** that explains the condition in more detail (e.g: The item is in like-new condition, backed by a one-year warranty...).
# - **Mapping conditions**: Repeated labels with different formats, such as `UsedAn item that has been used previously` and `Used: An item that has been used previously` are mapped to a single `condition_label` (`'Used'`) which helps us achieve a better label representation by getting rid of different categories that represent the same status (`Used`, `New`, etc.).
# 

# In[23]:


# Transforming the dataframe, creating the `condition_label` and `condition_description` variables
df_lazy_conditions = df.lazy().with_columns(
    pl.col('condition')
    # Applying the `map_condition` function, which maps a string to a predefined label and its corresponding description
    .map_elements(
        function=map_condition,
        skip_nulls=False,
        return_dtype=pl.List(pl.Utf8)
    ).alias('new_condition')  # Creating a temporary column to store results
).with_columns(
    pl.col('new_condition').list.get(0).alias('condition_label_clean'),  # Extract label
    pl.col('new_condition').list.get(1).alias('condition_description_clean')  # Extract description
).drop('new_condition')  # Dropping the temporary column

# Visualize the query plan for the transformation
LazyFrame.show_graph(df_lazy_conditions)


# Here we take a look at the values from our `condition` column, and their correspondent **condition labels** and **condition descriptions**.

# In[24]:


# Selecting all unique values for condition, and their soon-to-be assigned condition labels and condition descriptions
df_lazy_conditions.unique(
    subset='condition',
    maintain_order=True,
).sort(
    by='condition',
    descending=False,
).select(
    'condition', 'condition_label_clean', 'condition_description_clean'
).collect().to_pandas()


# ### **Reformatting `condition`:**
# 
# After splitting our `condition` column into `condition_label` and `condition_description`, we will further clean the `condition_label` column by doing the following string operations:
# - **Lowercasing:** Modifying all strings to their lowercase equivalent.
# - **Standardizing format:** Replacing occurrences hyphens and whitespaces with an underscore (`_`), making the labels more consistent and clean.
# 

# In[25]:


# Reformatting `condition_label_clean` and `condition_description_clean`: replacing blank spaces for underscores and lowercasing
df = df_lazy_conditions.with_columns(
    pl.col('condition_label_clean')
    .str.to_lowercase()  # Converting the text to lowercase
    .str.replace_all(' - ', '_')  # Replacing hyphen and spaces between words with underscores
    .str.replace_all(' ', '_')  # Replacing spaces with underscores
    .alias('condition_label_clean')
).collect()  # Materializing the LazyFrame into a DataFrame

# Selecting all unique values for condition, and their respective reformatted condition labels (condition description is not modified, so it is omitted from this lookup)
df.unique(
    subset='condition',
    maintain_order=True,
).sort(
    by='condition',
    descending=False,
).select(
    ['condition', 'condition_label_clean']
)


# Finally, we take a look at the new `condition_label` variable and its value counts:

# In[26]:


# Taking a look at the values counts for the `condition_label_clean` column
df['condition_label_clean'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# Observations:
# - **New label frequencies**: After cleansing, the most common conditions are `used` (42%) and `new` (29%), which account for the majority of the dataset (71%).
# - **Lower cardinality**: We have a considerably minor number of unique values within our `condition_label` variable (**just 10 unique values**), than we did with `condition`, due to the proper cleansing of the data.

# ## Cleaning `processor`
# - **Formatting**: Similarly, as done with other variables, we'll reformat the values, so that `Intel Celeron` is transformed into `intel_celeron`.
# - **Unknown values**: There's an abundance of brands with very low frequencies, or unknown values (like `?` or `Does not apply` or `no` or `none`).
# 

# In[27]:


# Taking a look at the values counts for the `processor` column
df['processor'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# Observations:
# - **Missing data**: 42% of the entries are marked as `null` and 8% as `Does not apply`, together representing 50% of the dataset.
# - **Low variety**: All processors in the top 10 are of `Intel` brand, with diversity in performance levels but very little in manufacturer.
# - **High cardinality**: `processor` contains 462 unique values, as some of its information contains details such as brand (`intel`/`amd`), model (`celeron`), generation (`8th gen`), etc.
# - **Feature engineering potential**: Some processors indicate both the model and the generation (`Intel core i5 8th gen`), which can be extracted for further analysis.

# ### **Reformatting `processor`:**
# 
# In this step of our analysis, we're focusing on cleaning up and standardizing the `brand` data to ensure consistency and accuracy. Brands can be listed in many different ways, which can lead to confusion or misinterpretation during analysis. So, we're transforming the brand names into a uniform format that makes it easier to work with.
# 
# - **Consistent Formatting:** We ensure all processor names are in lowercase, and we remove any unnecessary spaces or punctuation that could create inconsistencies. (transforming `Intel core - 7th gen.` into `intel_core_7th_gen`).
# - **Replacing Uncertain Values:** We also handle special cases where the data might have uncertain or irrelevant values (e.g., `?` or `no`), replacing them with more meaningful values such as `unknown` or `not_applicable`.

# In[28]:


# Define a mapping to handle various non-standard or missing values in the 'processor' column.
processor_replace_map = {
   '?': 'unknown',
   'none': 'unknown',
   'no': 'unknown',
   '^?': 'unknown',
   'does not apply': 'not_applicable'
}

# Apply the replacement logic to the 'processor' column to clean and standardize the values.
df = df.with_columns(
   pl.col('processor')
   .str.to_lowercase()  # Convert text to lowercase for consistency.
   .str.strip_chars()  # Remove any leading or trailing whitespace.
   .str.replace_all(r'\s+|-', '_')  # Replace spaces / hyphens with underscores for consistent naming.
   .str.replace(',', '')  # Removing comas
   .str.strip_chars_end('.')  # Remove any trailing dots.
   .replace(processor_replace_map)  # Apply the replacement map to clean non-standard values.
   .alias('processor_clean')
)

df.select(
   'processor', 'processor_clean'
).head(n=10)


# In[29]:


# Taking a look at the poorly formatted `processor` values, with their correspondent values in `processor_clean`
df.select(
    'processor', 'processor_clean',
).filter(
    pl.col('processor').str.to_lowercase().is_in(processor_replace_map)
).unique(
    subset='processor',
    maintain_order=True,
).sort(
    by='processor',
    descending=False,
)


# Finally, we inspect visually our newly created `processor_clean` variable:

# In[30]:


# Taking a look the new `processor_clean` column and its value counts
df['processor_clean'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# ## Cleaning `screen_size`

# In[31]:


# Screen size value counts and their proportion
df['screen_size'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
).to_pandas()


# ### Extracting inches from `screen_size`
# 
# In order to have an accurate processing of `screen_size`, we'll extract the inches from the `screen_size` column and assign them to a numerical variable named `screen_size_inches`.
# 
# In this way, we can perform further analysis, visualization and numerical computations on the screen size, ensuring that we no longer have to deal with mixed formats.

# In[32]:


# Define the regex pattern to extract numeric values (floating point or integer) from screen_size column
screen_size_pattern = r'(\d+(\.\d+)?)'

df = df.with_columns(
    # Clean the screen_size column by extracting the numeric part (screen size in inches)
    pl.col('screen_size')
    .str.extract(screen_size_pattern)  # Extract the numeric part
    .cast(pl.Float32)  # Convert the extracted value to Float64
    .alias('screen_size_inches_clean')
)

df['screen_size_inches_clean'].head(n=10)


# We can now take a look at the summary statistics for the new `screen_size_inches` variable

# In[33]:


# Looking at the summary statistics for the `screen_size_inches` variable
df['screen_size_inches_clean'].describe()


# Observations:
# - **Wide Distribution**: The `screen_size_inches_clean` column has a mean value of 14.40 inches, with a standard deviation of 24.22 inches, indicating a relatively wide spread in values. 25% of the data is below 13.0 inches, and 75% is below 15.0 inches.
# - **Missing Data**: The column contains a significant amount of missing values (`null_count = 3289`). This highlights the need for potential imputation or cleaning strategies to handle missing data effectively.
# - **Extreme Outliers**: The minimum value is 2.0 inches, while the maximum reaches up to 1000.0 inches, suggesting some extreme outliers, which could distort any analysis or model using this data. These outliers should be further investigated and possibly removed or transformed, when doing *statistical analysis*.

# ### Correcting outliers
# 
# We're interested in correcting the outliers in the `screen_size_inches` column, as the maximum value is that of a `1000` inches, which makes very little sense. We'll start by visualizing the top values from that column, to see whether we have several outliers or this is just the product of an error in the data extraction process:

# In[34]:


df['screen_size_inches_clean'].filter(
    predicate=df['screen_size_inches_clean'].is_not_null()
).sort(
    descending=True,
)


# We immediately realize we have just two values of `1000` inches, which are heavily skewing the data. The rest of the distribution lies between the range of `2-18` inches, which is sensible.
# In order to address this, we need to find where these values are originally located. Which is, the `screen_size` column:

# In[35]:


df.select('screen_size').filter(pl.col('screen_size').str.contains('1000'))


# We come to the immediate realization that `Jumbo Heuer 1000`, the original `screen_size` value, is clearly not a valid screen size.
# 
#  It doesn't contain any numeric value that would correspond to a typical screen size in inches (e.g., "15.6", "12.5", etc.). Instead, the word "1000" is part of a non-numeric string, likely from a misformatted entry during the web scraping process, or the scraping of a wrong tag. As a result, extracting 1000 as the screen size would lead to misleading or incorrect analysis of the data, as screen sizes in inches don't reach values like 1000 inches.
# 
# Hence, we replace these unreasonable values with `None` to ensure that the data used for analysis remains consistent and meaningful.

# In[36]:


# Replace rows where screen_size_inches_clean is equal to a thousand (which is incorrect data)
df = df.with_columns(
    pl.when(pl.col('screen_size_inches_clean').is_in([1000]))  # Handle extreme values
    .then(None)  # Replace with None (we might also consider using a default value, like the mean or median)
    .otherwise(pl.col('screen_size_inches_clean'))
    .alias('screen_size_inches_clean')
)

# Sorting the screen size in inches in descending order
df['screen_size_inches_clean'].filter(
    predicate=df['screen_size_inches_clean'].is_not_null()
).sort(
    descending=True,
)


# Now, we see that the maximum value for the *screen size in inches* variable make a lot more sense. We proceed to visualize its summary statistics once again:

# In[37]:


# Looking back at the summary statistics for the `screen_size_inches` variable
df['screen_size_inches_clean'].describe()


# Observations:
# - **Narrower distribution**: The new statistics show a mean of `13.81` inches with a standard deviation of `1.61` inches. This is an **HIGH REDUCTION** in variability, particularly compared to the previous standard deviation of `24.22` inches. This suggests that removing extreme outliers has led to a more consistent and predictable distribution of screen sizes.

# ## Cleaning `manufacturer_color` & `color`
# Manufacturer color consists of the color specified by the laptop manufacturer. By analyzing the value counts in the `manufacturer_color` column, we can visualize that:
# - Manufacturer color is almost composed entirely of `NULL` values, with 97% of its values being null.
# - Some entries represent multiple colors in a single field (e.g: `Black & Silver`).
# - Most entries have very little frequency, appearing a single time in the entire column. These rare values might be considered noise (and grouped into a rare label category, such as `rare` or `other`) or might to be grouped into broader categories by using the color label (such as grouping `Mica Silver` and `Ice Blue` into a broader `Blue` category).

# In[38]:


# Color value counts and their proportion
df['manufacturer_color'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# Observations:
# - **Almost all data is NULL**: The `manufacturer_color` column contains a large amount of missing data, with 6435 instances marked as `null`, representing 97% of the dataset. This makes the variable have very little influence for further analysis or modeling techniques.
# - **Diverse color variants**: The column contains various color names with different variations and formats, such as `Black & Silver` and `Matte Black`, as well as entries like `Iron Grey, Ice Blue` and `Scarlet red cover and base, natural silver keyboard frame` which suggest complex multi-color or detailed descriptions. These need to be **STANDARDIZED FOR CONSISTENCY**.
# - **Low Frequency Entries**: A significant number of color values appear only once or twice. These rare values might be considered noise or need to be grouped into broader categories to improve the utility of this feature for analysis or modeling.
# 

# Similarly, `color` consists of the actual color of the laptop as listed by the seller. By analyzing the value counts in the `color` column, we can visualize that:
# - Color is currently a variable with a high number of nulls, which compose approx. 68% of the column.
# - Color contains data in spanish, as evidenced by some of its values: `borgoña` (burgundy), `blanco` (white) or `negro` (black).
# - Some entries represent multiple colors in a single field (e.g: `Black/ Blue / Sandtone / Platinum`).

# In[39]:


# Color value counts and their proportion
df['color'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# Observations:
# - **Missing Data**: A significant portion of the data is missing, with 68% of instances marked as `null`. This indicates that a large part of the `color` information is unavailable, which may require additional cleaning, maybe imputation strategies, to handle effectively.
# - **Different Languages**: The `color` column includes color names in spanish, such as `Negro` (Spanish for black) and `Gris` (Spanish for gray), as well as English terms like `Carbon Black` and `Silver`. This inconsistency in language could lead to confusion in analysis or models, and standardizing the language or grouping similar colors may be beneficial.
# - **Multiple Colors**: Some entries, like `Black/ Blue / Sandtone / Platinum` or `Multicolor`, represent multiple colors in a single field. These values may need to be split into individual colors or categorized into a broader "multicolor" group to maintain consistency and facilitate analysis.

# ### **Reformatting `manufacturer_color`:**
# In this step of our analysis, we're focusing on cleaning up and standardizing the `manufacturer_color` data to ensure consistency and accuracy.
# - **Reducing cardinality:** The color data may contain variations or inconsistencies in how colors are labeled (e.g., `multi-color` vs. `multicolor`). We address this by grouping similar colors under consistent labels, reducing the number of unique color categories. We also map color variations (e.g, `sky blue`, `light blue`) to a single color (`blue`).
# - **Standardizing format:** Check whether the color listed for each item matches a set of predefined valid colors. Ensures that all color data follows a consistent format.

# In[40]:


# List of valid color values for consistency checks
valid_color_values = ['beige', 'black', 'blue', 'bronze', 'brown', 'burgundy', 'gold', 'gray', 'green', 'grey',
                      'orange', 'pink', 'platinum', 'purple', 'red', 'silver', 'teal', 'white', 'yellow']

# Spanish-to-English color translation map
colors_translation = {
    'negro': 'black',
    'borgoña': 'burgundy',
    'platino': 'platinum',
    'gris': 'grey',
    'blanco': 'white',
    'plata transparente': 'transparent silver',
}

# Color replacement map
color_replace_map = {
    'multi': 'multicolor',
    'multi-color': 'multicolor',
    'blk': 'black',
    **colors_translation,
}

color_replace_map


# In[41]:


def clean_color(color: str, color_list: list[str] = valid_color_values) -> str:
    """
    Cleans the input color string by categorizing it based on the presence of valid colors.

    The function checks if the input `color` matches any of the colors in the `color_list`.
    It returns a category based on the number of matches:
    - If the color matches more than one valid color, it returns 'multicolor'.
    - If the color matches exactly one valid color, it returns that color (e.g: 'red').
    - If no valid colors are found, it returns 'other'.

    Args:
        color (str): The input color string to be cleaned and categorized.
        color_list (list, optional): A list of valid colors to check against. Defaults to `valid_color_values`.

    Returns:
        str: The cleaned and categorized color, either a valid color, 'multicolor', or 'other'.
    """
    color_count = sum(1 for c in color_list if c in color.lower())

    if color_count > 1:
        return 'multicolor'
    elif color_count == 1:
        return next((c for c in valid_color_values if c in color.lower()), 'other')
    else:
        return 'other'


# In[42]:


# Creates an expression to search for valid colors
valid_color_string_pattern = "|".join(re.escape(color) for color in valid_color_values)

# Prepare the color column by converting to lowercase and applying the color replacements
manufacturer_color_reformatted = (
    pl.col('manufacturer_color')
    .str.to_lowercase()
    .replace(color_replace_map)
)

df = df.with_columns(
    pl.coalesce(
        # If color is valid then apply the `clean_color` function, and map its value to a valid color, multicolor or other (rare label, for invalid color categories)
        pl.when(manufacturer_color_reformatted.str.contains(valid_color_string_pattern))
        .then(manufacturer_color_reformatted.map_elements(clean_color, return_dtype=pl.Utf8)),
        # If color value is set to multicolor, then return multicolor
        pl.when(manufacturer_color_reformatted.eq('multicolor'))
        .then(pl.lit('multicolor'))
        # If value is not multicolor nor a single color,then assign it to `other`
        .otherwise(pl.lit('other'))
    )
    .str.replace('grey',
                 'gray')  # Naturally, we need to map grey to gray (or vice versa) as it represents the same color
    .alias('manufacturer_color_clean')
)

df.select(
    'manufacturer_color', 'manufacturer_color_clean'
).unique(
    subset='manufacturer_color',
    maintain_order=True,
).head(10)


# We further inspect the original unique values of the `manufacturer_color` column, and its correspondent mapping in `manufacturer_color_clean`:

# In[43]:


# Taking a look at all the transformed color labels
df.select(
    'manufacturer_color', 'manufacturer_color_clean',
).unique(
    subset='manufacturer_color',
    maintain_order=True,
).sort(
    by='manufacturer_color',
).transpose(
    include_header=True,
    header_name='column_name',
)


# ### **Reformatting `color`:**
# We do similarly with `color` as we did with `manufacturer_color` by translating, reducing cardinality and standardizing its format in the following manner:
# - **Translating from Spanish to English:** We translate color names from Spanish to English to ensure that all color information is standardized, regardless of the language it was originally provided in.
# - **Reducing cardinality:** The color data may contain variations or inconsistencies in how colors are labeled (e.g., `multi-color` vs. `multicolor`). We address this by grouping similar colors under consistent labels, reducing the number of unique color categories. We also map color variations (e.g, `sky blue`, `light blue`) to a single color (`blue`).
# - **Standardizing format:** check whether the color listed for each item matches a set of predefined valid colors. Ensures that all color data follows a consistent format.

# In[44]:


# Prepare the color column by converting to lowercase and applying the color replacements
color_reformatted = (
    pl.col('color')
    .str.to_lowercase()
    .replace(color_replace_map)
)

df = df.with_columns(
    pl.coalesce(
        # If color is valid then apply the `clean_color` function, and map its value to a valid color, multicolor or other (rare label, for invalid color categories)
        pl.when(color_reformatted.str.contains(valid_color_string_pattern))
        .then(color_reformatted.map_elements(clean_color, return_dtype=pl.Utf8)),
        # If color value is set to multicolor, then return multicolor
        pl.when(color_reformatted.eq('multicolor'))
        .then(pl.lit('multicolor'))
        # If value is not multicolor nor a single color,then assign it to `other`
        .otherwise(pl.lit('other'))
    )
    .str.replace('grey',
                 'gray')  # Naturally, we need to map grey to gray (or vice versa) as it represents the same color
    .alias('color_clean')
)

df.select(
    'color', 'color_clean'
).head(n=10)


# We further inspect the original unique values of the `color` column, and its correspondent mapping in `color_clean`:

# In[45]:


# Taking a look at all the transformed color labels
df.select(
    'color', 'color_clean',
).unique(
    subset='color',
    maintain_order=True,
).sort(
    by='color',
).transpose(
    include_header=True,
    header_name='column_name',
)


# We can also visualize its new value counts:

# In[46]:


# Color cleaned value counts and their proportion
df['color_clean'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# Observations:
# - **Lower cardinality and higher frequencies**: As a consequence of mapping colors into broader color categories and the rare category (`other`) the cardinality has decreased from more than 80 different colors to 20. This has also caused the increase in the proportion of several categories, such as `black` increasing from 15% to 19%.

# ## Cleaning `gpu`

# In[47]:


df['gpu'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# In[48]:


def map_gpu_type(gpu_str: str) -> str:
    """
    Maps GPU names to broader categories based on brand.

    - 'intel' if the GPU name contains 'intel'.
    - 'amd' if the GPU name contains 'amd' or 'radeon'.
    - 'nvidia' if the GPU name contains 'nvidia' or 'geforce'.
    - 'mali' if the GPU name contains 'mali'.
    - 'other' for unknown or unclassified GPU names.

    Args:
        gpu_str (str): The GPU name.

    Returns:
        str: Mapped GPU category.

    Examples:
        >>> map_gpu_type("Intel UHD Graphics")
        'intel'
        >>> map_gpu_type("AMD Radeon RX 6800")
        'amd'
        >>> map_gpu_type("NVIDIA GeForce RTX 3080")
        'nvidia'
        >>> map_gpu_type("ARM Mali-G76")
        'mali'
        >>> map_gpu_type("Unknown GPU")
        'other'
    """
    if gpu_str is None or gpu_str == "":
        return "unknown"

    str_casefold = gpu_str.casefold()

    if "intel" in str_casefold:
        return "intel"
    elif "amd" in str_casefold or "radeon" in str_casefold:
        return "amd"
    elif "nvidia" in str_casefold or "geforce" in str_casefold:
        return "nvidia"
    elif "mali" in str_casefold:
        return "mali"
    else:
        return "other"


# In[49]:


df.with_columns(
    pl.col("gpu")
    .map_elements(map_gpu_type, skip_nulls=False, return_dtype=pl.Utf8)
    .alias("gpu_type_clean")
)['gpu_type_clean'].value_counts(sort=True)


# ## Cleaning `type`
# 
# The `type` variable represents the specific type of laptop, such as notebook, ultrabook, 2-in-1, gaming laptop, etc. It is a categorical variable that helps categorize laptops based on their physical design, usage, or form factor.
# 
# The main issues with this variable are:
# - **Inconsistent format**: The values in the `type` column may not follow a consistent format or include variations in the naming convention (e.g., `Notebook` vs `notebook` or `Ultrabook` vs `ultrabook`).
# - **Missing Data**: Like many other columns, the `type` column may also contain missing or null values, which need to be handled.
# - **Ambiguity and Overlap**: Some types may overlap or be ambiguous (e.g., laptops that could be classified as both a `notebook` and `ultrabook`), leading to potential confusion or misclassification.
# 
# Therefore, it is essential to clean this variable by *standardizing* the names, *handling missing values* & *grouping* similar types.

# In[50]:


df['type'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# We will clean the variable as aforementioned, by:
# - **Standardizing** the names (e.g., converting everything to `lowercase`, trimming, removing whitespaces).
# - **Handling missing values** (e.g., by imputing or categorizing them as 'unknown').
# - **Grouping similar types** into broader categories to ensure consistency and ease of analysis (e.g: `chromebook` and `notebooks` both get mapped to the general label `laptop`)).
# 

# In[51]:


def map_type(type_str: str) -> str:
    """
    Maps the input string representing the type of device to a standard category.

    The function processes the input string to determine the device type based on the following categories:
    - 'laptop' if any of the terms related to laptops are found (e.g., 'netbook', 'notebook', 'laptop', 'macbook', etc.).
    - 'pc' if any of the terms related to personal computers are found (e.g., 'pc', 'computer').
    - 'tablet' if the string mentions 'tablet'.
    - 'other' for any other cases or unrecognized device types.

    If the input string is None, empty, or contains no recognizable device type, the function returns 'unknown'.

    Args:
        type_str (str): The device type string to be processed.

    Returns:
        str: A standardized device type category.

    Example Usage:
    --------------
    >>> map_type("Macbook Pro 16")
    'laptop'

    >>> map_type("Desktop PC")
    'pc'

    >>> map_type("Samsung Tablet")
    'tablet'

    >>> map_type("Unknown Device")
    'other'

    >>> map_type(None)
    'unknown'

    >>> map_type("")
    'unknown'
    """
    if type_str is None or type_str == "":  # Check for None or blank
        return 'unknown'

    str_casefold = type_str.casefold()

    laptops = ('netbook', 'notebook', 'laptop', 'macbook', 'thinkpad', 'chromebook', 'ultrabook')
    if any(laptop in str_casefold for laptop in laptops):
        return 'laptop'

    computers = ('pc', 'computer')
    if any(pc in str_casefold for pc in computers):
        return 'pc'

    if 'tablet' in str_casefold:
        return 'tablet'

    return 'other'


# In[52]:


df = df.with_columns(
    pl.col('type')
    .map_elements(map_type, skip_nulls=False, return_dtype=pl.Utf8)  # Map the function to the 'Type' column
    .alias('type_clean')  # Create the new column
)


# In[53]:


df['type_clean'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# Observations:
# - **Missing Data**: 50% of the data was missing, now mapped to `unknown`, making the dataset more consistent.
# - **Consolidation**: Similar types like `Notebook`, `Laptop`, and `Pc Portable` were grouped into broader categories like `laptop` and `pc`, reducing noise. Similarly, rare categories were mapped to the `other` label.
# - **Dominant Categories**: The majority of the data now falls into `laptop` (44%), with `unknown` (50%) remaining the largest category due to missing values. Both these categories contain more than 90% of the data.
# 

# ## Cleaning `release_year`

# In[54]:


df['release_year'].value_counts(
    sort=True,
    parallel=True
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# In[55]:


df = df.with_columns(
    pl.col("release_year")
    .str.extract(r"\b(20[0-2][0-9]|200[0-9])\b")  # Extracts valid years (2000-2024)
    .cast(pl.Int64)  # Convert to integer
    .alias("release_year_clean")
)


# In[56]:


df['release_year_clean'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# ## Cleaning `maximum_resolution`

# In[57]:


df['maximum_resolution'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# In[58]:


# Dictionary mapping resolution types to their corresponding width and height values
resolution_replace_map = {
    'full hd': ['1920', '1080'],
    'hd': ['1280', '720'],
    '2k': ['2048', '1080'],
    '4k': ['3840', '2160']
}


def try_get_resolution_value(resolution: str, resolution_dic: dict[str, list[str]], index: int) -> Optional[str]:
    """
    Helper function to retrieve the width or height from the resolution dictionary or the 'width x height' format.

    Args:
        resolution (str): The resolution string to process (e.g., '1920x1080', 'full hd').
        resolution_dic (dict): A dictionary mapping resolution names to their respective width and height.
        index (int): The index (0 for width, 1 for height) to retrieve from the dictionary.

    Returns:
        Optional[str]: The width or height of the resolution as a string, or None if no valid resolution can be extracted.
    """
    # Null check to guard against None values
    if resolution is None:
        return None

    # Normalize the resolution string by removing spaces and converting to lowercase
    resolution = ''.join(resolution.split()).lower()

    # Check if the resolution matches one of the predefined names in the dictionary
    if resolution in resolution_dic:
        return resolution_dic[resolution][index]

    # Check if the resolution contains 'k' and is in the dictionary (e.g., '4k')
    elif 'k' in resolution and resolution in resolution_dic:
        return resolution_dic[resolution][index]

    # Check if the resolution is in the format 'width x height'
    elif 'x' in resolution:
        parts = resolution.split('x')
        if len(parts) == 2:
            # Return the corresponding part (width or height) based on the index
            return re.sub("[a-z()]", '', parts[index]).strip()

    return 'unknown'


# In[59]:


def extract_width_from_resolution_str(resolution: str,
                                      resolution_dic: dict[str, list[str]] = resolution_replace_map) -> Optional[str]:
    """
    Extract the width of a display from a resolution string.

    Uses the helper function to retrieve the width from the resolution dictionary or from the 'width x height' format.

    Args:
        resolution (str): The resolution string to process (e.g., '1920x1080', 'full hd').
        resolution_dic (dict, optional): A dictionary mapping resolution names to their respective width and height.
                                         Defaults to `resolution_replace_map`.

    Returns:
        Optional[str]: The width of the resolution as a string, or None if no valid resolution can be extracted.
    """

    # Try getting the width using the helper function
    return try_get_resolution_value(resolution, resolution_dic, index=0)


def extract_height_from_resolution_str(resolution: str,
                                       resolution_dic: dict[str, list[str]] = resolution_replace_map) -> Optional[str]:
    """
    Extract the height of a display from a resolution string.

    Uses the helper function to retrieve the height from the resolution dictionary or from the 'width x height' format.

    Args:
        resolution (str): The resolution string to process (e.g., '1920x1080', 'full hd').
        resolution_dic (dict, optional): A dictionary mapping resolution names to their respective width and height.
                                         Defaults to `resolution_replace_map`.

    Returns:
        Optional[str]: The height of the resolution as a string, or None if no valid resolution can be extracted.
    """
    # Try getting the height using the helper function
    return try_get_resolution_value(resolution, resolution_dic, index=1)


# In[60]:


df = df.with_columns(
    # Transform the 'maximum_resolution' column to extract the display width
    pl.col('maximum_resolution')
    .str.to_lowercase()  # Convert the resolution string to lowercase for uniformity
    .map_elements(function=extract_width_from_resolution_str, skip_nulls=False,
                  return_dtype=pl.String)  # Apply the width extraction function
    .cast(dtype=pl.Int64, strict=False)  # Cast the result to an integer type
    .shrink_dtype()  # Shrink the data type to the smallest possible type to save memory (if applicable)
    .alias('display_width_clean'),
    # Transform the 'maximum_resolution' column to extract the display height
    pl.col('maximum_resolution')
    .str.to_lowercase()  # Convert the resolution string to lowercase for uniformity
    .map_elements(function=extract_height_from_resolution_str, skip_nulls=False,
                  return_dtype=pl.String)  # Apply the height extraction function
    .cast(dtype=pl.Int64, strict=False)  # Cast the result to an integer type
    .shrink_dtype()  # Shrink the data type to the smallest possible type to save memory (if applicable)
    .alias('display_height_clean')
)


# In[61]:


df['display_width_clean'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# In[62]:


df['display_height_clean'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# ## Cleaning `model`
# 
# The `model` column represents the specific model name or number of the laptop. This could include a combination of words, numbers, and special characters that uniquely identify each laptop model. These values typically provide detailed information about the device, such as the brand, series, generation, and specifications. However, it is essential to understand the structure and characteristics of the data before cleaning it.
# 
# Hence, we clean and standardize these model names to ensure consistency and meaningful analysis.

# In[63]:


# Cleaning and standardizing the `model` variable
df = df.with_columns(
    pl.col('model')
    .str.to_lowercase()  # Convert the string to lowercase
    .str.replace_all(r'\s+', '_')  # Replace spaces with underscores
    .str.replace_all(r'[^a-z0-9_]', '')  # Remove non-alphanumeric characters (except underscores)
    .alias('model_clean')  # Create a new column with the cleaned model names
)

df.select(
    'model', 'model_clean'
).head(n=10)


# ## Cleaning `os`
# 
# #### What does it represent?
# The `os` column represents the operating system installed on the device. This could include common OS types such as `Windows`, `macOS`, `Linux`, `ChromeOS`, and `Android`.
# 
# #### What is wrong with it?
# - **Inconsistent formatting**: Some values might be uppercase, lowercase, or a mix.
# - **Synonyms and variations**: For example, `"Windows 10"`, `"Win 10"`, and `"Windows10"` should all be grouped under `Windows`.
# - **Noisy or irrelevant values**: Some entries might not be valid OS names (e.g., `"See Description"`, `"N/A"`, or random text).
# - **Null values**: Devices with no recorded `os`
# 
# We now proceed to take a further look at the value counts of the `os` variable:

# In[64]:


# Operating system value counts and their proportion.
df['os'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# Observations:
# - **Missing Data**: A significant portion of the data is missing, with 54% of instances marked as `null`. This indicates that more than half of the `os` information is unavailable, which will require appropriate handling, such as imputing missing values with an `'unknown'` category or further investigation into why the data is missing.
# - **Multiple Variations**: The `os` column contains several variations of the same operating system name, such as `Windows 10 Pro`, `Windows 10`, `Windows 10 Pros`, `Windows 11 Pro`, and `Windows 11 Home`. These should be standardized and mapped to a common label, such as `Windows 10` and `Windows 11` or even `Windows`, in order to reduce redundancy and ensure consistency.
# - **Inconsistent Labels**: Entries like `Not Included` or `null` indicate missing or unspecified data that should be grouped as `unknown`. These values should be mapped to a uniform label that indicates the absence of OS information.

# We now map the `os` variable to common categories: `linux`, `windows`, `chrome`, `android`, `mac` and `other`.

# In[65]:


# Define a function that maps the OS types based on the text
def map_os_type(os: str) -> str:
    """
    Maps a given operating system string to a predefined category.

    The function checks for various operating system names within the input string
    and returns a simplified category. It handles different variations of OS names
    and returns an 'unknown' category for missing or non-recognized values.

    Parameters:
    os (str): The input string representing the operating system. This can include
              different OS names such as 'Windows', 'Linux', 'macOS', etc.

    Returns:
    str: A standardized operating system category. Possible categories include:
         'linux', 'windows', 'chrome', 'android', 'mac', 'other', 'unknown'.

    Example:
    >>> map_os_type("Windows 10 Pro")
    'windows'

    >>> map_os_type("Ubuntu 20.04")
    'linux'

    >>> map_os_type("Android 11")
    'android'

    >>> map_os_type(None)
    'unknown'

    >>> map_os_type("Not Included")
    'unknown'
    """
    if os is None or isinstance(os, float) or os == "":  # Check for None or NaN (float type)
        return 'unknown'

    str_casefold = os.casefold()

    # Check for various OS types
    if 'linux' in str_casefold or 'kali' in str_casefold or 'ubuntu' in str_casefold:
        return 'linux'
    elif 'window' in str_casefold or 'windows' in str_casefold or 'win' in str_casefold:
        return 'windows'
    elif 'chrome' in str_casefold:
        return 'chrome'
    elif 'android' in str_casefold:
        return 'android'
    elif 'mac' in str_casefold or 'macos' in str_casefold:
        return 'mac'
    else:
        return 'other'


# In[66]:


df = df.with_columns(
    pl.col("os")  # Select the OS column
    .map_elements(map_os_type, skip_nulls=False, return_dtype=pl.Utf8)  # Apply the function using map_elements
    .alias("os_clean")  # Create a new column with the mapped values
)

# Operating system cleaned value counts and their proportion.
df['os_clean'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# ## Cleaning `country_region_of_manufacturer`
# 
# This variable represents country where the laptop was manufactured. As it is a string variable, it needs to be standardized and reformatted as required.
# 
# We'll start by taking a look at its value counts:

# In[67]:


# Country region of manufacturer value counts and their proportion.
df['country_region_of_manufacturer'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# Observations:
# - **Missing Data**: The `country_region_of_manufacturer` feature has a large proportion of missing values, with 6417 entries marked as `null`, representing 97% of the dataset. This significant amount of missing data should be addressed through imputation or exclusion strategies.
# - **Concentration in Few Regions**: The majority of the values are concentrated in a small set of countries, with China being the most common, followed by the United States and Taiwan. This indicates that the data might be heavily skewed towards certain manufacturers. Other countries such as Japan, Ireland, and Australia have very few entries (less than 1% of the data), which suggests that these values are underrepresented and can be grouped into broader categories for analysis or modeling.
# 

# ### Reformatting `country_region_of_manufacturer`
# We now proceed to reformat the `country_region_of_manufacturer` variable, by doing string operations (such as lowercasing, replacing all whitespaces with underscores) and assigning all `null` values to the `unknown` category.

# In[68]:


df = df.with_columns(
    pl.col('country_region_of_manufacturer')
    .str.to_lowercase()  # Convert all text to lowercase
    .str.replace_all(r'\s+', '_')  # Replace all whitespaces with underscores
    .fill_null('unknown')  # Replace null values with 'unknown'
    .alias('country_of_manufacturer_clean')  # Save as a new column
)

df.select(
    'country_region_of_manufacturer', 'country_of_manufacturer_clean'
).unique(
    subset='country_region_of_manufacturer'
)


# In[69]:


# Country of manufacturer cleaned value counts and their proportion.
df['country_of_manufacturer_clean'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# ## Cleaning `storage_type`
# 
# The `storage_type` column represents the type of storage technology used in the device, such as `HDD`, `SSD`, `eMMC`, or others. This variable typically helps identify the technology driving the device's storage and its performance characteristics.
# 
# Possible values in the `storage_type` column include:
# - **Standard Storage Types**: Entries like `HDD` (Hard Disk Drive), `SSD` (Solid State Drive), `eMMC` (Embedded MultiMedia Card), and `NVMe`, which are the common and widely recognized storage technologies in laptops.
# - **Inconsistent Labels**: Variations in naming conventions, such as `SSD` being labeled as `solid state drive`, or `HDD` being labeled as `Hard Disk` or `hard disk drive`, could create redundancy and inconsistency.
# - **Multiple Values**: Some entries might combine multiple storage types, such as `SSD/HDD` or `HDD+SSD`, making it harder to categorize them properly.
# - **Missing or Placeholder Data**: Missing values or incorrect placeholders like `null` or `unknown` could be present, which would need to be remapped or cleaned.
# 
# We are cleaning this variable to standardize the labels, group similar values, and handle missing or incorrectly formatted data. This will allow for more accurate analysis and ensure the column is more usable in subsequent processes.

# In[70]:


# Storage type value counts and their proportion.
df['storage_type'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# Observations:
# - **Missing Data**: The `storage_type` column contains a significant number of missing values (`null_count = 4232`), representing 64% of the data.
# - **Multiple Formats**: The data includes various representations for similar storage types (e.g., "SSD", "SSD (Solid State Drive)", "Solid State Drive"), which could lead to inconsistencies. These variations need to be standardized and grouped for better analysis and modeling.
# - **Uncommon and Irrelevant Entries**: There are a few outlier values such as "Touchscreen", "256GBSSD", and "SSD or SSD", which seem irrelevant to the `storage_type` category. These should be reviewed and cleaned accordingly to ensure data quality.

# In[71]:


def map_storage_type(string: str) -> str:
    """
    Maps the input string representing a storage type to a standard category.

    The function processes the input string to determine the storage type based on categories: `hdd_or_ssd`, `ssd`, `hdd`, `emmc`, `other`, `unknown` etc.

    If the input string is None, empty, or contains no recognizable storage type, the function returns `unknown`.

    Args:
        string (str): The storage type string to be processed.

    Returns:
        str: A standardized storage type category.

    Usage:
    --------------
    >>> map_storage_type("Samsung SSD 256GB")
    'ssd'

    >>> map_storage_type("Seagate HDD 1TB")
    'hdd'

    >>> map_storage_type("Intel SSD + HDD")
    'hdd_or_ssd'

    >>> map_storage_type("SanDisk eMMC")
    'emmc'

    >>> map_storage_type("Unknown Storage")
    'other'

    >>> map_storage_type(None)
    'unknown'

    >>> map_storage_type("")
    'unknown'
    """
    if string is None or string == "":
        return 'unknown'  # Handle None or empty strings before processing

    # String suitable for caseless comparisons
    str_casefold = string.casefold()

    if 'ssd' in str_casefold and 'hdd' in str_casefold:
        return 'hdd_or_ssd'
    elif 'ssd' in str_casefold:
        return 'ssd'
    elif 'hdd' in str_casefold:
        return 'hdd'
    elif 'emmc' in str_casefold:
        return 'emmc'
    else:
        return 'other'


# In[72]:


df = df.with_columns(
    pl.col("storage_type")  # Select the storage_type column
    .map_elements(map_storage_type, skip_nulls=False, return_dtype=pl.Utf8)  # Apply the function using map_elements
    .alias("storage_type_clean")  # Create a new column with the mapped values
)

df.select(
    'storage_type', 'storage_type_clean'
).head(n=10)


# Now we can analyze the value counts from our new `storage_type_clean` variable, expecting a big reduction in cardinality and a standard formatting for categorical labels.

# In[73]:


# Storage type cleaned value counts and their proportion.
df['storage_type_clean'].value_counts(
    sort=True,
    parallel=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# Observations:
# - **Missing Data**: A significant portion of the data, 61.6%, is marked as `unknown`, which indicates that a large number of storage types are missing or not provided. This missing data may require further investigation or imputation to handle effectively.
# - **Dominant Category**: The most frequent category is `ssd` (35.6%), which suggests that SSD is the predominant storage technology in the dataset, reflecting the growing trend of SSD usage in modern devices.
# - **Reduced cardinality**: Cardinality was reduced from 45 different unique values to 6.
# - **Minor Categories**: The categories `other` (0.01) and `hdd_or_ssd` (0.01) are very small, suggesting that these cases are very uncommon in the dataset.

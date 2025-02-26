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

# In[1]:


# Importing libraries and setting constants
import polars as pl
from polars import LazyFrame
import re

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


# ## Transforming the `brand` variable
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
def return_actual_brand_from_string(string: str, brand_list: list[str] = top_brands) -> str:
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
    .then(pl.col('brand_clean').map_elements(return_actual_brand_from_string, return_dtype=pl.Utf8))
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

# ## Transforming the `price` variable
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


# ## Transforming `rating` & `ratings_count`

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

# ### Transforming `rating`
# 
# In this step, we transform the `rating` variable to variable `five_star_scale_rating` ( which, as its name implies, is a numerical variable ranging from 1-5; representing the product rating on a five star scale).

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


# ## Transforming the `condition` variable
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
def map_condition(value, condition_dict=condition_replace_map):
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


# Finally, we take a look at the new `condition_label` variable:

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

# ## Transforming the `processor` variable
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


# # Transforming the `color` variable
# - `color` is currently a variable with a high number of nulls (approx. 68%)
# - `color` contains data in spanish, as evidenced by some of its values: `borgoña` (burgundy), `blanco` (white) or `negro` (black)

# In[31]:


# Color value counts and their proportion`
df['color'].value_counts().sort(
    by='count',
    descending=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# In[32]:


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


# In[33]:


def clean_color(color: str, color_list: list = valid_color_values) -> str:
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


# ### **Reformatting `color`:**
# In this step of our analysis, we're focusing on cleaning up and standardizing the `color` data to ensure consistency and accuracy.
# - **Translating from spanish to english:** We translate color names from Spanish to English to ensure that all color information is standardized, regardless of the language it was originally provided in.
# - **Reducing color cardinality:** The color data may contain variations or inconsistencies in how colors are labeled (e.g., "multi-color" vs. "multicolor"). We address this by grouping similar colors under consistent labels, reducing the number of unique color categories.
# - **Standardizing format:** check whether the color listed for each item matches a set of predefined valid colors. Ensures that all color data follows a consistent format.

# In[34]:


# Creates an expression to search for valid colors
valid_color_string_pattern = "|".join(re.escape(color) for color in valid_color_values)

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
    ).alias('color_clean')
)

df.select(
    'color', 'color_clean'
).head(n=10)


# We further inspect the original unique values of the `color` column, and its correspondent mapping in `color_clean`:

# In[35]:


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


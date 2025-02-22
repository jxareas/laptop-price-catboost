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

# In[97]:


# Importing libraries and setting constants
import polars as pl
import re
from polars import LazyFrame


# In[98]:


# Loading the dataset
df = pl.read_csv('../data/ebay_laptops_and_notebooks.csv')

# Top 10 rows from the dataframe
df.head(n=10)


# Observations:
# 
# * `Price` variable contains information about the price itself, but also the currency. Also, some prices are in a range format, such as `$399.99 to $634.99` which needs to be handled before converting `Price` into a numerical variable.
# * **Several** columns are being treated as `String` (such as `Price`), which might not be adequate.
# * **Several** columns have **missing values**, which needs to be treated during further data preprocessing.

# In[99]:


# Summary statistics
df.describe()


# In[100]:


# Data Types
pl.DataFrame(data={
    'column': df.columns,
    'dtype': df.dtypes
})


# As aforementioned, many variables are currently being treated as `String`. In the case of `Price`, it is a numerical variable which is being incorrectly read as a categorical variable.

# ## Data Preparation

# In[101]:


# Renaming columns: replacing blank spaces for underscores and lowercasing columns
df = df.rename({col: col.lower().replace(' ', '_') for col in df.columns})
print(df.columns)


# In[102]:


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
# - Extremely high percentage of nulls for columns such as `country_region_of_manufacturer`, `ratings_count` or `release_year`.
# - `Price` column has no null values.

# ## Transforming the `brand` variable
# - TODO: Write `brand` variable exploratory analysis

# In[103]:


df['brand'].value_counts().with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
).sort(by='count', descending=True)


# In[104]:


brand_replace_map = {
    '?': 'unbranded',
    'does_not_apply': 'not_applicable',
}

df = df.with_columns(
    pl.col('brand')
    .str.to_lowercase()
    .str.strip_chars()
    .str.strip_chars_end('.')
    .str.replace_all(' / |\\/', ' or ')
    .str.replace_all(' ', '_')
    .replace(brand_replace_map)
    .fill_null('unknown')
    .alias('brand_clean')
)


# In[105]:


top_brands = df.filter(pl.col('brand_clean').ne('unknown')) \
    .group_by('brand_clean') \
    .agg(pl.len().alias('count')) \
    .sort('count', descending=True) \
    .head(5) \
    .select('brand_clean') \
    .to_series()


is_a_top_brand_string_pattern = "|".join(re.escape(color) for color in top_brands + '_')

def return_actual_brand_from_string(string, brand_list=top_brands):
    for brand in brand_list:
        if brand + '_' in string:
            return brand
    return string

df = df.with_columns(
    pl.when(
        pl.col('brand_clean').str.contains(is_a_top_brand_string_pattern) &
        ~pl.col('brand_clean').str.contains('_or_')
    ).then(pl.col('brand_clean').map_elements(return_actual_brand_from_string, return_dtype=pl.Utf8))
    .otherwise(pl.col('brand_clean'))
    .alias('brand_clean')
)


# In[106]:


df['brand_clean'].value_counts().with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
).sort(by='count', descending=True)


# Key Observations:
# - `dell` has more than 20 new records.
# - # TODO: Write more key findings

# ## Transforming the `price` variable
# 
# - By analyzing the `price` variable, we realize it has information about the currency and price of a laptop.
# - The cost of a laptop might be fixed (e.g: `880`) or a range (e.g: `from 500 to 999`)
# - `price` is currently a categorical (`String`) variable, due to commas, whitespaces and symbols. We **must convert** it into **float**.

# In[107]:


# Taking a look at the `price` variable as is

# First 5 records from the dataframe
df['price'].head(n=5)


# In[108]:


df = df.with_columns(
    pl.col('price').str.slice(0, 1).alias('currency_clean'),
    # Removing the currency symbol, commas from the price and trimming whitespace
    pl.col('price')
    .str.slice(1)  # Removing the currency symbol
    .str.replace(',', '')  # Removing commas
    .str.split("to")  # Separating the price into min_price and max_price range
    .list.first()  # Obtaining the min_price, the first element of the list
    .str.strip_chars()  # Remove leading and trailing characters
    .cast(pl.Float64)  # Casting the min_price to float
    .alias('min_price_clean')
)

df.select([
    'price', 'currency_clean', 'min_price_clean'
]).head(n=5)


# ## Transforming the `condition` variable
# 
# - By analyzing the `condition` variable, we realize it is composed of two separate items, the `condition_label` and the `condition_description`
# - **`condition_label`**: A categorical label which represents the condition of a laptop (e.g: `new`, `used`, `certified_refurbished`)
# - **`condition_description`**: A description for the `condition_label` (e.g: `A brand-new, unused, unopened laptop.`)

# In[109]:


# Taking a look at the values counts for the `condition` column

df['condition'].value_counts().with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
).sort(by='count', descending=True)


# In[110]:


# Creating a dictionary which represents the `condition` labels and descriptions

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


# Creating a function to map condition values to a condition dictionary, which represents labels (e.g: `New`) as keys and description as values (e.g: `A brand-new, unused item`).
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


# In[111]:


# Transforming the dataframe, creating the `condition_label` and `condition_description` variables
df_lazy_conditions = df.lazy().with_columns(
    pl.col('condition').map_elements(
        function=map_condition,
        skip_nulls=False,
        return_dtype=pl.List(pl.Utf8)
    ).alias('new_condition')
).with_columns(
    pl.col('new_condition').list.get(0).alias('condition_label_clean'),
    pl.col('new_condition').list.get(1).alias('condition_description_clean')
).drop('new_condition')

LazyFrame.show_graph(df_lazy_conditions)


# In[112]:


# Selecting all unique values for condition, and their newly created condition labels and condition descriptions
df_lazy_conditions.unique(
    subset='condition',
    maintain_order=True,
).sort(
    by='condition',
    descending=False,
).select(
    ['condition', 'condition_label_clean', 'condition_description_clean']
).collect().to_pandas()


# In[113]:


# Reformatting `condition_label_clean` and `condition_description_clean`: replacing blank spaces for underscores and lowercasing
df = df_lazy_conditions.with_columns(
    pl.col('condition_label_clean')
    .str.to_lowercase()
    .str.replace_all(' - ', '_')
    .str.replace_all(' ', '_')
    .alias('condition_label_clean')
).collect()

# Selecting all unique values for condition, and their respective reformatted condition labels (condition description is left as is)
df.unique(
    subset='condition',
    maintain_order=True,
).sort(
    by='condition',
    descending=False,
).select(
    ['condition', 'condition_label_clean']
)


# ## Transforming the `processor` variable
# 
# - TODO : Write analysis about the processor variable

# In[114]:


processor_replace_map = {
    '?': 'unknown',
    'none': 'unknown',
    'no': 'unknown',
    '^?': 'unknown',
    'does not apply': 'not_applicable'
}

df = df.with_columns(
    pl.col('processor')
    .str.to_lowercase()
    .replace(processor_replace_map)
    .alias('processor_clean')
)

df.select(
    'processor', 'processor_clean'
).head()


# In[115]:


# Taking a look at unique `processor` with their respective `processor_clean` variables
df.select([
    'processor', 'processor_clean',
]).filter(
    pl.col('processor').str.to_lowercase().is_in(processor_replace_map)
).unique(
    subset='processor',
    maintain_order=True,
).sort(
    by='processor',
    descending=False,
)


# # Transforming the `color` variable
# 
# - `color` is currently a variable with a high number of nulls (approx. 68%)
# - `color` contains data in spanish, as evidenced by its values `borgoña` (burgundy), `blanco` (white) or `negro` (black)

# In[116]:


# Color value counts and their proportion`
df['color'].value_counts().sort(
    by='count',
    descending=True,
).with_columns(
    (pl.col('count') / df.height).round(2).alias('proportion')
)


# In[117]:


valid_color_values = ['beige', 'black', 'blue', 'bronze', 'brown', 'burgundy', 'gold', 'gray', 'green', 'grey',
                      'orange', 'pink', 'platinum', 'purple', 'red', 'silver', 'teal', 'white', 'yellow']

colors_translation = {
    'negro': 'black',
    'borgoña': 'burgundy',
    'platino': 'platinum',
    'gris': 'grey',
    'blanco': 'white',
    'plata transparente': 'transparent silver',
}

color_replace_map = {
    'multi': 'multicolor',
    'multi-color': 'multicolor',
    'blk': 'black',
    **colors_translation,
}

color_replace_map


# In[118]:


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


# In[119]:


# Creates an expression to search for valid colors
valid_color_string_pattern = "|".join(re.escape(color) for color in valid_color_values)

color_reformatted = pl.col('color') \
    .str.to_lowercase() \
    .replace(color_replace_map)

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

df


# In[120]:


# Taking a look at the newly transformed color cleansed labels
df.select([
    'color', 'color_clean',
]).unique(
    subset='color',
    maintain_order=True,
).sort(
    by='color',
).transpose(
    include_header=True,
    header_name='column_name',
)


#!/bin/bash

# This script converts a Jupyter notebook (.ipynb) to a Python script (.py) using `nbconvert`.
# It allows for variable notebook filenames, making the script reusable for different notebooks.
# If no notebook name is provided, it defaults to `laptop_data_cleansing`.

# Check if a filename is passed as an argument ($1), otherwise use the default name 'laptop_data_cleansing'.
NOTEBOOK_NAME=${1:-laptop_data_cleansing}

# Navigate to the 'notebooks' directory (relative to the current directory).
# If the directory doesn't exist, print an error message and exit the script.
cd ../notebooks || echo 'Notebook directory could not be found'

# Use `jupyter nbconvert` to convert the specified notebook to a Python script.
# The --to script flag specifies that we want to convert to a .py file.
# The --output flag specifies the name of the output script (stored in ../scripts/ directory).
# The notebook filename is dynamically determined based on the variable NOTEBOOK_NAME.
jupyter nbconvert --to script --output "../scripts/${NOTEBOOK_NAME}" "./${NOTEBOOK_NAME}.ipynb"

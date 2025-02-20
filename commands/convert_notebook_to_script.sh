cd ../notebooks  || echo 'Notebook directory could not be found'
jupyter nbconvert --to script --output ../scripts/laptop_data_cleansing ./laptop_data_cleansing.ipynb
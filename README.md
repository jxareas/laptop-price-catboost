<a name="readme-top"></a>
<br />
<div align="center">
  <a href="#">
   <!-- Replace this logo for a custom official logo -->
    <img src="./assets/logos/catboost_logo.png" alt="Logo" width="150" height="150">
  </a>

<h1 align = "center">
    <i><b>CatBoost</b> Laptop Prices</i>
</h1>
    <!-- Add/Remove categories depending on your project -->
  <p align="center">
    Leveraging CatBoost for Laptop Price Regression
    <br />
    <!-- IMPORTANT NOTE: If you want to append emojis you'll need to add the '-' sign before and after the header, as shown below:  -->
    <a href="#-requirements-">Requirements</a>
    ·
     <a href="#-technologies-">Technologies</a>
    ·
    <a href="#-license-">License</a>
  </p>
</div>

<!-- Here goes the project description -->
This project aims to develop a **predictive model for estimating laptop prices** using a dataset sourced from eBay's
*Laptops & Netbooks* category. 

The dataset, obtained via web scraping, has been cleaned and includes key product
attributes such as **brand, specifications, and listing details**.

## 📝 Requirements 📝

This project uses **Conda (Miniforge)** for managing dependencies. To set up the environment, follow these steps:

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/jxareas/laptop-price-catboost.git
cd laptop-price-catboost
```

### 2️⃣ Create and Activate the Conda Environment`
```bash
conda env create -f environment.yml
conda activate ebay-laptops
```

3️⃣ Verify Installation
```bash
conda list
```

Now you're ready to explore the project's notebooks! 🚀

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🦾 Technologies 🦾

This project leverages a variety of powerful libraries for **machine learning, data preprocessing, and model evaluation**:

- **[CatBoost](https://catboost.ai/)** – The **core modeling library**, designed for gradient boosting with **categorical feature handling** and strong performance in regression tasks.
- **[Scikit-learn](https://scikit-learn.org/)** – Used for **model training**, evaluation, and dataset splitting (*train, validation, and test*).
- **[Polars](https://pola.rs/)** – A high-performance **DataFrame library**, optimized for speed and low memory usage.
- **[Feature Engine](https://feature-engine.readthedocs.io/)** – Provides **feature engineering** techniques for transforming and selecting relevant features.
- **[Optuna](https://optuna.org/)** – A hyperparameter optimization framework for **automated model tuning** to enhance performance.
- **[SHAP](https://shap.readthedocs.io/)** – Powers **Explainable AI**, offering insights into **feature importance** and how they impact model predictions.
- **[Seaborn](https://seaborn.pydata.org/)** – Used for **data visualization** and **Exploratory Data Analysis (EDA)**, leveraging the `seaborn.objects` API for a **Grammar of Graphics**-based approach.
- **[TextBlob](https://textblob.readthedocs.io/)** – A simple yet powerful **Natural Language Processing (NLP)** library, used for **sentiment analysis** on text-based features.

These tools collectively enhance our **data processing, model training, and interpretability** capabilities. 🚀
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📜 License 📜

<!-- Change this license for the one used in your project -->

```
MIT License

Copyright (c) 2025 Jonathan Areas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- This is a custom version of the Read-My-README template, by Jon Areas, 
found at: https://github.com/jxareas/read-my-readme -->



# REAL ESTATE PRICE PREDICTION - IMMO ELIZA

### Table of Contents

1. [Description](#description)
2. [Technology Used](#technology-used)
3. [Data Dictionary](#data-dictionary)
4. [Repo Structure](#repo-structure)
5. [Installation and Setup](#installation-and-setup)
6. [Data Preprocessing](#data-preprocessing)
7. [Modeling Approach](#modeling-approach)
8. [Model Evaluation](#model-evaluation)
9. [Visualization](#visualization)
10. [Usage Instructions](#usage-instructions)
11. [Results](#results)
12. [Acknowledgments](#acknowledgments)

---

### Description

This project is part of the AI Bootcamp at BeCode, focusing on real estate price prediction for Immo Eliza. Using various property features, including location, type, size, and condition, we apply machine learning techniques to develop a regression model that accurately estimates property prices.

### Technology Used

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* Plotly
* Jupyter Notebook
* Matplotlib
* JSON

### Data Dictionary

| COLUMN NAME                       | DATA TYPE | UNIT  | DESCRIPTION                                   |
|-----------------------------------|-----------|-------|-----------------------------------------------|
| id                                | int64     | ----  | Unique identifier for each property.          |
| price                             | float     | €     | Price of the property.                        |
| property_type                     | object    | ----  | General type of the property (e.g., house, apartment). |
| subproperty_type                  | object    | ----  | Specific subtype of the property.             |
| region                            | object    | ----  | Region where the property is located.         |
| province                          | object    | ----  | Province where the property is located.       |
| locality                          | object    | ----  | Local area or town of the property.           |
| zip_code                          | int64     | ----  | Postal code of the property's location.       |
| latitude                          | float     | ----  | Latitude coordinate of the property.          |
| longitude                         | float     | ----  | Longitude coordinate of the property.         |
| construction_year                 | float     | year  | Year the property was constructed.            |
| total_area_sqm                    | float     | m²    | Total area of the property in square meters.  |
| surface_land_sqm                  | float     | m²    | Surface area of the land in square meters.    |
| nbr_frontages                     | float     | ----  | Number of frontages the property has.         |
| nbr_bedrooms                      | float     | ----  | Number of bedrooms in the property.           |
| equipped_kitchen                  | object    | ----  | Indicates if the property has an equipped kitchen. |
| fl_furnished                      | int64     | ----  | Indicates if the property is furnished (1 = Yes, 0 = No). |
| fl_open_fire                      | int64     | ----  | Indicates if the property has an open fire (1 = Yes, 0 = No). |
| fl_terrace                        | int64     | ----  | Indicates if the property has a terrace (1 = Yes, 0 = No). |
| terrace_sqm                       | float     | m²    | Area of the terrace in square meters.         |
| fl_garden                         | int64     | ----  | Indicates if the property has a garden (1 = Yes, 0 = No). |
| garden_sqm                        | float     | m²    | Area of the garden in square meters.          |
| fl_swimming_pool                  | int64     | ----  | Indicates if the property has a swimming pool (1 = Yes, 0 = No). |
| fl_floodzone                      | int64     | ----  | Indicates if the property is in a flood zone (1 = Yes, 0 = No). |
| state_building                    | object    | ----  | Condition or state of the building.           |
| primary_energy_consumption_sqm    | float     | kWh/m² | Energy consumption per square meter.          |
| epc                               | object    | ----  | Energy Performance Certificate rating.        |
| heating_type                      | object    | ----  | Type of heating system in the property.       |
| fl_double_glazing                 | int64     | ----  | Indicates if the property has double glazing (1 = Yes, 0 = No). |
| cadastral_income                  | float     | €     | Cadastral income of the property.             |

### Repo Structure

```
.
├── data/
│ └── data_preprocessed.csv/
│ └── new_data.csv/
│ └── properties.csv/
│ └── standardized_data.csv/
├── models_notebooks/
│ └── bayesian_ridge_regression.ipynb/
│ └── catboost_regressor.ipynb/
│ └── decision_tree_regressor.ipynb/
│ └── elastic_net_regression.ipynb/
│ └── k_nearest_neighbors_regressor.ipynb/
│ └── lasso_regression.ipynb/
│ └── LightGBM_regressor.ipynb/
│ └── model_linear_regression.ipynb/
│ └── random_forest_regression.ipynb/
│ └── ridge_regression.ipynb/
│ └── super_vector_regressor.ipynb/
│ └── XGBoost_regressor.ipynb/
├── notebooks_preprocesing_viz/
│ └── preprocesing.ipynb/
│ └── models_vizualizations.ipynb/
├── results/
│ └── [model results and .joblib files]
├── scripts/
│ └── train.py/
│ └── price_stats.json
│ └── predict.py/
├── README.md/
├── requirements.txt/
├── .gitignore/
```

### Installation and Setup

1. Clone the repository.
2. Install required packages with:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure paths to data files are correctly set within scripts and notebooks.

### Data Preprocessing

Data preprocessing includes handling missing values, encoding categorical variables, and scaling features for model input. We standardized each feature using the mean and standard deviation, enabling the model to process consistent input data.

### Modeling Approach

We experimented with multiple models to predict real estate prices, including:
* Linear Regression
* Random Forest
* CatBoost
* Ridge, Lasso, and Elastic Net
* K-Nearest Neighbors
* LightGBM
* XGBoost

### Model Evaluation

Each model was evaluated based on Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² scores on the training and testing sets. This allowed us to compare performance across various algorithms to select the most accurate and generalizable model.

### Visualization

We include PNG visualizations of:
* Feature importance in tree-based models
* Residuals plots
* Comparison of model performance

These images can be found in the `notebooks_preprocessing_viz/models_vizualizations.ipynb` notebook or directly within the README if hosted on GitHub.
![train_R2](https://github.com/user-attachments/assets/a7c573bd-3c83-438a-8cca-43436f2e1bd4)
![train_mse](https://github.com/user-attachments/assets/74ad3a19-c25a-49b0-8745-b8d4d02568d6)
![train_mae](https://github.com/user-attachments/assets/11524686-277f-4bda-8c5a-95c563747ee9)
![test_R2](https://github.com/user-attachments/assets/78c2467f-3b51-4179-9f24-0c3a00f5f79a)
![test_mse](https://github.com/user-attachments/assets/bf91b10f-6290-4495-9e93-a2e84205b3f0)
![test_mae](https://github.com/user-attachments/assets/0c28c9e6-7ea6-4710-bc07-c58439e9c979)
### Usage Instructions

To train a model:
```bash
python scripts/train.py
```

To make predictions on new data:
```bash
python scripts/predict.py
```

### Results

Among the models tested, CatBoost, LightGBM, Random Forest, and XGBoost show the best performance, with higher R² scores (around 0.72–0.96) and lower MSE and MAE values, indicating strong predictive accuracy. In contrast, simpler models like Linear Regression and Lasso Regression show lower R² scores (around 0.23–0.31), suggesting they may not capture the complexity of the data as effectively. Overall, ensemble models and gradient boosting techniques appear to provide more reliable results for this dataset.

### Acknowledgments

Special thanks to BeCode for the support and guidance throughout the project. Thanks also to the creators of the open-source libraries used in this project.


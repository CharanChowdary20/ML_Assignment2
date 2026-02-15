### ML Assignment 2 ###



### a. Problem statement

The goal of this project is to create a powerful machine learning classification system that can forecast the success of diverse products based on market and supply chain dynamics. The purpose is to implement and compare six different classification algorithms: Logistic Regression, Decision Tree, k-Nearest Neighbors (kNN), Naive Bayes, Random Forest, and XGBoost.

The initiative focuses on the entire ML deployment procedure, not just model training. 

This includes: Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC) are some of the measures used in rigorous performance assessment.

**UI Development** : Created an interactive web interface with Streamlit that allows users to submit test data and select models for real-time inference .

**Deployment**     : Making the solution available on the Streamlit Community Cloud.

By comparing these models, we hope to determine which method best handles the intricacies of product success prediction within the given dataset.


### b. Dataset Description

The dataset used for this assignment is a repository of product performance data used for classification tasks. It has 500 instances and 14 features and 12 of which are used as independent variables in modeling, which meets the minimum assignment requirements of 500 instances and 12 features.

Feature Overview:

Product_Name: The designation of the product (Categorical).

Category & Sub_category: Levels of classification for the product utilized for label encoding (Categorical).

**Price**     : The individual cost of the product (Numerical).

**Rating**    : Average score from customers ranging from 1.0 to 5.0 (Numerical).

**No_rating** : Total number of ratings received (Numerical).

**Discount**  : Percentage of reduction offered on the product (Numerical).

**M_Spend**   : Budget allocated for marketing the product (Numerical).

**Supply_Chain_E**    : Score reflecting the efficiency of the supply chain (Numerical).

**Sales_y & Sales_m** : Total sales figures on a yearly and monthly basis (Numerical).

**Market_T & Seasonality_T** : Indices that indicate market trends and seasonal variations (Numerical).


## c. Comparison Table with the evaluation metrics calculated for all the 6 models as below:


| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
| :---                     | :---     | :---  | :---      | :---   | :---  | :---  |
| Logistic Regression      | 0.988    | 1.000 | 0.995     | 0.977  | 0.986 | 0.976 |
| Decision Tree            | 0.914    | 0.943 | 0.868     | 0.944  | 0.904 | 0.829 |
| kNN                      | 0.880    | 0.947 | 0.889     | 0.823  | 0.855 | 0.755 |
| Naive Bayes              | 0.936    | 0.987 | 0.964     | 0.884  | 0.922 | 0.870 |
| Random Forest (Ensemble) | 0.984    | 0.998 | 0.986     | 0.977  | 0.981 | 0.967 |
| XGBoost (Ensemble)       | 0.984    | 0.998 | 0.981     | 0.981  | 0.981 | 0.967 |


## d. observations on the performance of each model on the chosen dataset.

| ML Model Name           | Observation about model performance |
| :---                    | :---                                |
| **Logistic Regression** | Top-performing model with the highest MCC, indicating the data is highly linearly separable |
| **Decision Tree**       | Performs well but shows lower precision compared to ensemble methods, indicating some variance in splits. |
| **kNN**                 | Lowest overall performer, indicating that simple data proximity is less predictive than the global structural patterns captured by other models .|
| **Naive Bayes**         | Highly robust with high precision, proving that feature independence assumptions work well here. |
| **Random Forest**       | Provides a significant stability boost over the single decision tree with high balanced metrics. |
| **XGBoost**             | Exceptionally balanced precision and recall, demonstrating the efficiency of gradient boosting. |


#Deployment Links :  

**GITHUB**   : https://github.com/CharanChowdary20/ML_Assignment2/tree/main

**STREMLIT** : https://mlassignment2-tsgu26gd2yvlcccznjpmcw.streamlit.app

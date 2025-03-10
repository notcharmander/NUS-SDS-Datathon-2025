# NUS-SDS-Datathon-2025

## Introduction

In today’s globalised economy, understanding corporate ownership and influence is critical for
businesses, investors, and regulators. Companies operate within complex hierarchies, with
entities spanning local (domestic) and international (global) levels. Our goal is to develop a
robust machine learning model to predict the "Is Domestic Ultimate" and "Is Global Ultimate"
classifications for companies based on their operational, financial, and structural characteristics.
Using the given dataset, we have conducted data analysis and model training to develop a
model that provides accurate predictions for these two target variables.

## Exploratory Data Analysis

For categorical variables (‘Ownership Type’, 'Industry'), encoding was necessary to convert
them into numerical values for model training.

For ‘Ownership Type’, since there are relatively few unique categories (Non-Corporates,
Nonprofit, Partnership, Private, Public, Public Sector), we used label encoding to convert each
industry type to numerical values.

For 'Industry’, because there are 581 unique categories in the original dataset, we decided to
use embedded encoding, because it captures semantic relationships between industries in a
continuous vector space, making it more effective than traditional encoding methods. Unlike
one-hot encoding, which creates a high-dimensional sparse matrix that grows with the number
of categories, Word2Vec reduces dimensionality and preserves meaningful similarities between
industries. Additionally, label encoding assigns arbitrary numerical values that do not reflect
relationships between categories, potentially misleading the model. Word2Vec, on the other
hand, generates dense vector representations where similar industries have closer embeddings,
helping machine learning models generalize better. This method is particularly beneficial for
models like XGBoost, which perform better with numerical features and can leverage these
embeddings to learn industry-related patterns effectively.

## Data Cleaning
Firstly, based on our domain knowledge, we removed columns that do not have significant
impact on our target variables ('AccountID','Company', '8-Digit SIC Description', 'Company
Description', 'Company Status (Active/Inactive)').

Secondly, based on summary statistics of the dataset, we decided to drop columns with a high
percentage of null values. For columns with relatively fewer null values, we adopted different
approaches for each column based on the type of data.

Thirdly, we used KNN imputation for columns such as Employees (Single Site), Employees
(Domestic Ultimate Total), Employees (Global Ultimate Total)) as they exhibit numerical and
structured patterns that KNN can effectively leverage. KNN imputation is a powerful technique to
handle missing values, especially since our dataset has many interrelated features. Although it
is computationally heavy, KNN generally outperforms simpler imputation methods such as
median data imputation in terms of accuracy.

## Outlier Analysis
Based on visual inspection of the distribution, we observed that
the values for 'Employees (Single Site)', 'Employees (Domestic Ultimate Total)', 'Employees
(Global Ultimate Total)', 'Sales (Domestic Ultimate Total USD)', and 'Sales (Global Ultimate Total
USD)' were extremely right-skewed. This skewness indicates that a few very large values
(outliers) dominate the distribution, while most data points are concentrated near the lower end.
To address this issue and improve the interpretability of the data, we applied a log
transformation to these variables. The log transformation helps normalize the distribution,
reduce the impact of extreme values, and make patterns in the data more evident for
downstream modeling and analysis. This adjustment is particularly useful for improving the
performance of machine learning models and ensuring better feature scaling.

## Feature Engineering
In our feature engineering step, we created two new ratio-based features:
1. Employee_Ratio = log_Employees_Domestic / log_Employees_Global

A high ratio (≫1) suggests a large international workforce, making it more
likely that the company is a Global Ultimate.
A low ratio (≈0) suggests that the company mostly operates domestically,
making it more likely to be a Domestic Ultimate or a subsidiary.

2. Sales_Ratio = log_Sales_Domestic / log_Sales_Global

A high ratio (≫1) means that most of the company's revenue comes from
international markets, indicating a Global Ultimate.
A low ratio (≈0) means that most sales happen in the home country,
suggesting that the company is either a Domestic Ultimate or a subsidiary.

We decided to use the log transformed values to compute the ratio because it compresses
large values, making the ratio more balanced and interpretable.

These features compare domestic vs. global values, providing insights into relative scale
differences rather than absolute numbers.

Instead of just using absolute employee or sales counts, these ratios help measure the
proportion of domestic to global. Ratio-based features help normalise the data and provide
a more structured comparison.

## Model Selection
### XGBoost Tree-Based Model

We decided to use the XGBoost Tree-Based Model due to its speed, accuracy, and ability to
handle complex data structures efficiently. Boosting corrects errors from previous trees, making
XGBoost less biased and hence more accurate. XGBoost also focuses more on hard-to-predict
samples. XGBoost is fast as compared to other models since it uses parallel computation and
GPU acceleration. Additionally, XGBoost has built-in regularisation to prevent overfitting.

Two separate models were trained:
1. XGBoost model for Global Ultimate classification
2. XGBoost model for Domestic Ultimate classification

Feature importance scores were extracted to identify the top 10 most relevant features for both
classification tasks.

## Train-Test-Validation Splitting
To ensure robust model performance, the dataset was split into training, validation, and test
sets.

1. Train-Test Split (80-20)
- The dataset was split into 80% training and 20% testing. This ensures that the
model learns from 80% of the data while keeping 20% unseen for final
evaluation.
- Stratified sampling was used to maintain the same proportion of Domestic
Ultimate and Global Ultimate companies in the training and test sets.

2. Train-Validation Split (Within Training Set)
- The 80% training data was further split into train (70%) and validation (10%).
- The validation set was used for hyperparameter tuning (Optuna).
- This ensures that hyperparameters are optimized on a separate validation set,
preventing overfitting.

## Handling Imbalanced Data with SMOTE
Since classification tasks often involve imbalanced classes, SMOTE (Synthetic Minority
Over-sampling Technique) was applied to the training data.
This generates synthetic minority class samples, ensuring that both classes have equal
representation in the training data.
This improves the model's ability to classify the underrepresented class (e.g., Domestic
Ultimate companies).

## Hyperparameter Tuning using Optuna
Optuna is a hyperparameter optimisation framework that employs an adaptive search
strategy to efficiently explore the hyperparameter space. Unlike grid search, which tries all
possible combinations, or random search, which selects combinations at random, Optuna uses
a Bayesian optimisation approach that intelligently learns from past trials to find the most
promising hyperparameter values.
This function selects the best combination of hyperparameters by maximising model accuracy.

For our XGBoost classifier, we optimised the following key hyperparameters:

- n_estimators: The number of boosting rounds or trees in the ensemble. A higher
number of estimators can improve performance but may lead to overfitting if not tuned
properly.
- max_depth: The maximum depth of individual trees. Deeper trees capture more
complex patterns but can lead to overfitting.
- learning_rate: The step size shrinkage factor that controls how much each tree
contributes to the final model. A smaller learning rate requires more boosting rounds but
helps prevent overfitting.
- subsample: The fraction of training data randomly sampled to grow each tree. Lower
values prevent overfitting by introducing randomness into the model training.
- colsample_bytree: The fraction of features randomly selected for each tree. Reducing
this value can improve generalization by preventing over-reliance on specific features.
- gamma: The minimum loss reduction required for a node split. Higher values make the
model more conservative, reducing the number of splits and controlling overfitting.
- min_child_weight: The minimum sum of instance weights (Hessian) in a child node.
Larger values prevent small leaves from being created, which helps in controlling
overfitting.
- scale_pos_weight: Adjusts the balance between positive and negative class weights,
particularly useful for handling imbalanced datasets.

## Model Training
After tuning, the best hyperparameters were used to train the final model.
The model was trained on SMOTE-balanced training data and evaluated on the validation
set.
The final test dataset was used to evaluate model performance.

## Results from XGBoost Tree-Based Model
### Global Ultimate Model Performance

The Global Ultimate model achieved a high accuracy of **91.47%**, indicating strong overall
classification performance.
A 0.84 F1-score for Class 1 suggests a good but slightly imbalanced performance, where
recall is stronger than precision.

### Domestic Ultimate Model Performance

The Domestic Ultimate model also achieved a high accuracy of **88.04%**, indicating strong
overall classification performance.
The F1-score balances precision and recall, with 0.89 for Class 1 indicating strong overall
performance. A high recall-driven F1-score shows that the model effectively captures most
Domestic Ultimate companies while maintaining good precision.

## Future Work
- Analyse time-series trends in employee ratios to predict company status changes.
- Include profitability metrics for deeper business insights.
- Experiment with other ML models (e.g., LightGBM, CatBoost) for further improvements.

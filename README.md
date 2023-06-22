# ML-based-Thermobarometry

This project uses **ML-based thermobarometry** to estimate **pressure (P)** and **temperature (T)** conditions using **garnet** and **amphibole** mineral data. The data was extracted using **LEPR**, and the model was developed using **Python** code.

It involves the following steps:

- **Loading the Dataset:** The dataset containing student information is loaded from a CSV file using the pandas library.

- **Handling Missing Values:** Missing values in the dataset are imputed using the KNNImputer from the scikit-learn library.

- **Splitting the Dataset:** The dataset is divided into training, validation, and test sets using the train_test_split function from scikit-learn. The training set is used to train the model, the validation set is used for hyperparameter tuning, and the test set is used for evaluating the model's performance.

- **Standardizing the Data:** The training and validation sets are standardized using the StandardScaler from scikit-learn. Standardization ensures that all features have zero mean and unit variance, which can improve the performance of certain machine learning algorithms.

- **Training the XGBoost Regression Model:** An XGBoostRegressor model is initialized with default parameters and is trained on the training set.

-**Hyperparameter Tuning:** GridSearchCV is employed to perform hyperparameter tuning. The model's hyperparameters, such as the number of estimators, maximum depth, learning rate, subsample, and colsample_bytree, are optimized using cross-validation on the validation set.

- **Model Training with Optimized Hyperparameters:** The XGBoostRegressor model is re-initialized with the best hyperparameters obtained from the hyperparameter tuning step. It is then trained on the training set.

- **Model Evaluation:** The performance of the trained model is evaluated on the test set. The mean squared error (RMSE), normalized RMSE (NRMSE), and R2 score are calculated to assess the model's predictive accuracy.

- **Displaying Results:** The evaluation metrics, including RMSE, NRMSE, and R2 score, are printed to provide insights into the model's performance.

## Dependencies
The following dependencies are required to run this code:

- pandas
- xgboost
- scikit-learn

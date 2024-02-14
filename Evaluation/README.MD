# Titanic Survival Prediction Model

This Python script utilizes the scikit-learn library to create a logistic regression model for predicting survival on the Titanic. The dataset is read from a tab-separated file, and features such as age, sex, and titles extracted from names are preprocessed. The selected features for the model are ["Age", "Sex_female", "Sex_male"]. The 'Cabin' and 'Ticket' columns are dropped, missing 'Age' values are replaced with the mean age, and categorical columns are one-hot encoded.

The data is split into training and testing sets, with 80% used for training and 20% for testing. A logistic regression model is created, trained on the training data, and used to predict survival on the testing set. The model's performance is evaluated using accuracy.

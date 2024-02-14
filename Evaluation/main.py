import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define features for the model
FEATURES = ["Age", "Sex_female", "Sex_male"]  # Best features for the model

# Read data from the Titanic dataset
alldata = pd.read_csv(
    "titanic.tsv",
    header=0,
    sep="\t",
    usecols=[
        "Survived",
        "PassengerId",
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked"
    ],
)

# Extract titles from the 'Name' column and one-hot encode categorical columns
mean_age = '{:.2f}'.format(alldata['Age'].mean())
alldata['Name'] = alldata['Name'].str.extract(r'\b([A-Za-z]+)\.')
alldata = pd.get_dummies(alldata, columns=['Sex', 'Embarked', 'Name'])

# Drop unnecessary columns
alldata = alldata.drop(['Cabin', 'Ticket'], axis='columns')

# Replace missing 'Age' values with the mean_age
alldata['Age'].fillna(mean_age, inplace=True)
alldata = alldata.replace({True: 1, False: 0})

# Split the data into training and testing sets
data_train, data_test = train_test_split(alldata, test_size=0.2)

y_train = pd.DataFrame(data_train["Survived"])
x_train = pd.DataFrame(data_train[FEATURES])

# Create a Logistic Regression model and fit it on the training data
model = LogisticRegression()
model.fit(x_train, y_train)

y_expected = pd.DataFrame(data_test["Survived"])
x_test = pd.DataFrame(data_test[FEATURES])

# Make predictions 
y_predicted = model.predict(x_test)

# Evaluate the model's performance
score = model.score(x_test, y_expected)

print(f"Model score: {score}")

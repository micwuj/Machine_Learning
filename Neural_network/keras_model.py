import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from keras.models import Sequential
from keras.layers import Dense

# Reading data from csv file
# Data was taken from http://untroubled.org/spam/
alldata = pd.read_csv(
    "data.csv",
    usecols=[
        "SPAM",
        "text",
    ],
)

# Removing 'Subject: ' from the beginning of each text in the 'text' column
alldata['text'] = alldata['text'].str.replace('Subject: ', '')

# Assigning features and labels
x = alldata['text']
y = alldata['SPAM']

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Creating a Count Vectorizer to convert text data into a bag-of-words representation
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)

# Convert SciPy sparse matrix to NumPy arrays
X_train = X_train.toarray()
X_test = X_test.toarray()

# Define a simple neural network model using Keras
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1  # Binary classification (spam or ham)

model = Sequential()
model.add(Dense(hidden_size, input_dim=input_size, activation='relu'))
model.add(Dense(output_size, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model in batches
num_epochs = 10
batch_size = 32
num_batches = X_train.shape[0] // batch_size

for epoch in range(num_epochs):
    for batch_start in range(0, X_train.shape[0], batch_size):
        batch_end = batch_start + batch_size
        X_batch = X_train[batch_start:batch_end]
        y_batch = y_train[batch_start:batch_end]
        # print(batch_start)
        
        model.train_on_batch(X_batch, y_batch)

# Evaluate the model on the test set
test_outputs = model.predict(X_test)
predicted_labels = (test_outputs > 0.5).astype(np.float32)
accuracy = accuracy_score(y_test, predicted_labels)
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, predicted_labels, average='binary')

# Evaluation metrics
print(f"Accuracy: {accuracy*100:.2f} %")
print(f"Precision: {precision*100:.2f} %")
print(f"Recall: {recall*100:.2f} %")
print(f"F-score: {fscore*100:.2f} %")

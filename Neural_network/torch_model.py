import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# Define a simple neural network model
class EmailClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EmailClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid() # beacuse loss function is BinaryCrossEntropyLoss()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

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

# Convert NumPy arrays to PyTorch tensors
X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Instantiate the model
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1  # Binary classification (spam or ham)
model = EmailClassifier(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train.view(-1, 1))
    loss.backward()
    optimizer.step()

    # Print the loss for each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}] -> Loss: {loss.item():.4f}')

# torch.save(model.state_dict(), 'email_classifier_model.pth')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predicted_labels = (test_outputs > 0.5).float()
    accuracy = accuracy_score(y_test.numpy(), predicted_labels.numpy())
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_test.numpy(), predicted_labels.numpy(), average='binary'
    )

# Evaluation metrics
print(f"Accuracy: {accuracy*100:.2f} %")
print(f"Precision: {precision*100:.2f} %")
print(f"Recall: {recall*100:.2f} %")
print(f"F-score: {fscore*100:.2f} %")
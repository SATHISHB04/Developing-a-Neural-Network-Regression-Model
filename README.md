# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: SATHISH.B

### Register Number: 212224040299

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
dataset1 = pd.read_csv('EXP 1 DL.csv')
X = dataset1[['INPUT']].values
y = dataset1[['OUTPUT']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
# Name:ARUN S
# Register Number: 212224230023
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x




# Initialize the Model, Loss Function, and Optimizer
# Write your code here
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)
# Name:Arun s
# Register Number:212224230023
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()




        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
loss_df = pd.DataFrame(ai_brain.history)
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```

### Dataset Information
<img width="163" height="403" alt="Screenshot 2026-01-30 143239" src="https://github.com/user-attachments/assets/656ae18c-b12e-458d-b312-1218583e95a5" />



### OUTPUT
LOSS


<img width="329" height="232" alt="Screenshot 2026-01-30 110236" src="https://github.com/user-attachments/assets/fcdcf3f4-7719-4a13-9960-c0af56e9845e" />
<img width="198" height="30" alt="Screenshot 2026-01-30 110255" src="https://github.com/user-attachments/assets/63f87c95-0d5e-4ee2-b21f-c1ff026a8f12" />

### Training Loss Vs Iteration Plot
<img width="702" height="565" alt="Screenshot 2026-01-30 110319" src="https://github.com/user-attachments/assets/fefb2612-e8df-4832-8537-2a261ac34280" />
ot here

### New Sample Data Prediction

<img width="302" height="27" alt="Screenshot 2026-01-30 110330" src="https://github.com/user-attachments/assets/f95b7ae6-173f-4d67-b067-161353d58a93" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.

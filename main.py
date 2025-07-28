import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.layers import Dense #type: ignore
from tensorflow.keras.models import Sequential #type: ignore

train = pd.read_csv("fashion-mnist_train.csv")
test = pd.read_csv("fashion-mnist_test.csv")

print(train.describe())
print(test.describe())

print(train.info())
print(test.info())

# Checking for null values
print(train.isnull().sum().max())
print(test.isnull().sum().max())

# Dropping null values
train.dropna(inplace=True)
test.dropna(inplace=True)

# Checking for duplicates
print(train.duplicated().sum())
print(test.duplicated().sum())

# Dropping duplicate values
train.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)

# Assigning independant and target variables
x_train = train.drop(['label'], axis=1)
y_train = train['label']

x_test = test.drop(['label'],axis=1)
y_test = test['label']

# Normalizing the independant variables to same range
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Building the neural network layers
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(y_train.unique()), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

y_pred = model.predict(x_test)

# Converting probablity distribution to class labels
y_pred_classes = [np.argmax(p) for p in y_pred]
y_true_classes = y_test

print("Accuracy :", accuracy_score(y_true_classes, y_pred_classes))

# Plotting the training and validation acc and loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training acc')
plt.plot(history.history['val_accuracy'], label='validation acc')
plt.title("Model accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label= 'Training loss')
plt.plot(history.history['val_loss'], label= 'validation loss')
plt.title("Model loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

print("Confusion matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("fashion-mnist_train.csv")
x_train = train.drop(['label'], axis=1)

img = x_train.iloc[101].values.reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()
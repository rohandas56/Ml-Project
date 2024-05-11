import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

df = sns.load_dataset('mpg')
df.isnull().sum()
df.dropna(inplace=True)

X = df[['displacement', 'horsepower', 'weight', 'acceleration']]
Y = df.mpg
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
filename = 'mpg_regression.sav'
pickle.dump(model, open(filename, 'wb'))
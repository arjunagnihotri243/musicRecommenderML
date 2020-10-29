import pandas as pd
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

df = pd.read_csv("music.csv")

x = df.drop(columns=['genre'])
y = df['genre']

model.fit(x, y)
predictions = model.predict([[21, 1]])
print(predictions)

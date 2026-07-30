import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
data = {
    'Outlook':['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature':['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity':['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
    'Wind':['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
    'Play':['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
}
df = pd.DataFrame(data)
encoder = LabelEncoder()
X = df.drop("Play", axis=1).apply(encoder.fit_transform)
y = encoder.fit_transform(df["Play"])
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)
importances = model.feature_importances_
print("Information Gain")
for feature, importance in zip(X.columns, importances):
    print(feature, ":", importance)
best_feature = X.columns[importances.argmax()]
print("\nHighest Information Gain Attribute:", best_feature)

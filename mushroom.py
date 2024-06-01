import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv('mushrooms.csv')

categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(data[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded = pd.concat([data, one_hot_df], axis=1)

df_encoded = df_encoded.drop(categorical_columns, axis=1)

x = df_encoded.drop(['class_p', 'class_e'], axis=1)
y = df_encoded['class_p']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
dot_data = graphviz(clf, out_file=None, 
feature_names=x.columns, 
filled=True, rounded=True, 
special_characters=True)
graph = graphviz.Source(dot_data)
graph
#y_pred = clf.predict(x_test)
#print("Acurácia: ", accuracy_score(y_test, y_pred))



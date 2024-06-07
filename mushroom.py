import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import plot_tree

# Lê a tabela
data = pd.read_csv('mushrooms.csv')
data = data.drop(["veil-type"], axis=1)

# OneHotEncoder
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(data[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
df_encoded = pd.concat([data, one_hot_df], axis=1)
df_encoded = df_encoded.drop(categorical_columns, axis=1)

# Separação de treino e teste
x = df_encoded.drop(['class_p', 'class_e'], axis=1)
y = df_encoded['class_e']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Árvore de decisão
clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
plt.figure(figsize=(12, 12))
plot_tree(clf, feature_names=df_encoded.columns, fontsize=5)
plt.show()

# Previsão e acurácia - 0: comestível - 1: venenoso
y_pred_dt = clf.predict(x_test)
print(y_pred_dt)
print("Accuracy: ", round(accuracy_score(y_test, y_pred_dt), 4), "%")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
disp.figure_.savefig('confusion_matrix.png')


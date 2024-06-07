import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import export_graphviz


# Lê a tabela
data = pd.read_csv('mushroom/mushrooms.csv')
data = data.drop(["veil-type"], axis=1)
#data = data.iloc[:, [0, 5]]

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
dot_data = export_graphviz(clf, out_file='ah', feature_names=x.columns, filled=True, rounded=True, 
                           special_characters=True)
graph = graphviz.Source(dot_data)

# Influência dos atributos no modelo
features_list = x.columns.values
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(8,7))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color ="red")
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Influência')
plt.title('Influência dos atributos')
plt.draw()
plt.savefig("atributoinf.png", format='png', dpi=500, bbox_inches='tight')

# Previsão e acurácia
y_pred_dt = clf.predict(x_test)
print(x_test)
print(y_pred_dt)
print("Accuracy: ", round(accuracy_score(y_test, y_pred_dt), 4), "%")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_dt, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
disp.figure_.savefig('confusion_matrix.png')


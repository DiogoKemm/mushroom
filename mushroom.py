import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import export_graphviz

# Lê a tabela
data = pd.read_csv('mushroom/mushrooms.csv')
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
clf.fit(x_train, y_train)
dot_data = export_graphviz(clf, out_file='tree', feature_names=x.columns, filled=True, rounded=True, special_characters=True)


# Previsão e acurácia
y_pred_dt = clf.predict(x_test)
print("Accuracy: ", round(accuracy_score(y_test, y_pred_dt), 4), "%")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_dt)
x_axis_labels = ["Comestível", "Venenoso"]
y_axis_labels = ["Comestível", "Venenoso"]
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Matriz de confusão para árvore de decisão')
plt.savefig("dtcm.png", format='png', dpi=500, bbox_inches='tight')
#plt.show()



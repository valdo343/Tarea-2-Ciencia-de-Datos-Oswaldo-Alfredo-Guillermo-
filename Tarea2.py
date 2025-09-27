import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import anderson
import seaborn as sns
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)


####### Funciones ######
########################


# Imputa los NaN de una columna categórica según la distribución de sus valores.
def imputar_aleatorio(col):
    # Distribución de las categorías existentes (excluyendo NaN)
    probs = col.value_counts(normalize=True, dropna=True)
    
    # Función para reemplazar NaN por una categoría aleatoria según probs
    return col.apply(lambda x: np.random.choice(probs.index, p=probs.values) if pd.isna(x) else x)

##########################
##----------------------##



#Read data
df = pd.read_csv('./data/data.csv', sep=';')

# Inicialmente vemos que no hay datos vacíos, pero hay varios "unknown". Dichos datos los consideraremos como vacíos.
print(df)
print(df.info())

# Revisamos cuáles columnas tienen datos "unknown"
cols_with_unknown = [col for col in df.columns if df[col].isin(["unknown"]).any()]
print(cols_with_unknown)
# Reemplazamos los "unknown" por NaN
df[cols_with_unknown] = df[cols_with_unknown].replace("unknown", np.nan)
# Vemos cuántos datos faltantes hay en cada columna por porcentaje
print(df.isnull().mean() * 100)
# Revisamos los datos de la columna default
print(df["default"].value_counts(dropna = True))
# Imputamos los NaN en las columnas categóricas con los valores aleatorios según la distribución de cada columna.
for col in cols_with_unknown:
    df[col] = imputar_aleatorio(df[col])
# Verificamos que ya no hay datos vacíos
print(df.info())



# Ahora reemplazaremos los datos binarios por 0 y 1 (estos están actualmente como "no" y "yes").
cols_yes_no = [col for col in df.columns if df[col].isin(["yes", "no"]).any()]
print(cols_yes_no)

df[cols_yes_no] = df[cols_yes_no].replace({'no': 0, 'yes': 1}).astype(int)

for col in cols_yes_no:
    print(df[col].value_counts())

print(df)


# Codificación one-hot para variables categóricas no ordinales y no binarias.
cols_not_ord = ['job', 'marital', 'contact', 'month', 'day_of_week', 'poutcome']

codif = pd.get_dummies(df[cols_not_ord], drop_first=True, dtype=int)

df_encoded = pd.concat([df.drop(columns=cols_not_ord), codif], axis=1)

print(df_encoded)


# Codificación ordinal para la variable "education"

# Revisamos los distintos valores
print(df_encoded["education"].value_counts())
# Definimos un orden 
df_encoded["education"] = df_encoded["education"].map({'illiterate':1, 'basic.4y':2, 'basic.6y':3, 'basic.9y':4, 'high.school':5, 'professional.course':6, 'university.degree':7})
# Revisamos que se haya hecho correctamente la codificación
print(df_encoded["education"].value_counts())


# Mapa de correlación entre nuestras variables
plt.figure(figsize=(10,8))
sns.heatmap(df_encoded.corr(), annot=False, cmap="coolwarm")
plt.show()


# Histograma de la variable objetivo
plt.figure(figsize=(8, 5))
sns.histplot(df_encoded['y'], kde=False, color="skyblue")
plt.xlabel("Variable Objetivo (y)")
plt.ylabel("Frecuencia")
plt.savefig("Histograma variable de respuesta.png", dpi = 500)



################
## Modelación ##
################

x_train, x_test, y_train, y_test = train_test_split(
    df_encoded.drop(columns=['y']), df_encoded['y'], test_size=0.3, stratify=df_encoded['y'], random_state=42
)


#=====================================
#= Naive Bayes =
#=====================================
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# Entrenar
nb = GaussianNB()
nb.fit(x_train, y_train)
# Evaluar
y_pred_nb = nb.predict(x_test)




#=====================================
#=LDA (Linear Discriminant Analysis)=
#=====================================
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Entrenar
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
# Evaluar
y_pred_lda = lda.predict(x_test)




#=====================================
# QDA (Quadratic Discriminant Analysis) =
#=====================================
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# Entrenar
qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
qda.fit(x_train, y_train)
# Evaluar
y_pred_qda = qda.predict(x_test)




#=====================================
# k-NN (k-Nearest Neighbors) =
#=====================================
from sklearn.neighbors import KNeighborsClassifier
# Entrenar (con k=5 vecinos)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
# Evaluar
y_pred_knn = knn.predict(x_test)



# Guardar resultados en un diccionario
results = {
    "Naive Bayes": {
        "Acc": accuracy_score(y_test, y_pred_nb),
        "Precision": precision_score(y_test, y_pred_nb),
        "Recall": recall_score(y_test, y_pred_nb),
        "F1": f1_score(y_test, y_pred_nb)
    },
    "LDA": {
        "Acc": accuracy_score(y_test, y_pred_lda),
        "Precision": precision_score(y_test, y_pred_lda),
        "Recall": recall_score(y_test, y_pred_lda),
        "F1": f1_score(y_test, y_pred_lda)
    },
    "QDA": {
        "Acc": accuracy_score(y_test, y_pred_qda),
        "Precision": precision_score(y_test, y_pred_qda),
        "Recall": recall_score(y_test, y_pred_qda),
        "F1": f1_score(y_test, y_pred_qda)
    },
    "k-NN (k=5)": {
        "Acc": accuracy_score(y_test, y_pred_knn),
        "Precision": precision_score(y_test, y_pred_knn),
        "Recall": recall_score(y_test, y_pred_knn),
        "F1": f1_score(y_test, y_pred_knn)
    }
}

# Convertir a DataFrame para visualización
df_results = pd.DataFrame(results).T
print("\n=== Comparación de modelos ===")
print(df_results.round(3))

#=====================================
# Validación cruzada comparativa
#=====================================

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score

models = {
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

scoring = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",       # sensibilidad 
    "F1": "f1",
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Validación Cruzada (5-fold) ===")
for name, model in models.items():
    results = cross_validate(
        model,
        df_encoded.drop(columns='y'),
        df_encoded['y'],
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )
    
    print(f"\n{name}")
    for metric in scoring.keys():
        scores = results[f"test_{metric}"]
        print(f"  {metric:12s}: {scores.mean():.3f} ± {scores.std():.3f}")



#=====================================
# Matrices de confusión
#=====================================


# Lista de modelos ya entrenados
trained_models = {
    "Naive Bayes": nb,
    "LDA": lda,
    "QDA": qda,
    "k-NN": knn
}

# Graficar matrices de confusión
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")

plt.tight_layout()
plt.show()
plt.savefig("Matrices de confusion1.png", dpi = 500)















#=====================================
######### Análisis quitando la columna default ############
#=====================================

# Leemos los datos
df = pd.read_csv('./data/data.csv', sep=';')

# Revisamos cuáles columnas tienen datos "unknown"
cols_with_unknown = [col for col in df.columns if df[col].isin(["unknown"]).any()]

# Reemplazamos los "unknown" por NaN
df[cols_with_unknown] = df[cols_with_unknown].replace("unknown", np.nan)
# Vemos cuántos datos faltantes hay en cada columna por porcentaje
print(df.isnull().mean() * 100)
# Imputamos los NaN en las columnas categóricas con los valores aleatorios según la distribución de cada columna.
for col in cols_with_unknown:
    df[col] = imputar_aleatorio(df[col])
# Verificamos que ya no hay datos vacíos
print(df.info())

# Eliminamos la columna default debido a que originalmente había solo 3 sí en más de 30 mil registros.
df_drop = df.drop(columns=['default'])
print(df_drop.columns)

# Ahora reemplazaremos los datos binarios por 0 y 1 (estos están actualmente como "no" y "yes").
cols_yes_no = [col for col in df_drop.columns if df_drop[col].isin(["yes", "no"]).any()]
print(cols_yes_no)

df_drop[cols_yes_no] = df_drop[cols_yes_no].replace({'no': 0, 'yes': 1}).astype(int)

# Codificación one-hot para variables categóricas no ordinales y no binarias.
cols_not_ord = ['job', 'contact', 'month', 'day_of_week', 'poutcome', 'marital']

codif = pd.get_dummies(df_drop[cols_not_ord], drop_first=True, dtype=int)

df_drop_encoded = pd.concat([df_drop.drop(columns=cols_not_ord), codif], axis=1)


# Codificación ordinal para la variable "education"

# Revisamos los distintos valores
print(df_drop_encoded["education"].value_counts())
# Definimos un orden 
df_drop_encoded["education"] = df_drop_encoded["education"].map({'illiterate':1, 'basic.4y':2, 'basic.6y':3, 'basic.9y':4, 'high.school':5, 'professional.course':6, 'university.degree':7})
# Revisamos que se haya hecho correctamente la codificación
print(df_drop_encoded["education"].value_counts())

# Mapa de correlación entre nuestras variables
plt.figure(figsize=(10,8))
sns.heatmap(df_drop_encoded.corr(), annot=False, cmap="coolwarm")
plt.show()

# Histograma de la variable objetivo
plt.figure(figsize=(8, 5))
sns.histplot(df_drop_encoded['y'], kde=False, color="skyblue")
plt.xlabel("Variable Objetivo (y)")
plt.ylabel("Frecuencia")




# Dataframe con columnas eliminadas
x_train, x_test, y_train, y_test = train_test_split(
    df_drop_encoded.drop(columns=['y']), df_drop_encoded['y'], test_size=0.3, stratify=df_drop_encoded['y'], random_state=42
)

#=====================================
#= Naive Bayes =
#=====================================
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Entrenar
nb = GaussianNB()
nb.fit(x_train, y_train)

y_pred_nb = nb.predict(x_test)




#=====================================
#=LDA (Linear Discriminant Analysis)=
#=====================================

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Entrenar
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)

# Evaluar
y_pred_lda = lda.predict(x_test)




#=====================================
# QDA (Quadratic Discriminant Analysis) =
#=====================================
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Entrenar
qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train, y_train)

# Evaluar
y_pred_qda = qda.predict(x_test)




#=====================================
# k-NN (k-Nearest Neighbors) =
#=====================================
from sklearn.neighbors import KNeighborsClassifier

# Entrenar (con k=5 vecinos)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# Evaluar
y_pred_knn = knn.predict(x_test)




# Guardar resultados en un diccionario
results = {
    "Naive Bayes": {
        "Acc": accuracy_score(y_test, y_pred_nb),
        "Precisión": precision_score(y_test, y_pred_nb),
        "Recall": recall_score(y_test, y_pred_nb),
        "F1": f1_score(y_test, y_pred_nb)
    },
    "LDA": {
        "Acc": accuracy_score(y_test, y_pred_lda),
        "Precisión": precision_score(y_test, y_pred_lda),
        "Recall": recall_score(y_test, y_pred_lda),
        "F1": f1_score(y_test, y_pred_lda)
    },
    "QDA": {
        "Acc": accuracy_score(y_test, y_pred_qda),
        "Precisión": precision_score(y_test, y_pred_qda),
        "Recall": recall_score(y_test, y_pred_qda),
        "F1": f1_score(y_test, y_pred_qda)
    },
    "k-NN (k=5)": {
        "Acc": accuracy_score(y_test, y_pred_knn),
        "Precisión": precision_score(y_test, y_pred_knn),
        "Recall": recall_score(y_test, y_pred_knn),
        "F1": f1_score(y_test, y_pred_knn)
    }
}

# Convertir a DataFrame para visualización
df_results = pd.DataFrame(results).T
print("\n=== Comparación de modelos ===")
print(df_results.round(3))

#=====================================
# Validación cruzada comparativa
#=====================================

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score

models = {
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5)
}


# Definimos la especificidad como el recall de la clase negativa (0)
specificity = make_scorer(recall_score, pos_label=0)

scoring = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",       # sensibilidad 
    "Specificity": specificity,
    "F1": "f1",
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Validación Cruzada (5-fold) ===")
for name, model in models.items():
    results = cross_validate(
        model,
        df_drop_encoded.drop(columns='y'),
        df_drop_encoded['y'],
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )
    
    print(f"\n{name}")
    for metric in scoring.keys():
        scores = results[f"test_{metric}"]
        print(f"  {metric:12s}: {scores.mean():.3f} ± {scores.std():.3f}")



#=====================================
# Matrices de confusión
#=====================================


# Lista de modelos ya entrenados
trained_models = {
    "Naive Bayes": nb,
    "LDA": lda,
    "QDA": qda,
    "k-NN": knn
}

# Graficar matrices de confusión
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")

plt.tight_layout()
plt.savefig("Matrices de confusion2.png", dpi = 500)
plt.show()
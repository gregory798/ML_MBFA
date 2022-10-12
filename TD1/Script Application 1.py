# ETAPE 0: IMPORTATION DES DONNEES 
# (IL FAUT CHANGER L EMPLACEMENT POUR QU IL CORRESPONDE A L ENDROIT OU SONT VOS DONNEES)
import pandas
X = pandas.read_excel("/Users/lecture9/data/data_X.xlsx")
y = pandas.read_excel("/Users/lecture9/data/data_y.xlsx")

# ETAPE 1: COMPRENDRE A QUOI RESSEMBLENT LES DONNEES
print(type(X))
print(X.head(10))
print(X.shape)
print(X.columns)
print(X.dtypes)
print(X.info())

X.head(10)
y.head(10)

X.shape
y.shape

# ETAPE 2: VISUALISER LES DONNEES
import matplotlib as mpl
import matplotlib.pyplot as plt
chiffre = X.iloc[1]
chiffre_image = chiffre.values.reshape(28, 28)
plt.imshow(chiffre_image,cmap=mpl.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()
y.iloc[1]

# ETAPE 3: DIVISER LA BASE EN UNE PARTIE TEST ET UNE PARTIE APPRENTISSAGE
from sklearn.model_selection import train_test_split
import numpy as np
y = y.astype(np.uint8).values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
X_train, X_test, y_train, y_test = X[:4000], X[4000:], y[:4000], y[4000:]

# ETAPE 3 BIS: CREATION DES VARIABLES DE PREDICTION
y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)

# ETAPE 4: ENTRAINER L ALGORITHME

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(loss='hinge',random_state=42)
sgd_clf.fit(X_train, y_train_0)
sgd_clf.predict([chiffre])


# ETAPE 5: MESURER LA PERFORMANCE DE L ALGORITHME EN UTILISANT LA CROSS VALIDATION

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy")

# ETAPE 6: COMPARAISON AVEC UN ALGORITHME NAIF

from sklearn.base import BaseEstimator
class Never0Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_0_clf = Never0Classifier()
cross_val_score(never_0_clf, X_train, y_train_0, cv=3, scoring="accuracy")

###############################################################################
# ETAPE 7: CREER LA MATRICE DE CONFUSION 
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_0, cv=3)

# CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# RAW NUMBERS
confusion_matrix(y_train_0, y_train_pred)

# PRECISION
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_0, y_train_pred) # == 4096 / (4096 + 1522)
# RECALL
recall_score(y_train_0, y_train_pred) # == 4096 / (4096 + 1325)

# RAPPORT DE CLASSIFICATION TOTAL
print(classification_report(y_train_0,y_train_pred))

###############################################################################

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(loss='log',random_state=42)
sgd_clf.fit(X_train, y_train_0)
sgd_clf.predict([chiffre])


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train_0)
        
          
# LOG
y_pred_log = sgd_clf.predict(X_test)

print(classification_report(y_test_0,y_pred_log))

# KNN
y_pred_neigh = neigh.predict(X_test)
print(classification_report(y_test_0,y_pred_neigh))


# 
error_rate = []
for i in range(1,40):
     neigh = KNeighborsClassifier(n_neighbors=i)
     neigh.fit(X_train,y_train_0)
     pred_i = neigh.predict(X_test)
     error_rate.append(np.mean(pred_i != y_test_0))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed',marker='o',markerfacecolor='red', markersize=10)
plt.title('Taux d erreur de classification vs. Nombre de voisins K')
plt.xlabel('K')
plt.ylabel('Taux d erreur de classification')
req_k_value = error_rate.index(min(error_rate))+1
print("Minimum error:-",min(error_rate),"at K =",req_k_value)

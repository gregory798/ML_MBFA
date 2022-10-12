# ETAPE 0: IMPORTEZ LES DONNEES
from sklearn.datasets import load_iris

iris = load_iris()
X_base = iris.data[:, 2:] # petal length and width
y_base = iris.target
X = X_base[:100]
y = y_base[:100]


# I. ANALYSE VISUELLE
##############################################################################
##############################################################################
# A - DEUX CLASSES PARFAITEMENT SEPARABLES
    # ETAPE 0: VISUALISEZ LES DONNEES
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.xlim(0, 6)
plt.ylim(0, 2)

    # ETAPE 1: LARGE MARGIN CLASSIFIER
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

svc = svm.SVC(kernel='linear', C=1).fit(X, y)

# création du cadre dans lequel on va représenter le graphique
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.xlim(0, 6)
plt.ylim(0, 2)
plt.show()

##############################################################################
# B - DEUX CLASSES NON PARFAITEMENT SEPARABLES
X = X_base[50:]
y = y_base[50:]
    # ETAPE 0: VISUALISEZ LES DONNEES
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Petal length")
plt.ylabel("Petal width")

    # ETAPE 1: SOFT MARGIN CLASSIFIER
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

svc1 = svm.SVC(kernel='linear', C=1).fit(X, y)
svc2 = svm.SVC(kernel='linear', C=10).fit(X, y)
svc3 = svm.SVC(kernel='linear', C=100).fit(X, y)
svc4 = svm.SVC(kernel='linear', C=1000).fit(X, y)

# création du cadre dans lequel on va représenter le graphique
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

titles = ['C = 1',
          'C = 10',
          'C = 100',
          'C = 1000']

for i, clf in enumerate((svc1, svc2, svc3, svc4)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
plt.show()

##############################################################################
# C - TROIS CLASSES
X = X_base
y = y_base

    # VISUALISATION DES DONNEES
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Petal length")
plt.ylabel("Petal width")

    # SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

svc = svm.SVC(kernel='linear', C=1).fit(X, y)

# création du cadre dans lequel on va représenter le graphique
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()

##############################################################################
X = X_base
y = y_base

    # SVC AVEC DIFFERENTS KERNELS
svc = svm.SVC(kernel='linear', C=1).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=1).fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

titles = ['SVC avec noyau linéaire',
          'SVC avec noyau RBF ',
          'SVC avec noyau polynomial (degré 3)']


for i, clf in enumerate((svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
plt.show()


##############################################################################
##############################################################################
##############################################################################
##############################################################################

# II. COMPARAISON DE LA PERFORMANCE
##############################################################################
##############################################################################
##############################################################################
##############################################################################
    # A. DIVISION DE LA BASE EN DEUX
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=6)
   
    # B. ENTRAINEMENT ET TEST DES MODELES
        # COMPARAISON SELON LA VALEUR DE C
list_C = [1,10,100,1000]
for i in list_C:
    svc = svm.SVC(kernel='linear', C=i).fit(X_train, y_train)
    print(f'% Classification Correct - Linéaire (Base Apprentissage) : {svc.score(X_train, y_train):.3f}')
    print(f'% Classification Correct - Linéaire (Base Test) : {svc.score(X_test, y_test):.3f}')
        
        # COMPARAISON SELON LE NOYAU
svc = svm.SVC(kernel='linear', C=10).fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=10).fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=10).fit(X_train, y_train)
print(f'% Classification Correct - Linéaire (Base Test) : {svc.score(X_test, y_test):.3f}')
print(f'% Classification Correct - RBF (Base Test) : {rbf_svc.score(X_test, y_test):.3f}')
print(f'% Classification Correct - Poly (Base Test) : {poly_svc.score(X_test, y_test):.3f}')


##############################################################################
##############################################################################
##############################################################################
##############################################################################

# III. MULTI-LAYER PERCEPTRON (Réseaux de neurones)
    # A. DIVISION DE LA BASE EN DEUX
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=6)
   
    # B. ENTRAINEMENT ET TEST DES MODELES
from sklearn.neural_network import MLPClassifier
n_couche = [1,10,20,50]
train_results = []
test_results = []
for i in n_couche:
   rf = MLPClassifier(hidden_layer_sizes=(i), solver = 'lbfgs', 
                      random_state=21,)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   test_r = rf.score(X_test,y_test)
   test_results.append(test_r)
   train_r = rf.score(X_train,y_train)
   train_results.append(train_r)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_couche, train_results, label='Erreur d apprentissage')
line2, = plt.plot(n_couche, test_results, label='Erreur de test')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Taux de Classification Correct')
plt.xlabel('Nombre de couches cachées')
plt.show()


##############################################################################
    # C. FONCTION D ACTIVATION

from sklearn.neural_network import MLPClassifier
f_activation = ['identity', 'logistic', 'tanh', 'relu']
train_results = []
test_results = []
for i in f_activation:
   rf = MLPClassifier(hidden_layer_sizes=(1), activation = i, solver = 'lbfgs', 
                      random_state=2)
   rf.fit(X_train, y_train)
   print(f'Test : {rf.score(X_test, y_test):.3f}')
   print(f'Train : {rf.score(X_train, y_train):.3f}')



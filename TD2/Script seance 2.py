# ETAPE 0: IMPORTEZ LES DONNEES
    # OPTION A (LA PLUS SIMPLE)
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

    # OPTION B (MOINS SIMPLE ET SEULEMENT SI A NE MARCHE PAS, ne pas oublier de remplacer le chemin d'accès)
import pandas
X = pandas.read_excel("/Users/data/data_X.xlsx")
y = pandas.read_excel("/Users/data/data_y.xlsx")
X = X.to_numpy()
y = y.to_numpy()

# ETAPE 0: DESCRIPTION DE LA BASE DE DONNEES
print(type(X))
print(X[:10])
print(X.shape)

# ETAPE 0-2: VISUALISEZ LES DONNEES
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Petal length")
plt.ylabel("Petal width")

# ETAPE 1: ENTRAINEZ UN ARBRE DE DECISION
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
predictions = tree_clf.predict(X)
predictions
print(f'% Classification Correct : {tree_clf.score(X, y):.3f}')


# ETAPE 2: VISUALISEZ L ARBRE DE DECISION
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=2)
clf.fit(X, y)
plot = tree.plot_tree(clf,feature_names=iris.feature_names[2:], class_names=iris.target_names, proportion = True, rounded=True,filled=True)

clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)
tree.plot_tree(clf,feature_names=iris.feature_names[2:], class_names=iris.target_names,proportion = True, rounded=True,filled=True)

# ETAPE 3 : PREDICTION SUR LA BASE DES CARACTERISTIQUES D UNE FLEUR
from sklearn import tree
tree_clf = tree.DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
tree_clf.predict_proba([[5, 1.5]])
tree_clf.predict([[5, 1.5]])


# ETAPE 4 : DIVISION DE LA BASE EN DEUX
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


from sklearn.datasets import make_classification
##############################################################################
# ETAPE 5 : PERFORMANCE BAGGING 
##############################################################################
from sklearn.ensemble import BaggingClassifier

clf = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(),n_estimators = 10, random_state=10)
clf.fit(X_train, y_train)
print(f'Test : {clf.score(X_test, y_test):.3f}')
print(f'Train : {clf.score(X_train, y_train):.3f}')

##############################################################################
# ETAPE 6 : PERFORMANCE RANDOM FOREST
##############################################################################
from sklearn.ensemble import RandomForestClassifier

RF_Model = RandomForestClassifier()
RF_Model = RandomForestClassifier(random_state=10,n_estimators = 10)
RF_Model.fit(X_train, y_train)
print(f'Test : {RF_Model.score(X_test, y_test):.3f}')
print(f'Train : {RF_Model.score(X_train, y_train):.3f}')


##############################################################################
# ETAPE 7 : PERFORMANCE BOOSTING
##############################################################################
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1,
     max_depth=1, random_state=0).fit(X_train, y_train)

clf.fit(X_train, y_train)
print(f'Test : {clf.score(X_test, y_test):.3f}')
print(f'Train : {clf.score(X_train, y_train):.3f}')


import matplotlib.pyplot as plt
##############################################################################
from sklearn.ensemble import RandomForestClassifier
##############################################################################
# EN FONCTION DU NOMBRE D ARBRES
n_arbre = [1,2,3,4,5]
train_results = []
test_results = []
for i in n_arbre:
   rf =   RF_Model = RandomForestClassifier(bootstrap=True,
                            n_estimators=i, 
                            random_state=100)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   test_r = rf.score(X_test,y_test)
   test_results.append(test_r)
   train_r = rf.score(X_train,y_train)
   train_results.append(train_r)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_arbre, train_results, label='Erreur d apprentissage')
line2, = plt.plot(n_arbre, test_results, label='Erreur de test')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Taux de Classification Correct')
plt.xlabel('Nombre d arbres')
fig1 = plt.gcf()
fig1.savefig('/Users/erreur_classification_n_arbre_rf.png')
plt.show()

# EN FONCTION DE LA PROFONDEUR D UN ARBRE
max_depths = [1, 2, 3, 4,5]
train_results = []
test_results = []
for i in max_depths:
   rf =   RF_Model = RandomForestClassifier(bootstrap=True, max_depth=i, random_state=100)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   test_r = rf.score(X_test,y_test)
   test_results.append(test_r)
   train_r = rf.score(X_train,y_train)
   train_results.append(train_r)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, label='Erreur d apprentissage')
line2, = plt.plot(max_depths, test_results, label='Erreur de test')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Taux de Classification Correct')
plt.xlabel('Profondeur d un arbre')
fig1 = plt.gcf()
fig1.savefig('/Users/erreur_classification_profondeur_arbre_rf.png')
plt.show()



##############################################################################
from sklearn.ensemble import GradientBoostingClassifier
##############################################################################
n_estimators_n = [1, 2, 5, 10, 20 ]
train_results = []
test_results = []
for i in n_estimators_n:
   rf =  GradientBoostingClassifier(n_estimators=i, learning_rate=0.1,
     max_depth=1, random_state=0).fit(X_train, y_train)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   test_r = rf.score(X_test,y_test)
   test_results.append(test_r)
   train_r = rf.score(X_train,y_train)
   train_results.append(train_r)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, label='Erreur d apprentissage')
line2, = plt.plot(max_depths, test_results, label='Erreur de test')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Taux de Classification Correct')
plt.xlabel('Nombre d arbres')
fig1 = plt.gcf()
fig1.savefig('/Users/erreur_classification_n_arbre_boosting.png')
plt.show()


##############################################################################
max_depths = [1, 2,3,4]
train_results = []
test_results = []
for i in max_depths:
   rf =  GradientBoostingClassifier(n_estimators=20, learning_rate=0.1,
     max_depth=i, random_state=0).fit(X_train, y_train)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   test_r = rf.score(X_test,y_test)
   test_results.append(test_r)
   train_r = rf.score(X_train,y_train)
   train_results.append(train_r)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, label='Erreur d apprentissage')
line2, = plt.plot(max_depths, test_results, label='Erreur de test')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Taux de Classification Correct')
plt.xlabel('Profondeur d un arbre')
fig1 = plt.gcf()
fig1.savefig('/Users/erreur_classification_profondeur_arbre_boosting.png')
plt.show()




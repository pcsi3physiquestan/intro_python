---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  formats: ipynb,md:myst
  split_at_heading: true
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Numpy : manipuler des tableaux de nombres.

Les listes classiques ont plusieurs limites quand on veut faire du calcul numérique. En effet, les opérations sur les listes classiques éléments par éléments (ajouter le même nombre à tous les éléments d'une liste, sommer deux à deux les éléments de deux listes...) nécessite de créer systématiquement des boucles.

C'est pourquoi on utilisera le module `numpy` qui propose un objet particulier : le __tableau numpy__ (ou _vecteur numpy_ qu'on il n'a qu'une dimension). Les opérations sur un tableau numpy sont __différentes des opérations sur une liste classiques__, il est important de différencier les deux pour ne pas se tromper.

## Importer la bibliothèque

(import-np)= 
```{code-cell}
import numpy as np
```

```{admonition} Question
:class: tip
Pour utiliser la fonction `array` de la bibliothèque `numpy`. On utilise alors `array(...)` ou `np.array(...)` ?
```

```{dropdown} Cliquez pour avoir la réponse.
On a importé numpy dans un espace nommé `np`, on utilisera donc la syntaxe `np.array(...)`.
```

## Créer un vecteur numpy

```{code-cell}
:tags: [remove-input, remove-output]
from myst_nb import glue

y = np.polyval([2, 3], [1, 4, 6, 7, 8, 24])
ty = type(y)
glue("ex_v", y)
glue("t_v", ty)

y2 = np.arange(2, 6, 0.5)
glue("ex_v2", y2)

y3 = np.linspace(2, 6, 9)
glue("ex_v3", y3)

```


````{tabbed} D'une liste
A partir d'une liste d'entiers ou de flottants (uniquement) :

```{code-block} ipython3
L = [1, 6, 2, 5, 3.4]  # Ne mettre QUE des nombres
Ln = np.array(L)  # Ln est un vecteur numpy

"""ou directement"""
Ln = np.array([1, 6, 2, 5, 3.4])  #
```
````

````{tabbed} D'une fonction
Plusieurs fonctions de la bibliothèque numpy renvoient un vecteur numpy.

```{code-block} ipython3
""" Exemple : La fonction suivante evalue le polynome 
p(x) = 2 + 3x (éléments du premièr argument [2, 3])
pour les valeurs de la liste [1, 4, 6, 7, 8, 24] (deuxième argument)
"""
y = np.polyval([2, 3], [1, 4, 6, 7, 8, 24])

```

`y` est un vecteur numpy contenant {glue:}`ex_v`

````

````{tabbed} arange()
A connaître : `arange(start, stop, step)` est une fonction de `numpy` similaire à `range` mais qui produit un tableau numpy. Elle peut aussi utiliser un pas non entier. _La valeur `stop` est exclus comme pour `range`_.

```{code-block} ipython3
y = np.arange(2, 6, 0.5)
```
`y` est un vecteur numpy contenant {glue:}`ex_v2`

````

````{tabbed} linspace()
A connaître : `linspace(start, stop, N)` est une fonction de `numpy` qui crée un vecteur de `N` éléments réparties uniformément entre les valeurs `start` et `stop` (cette fois `stop` est inclus, il faut la compter dans les N valeurs !).

```{code-block} ipython3
y = np.linspace(2, 6, 9)
```
`y` est un vecteur numpy contenant {glue:}`ex_v3`

````

````{tabbed} zeros()
A connaître : `zeros(N)` est une fonction de `numpy` qui crée un vecteur de `N` éléments nuls.  
A connaître : `ones(N)` est une fonction de `numpy` qui crée un vecteur de `N` éléments tous égaux à 1.

```{code-block} ipython3
y = np.zeros(5)  # Crée un vecteur [0, 0, 0, 0, 0]
y = np.ones(5)  # Crée un vecteur [1, 1, 1, 1, 1]
```
````

```{margin}
La fonction `type(y)` renverra {glue:}`t_v`.
```


## Opérations sur les vecteurs numpy

### Opérations générales
`numpy` permet de faire des opérations classiques : `+, -, /, *`
*  __termes à termes__ entre deux vecteurs
* __appliqué à chaque de la liste__ entre un vecteur et une valeur seule (un _scalaire_) (cf. exemple ci-dessous)

```{attention}
L'opérateur `+`  na pas la même fonction que pour des listes classiques.
```


```{code-cell}
:tags: [hide-output]
"""Essayer de prévoir ce que va afficher chaque print puis afficher le résultat (croix à droite en bas)"""

L1 = np.array([1, 2, 3, 4])
L2 = np.array([2, 3, 4, 5])

print(L1 + L2)
print(L1 * L2)
print(L1 - L2)
print(L1 / L2)

L3 = np.ones(5) * 4  # Méthode très utile
print(L3)

```

### Fonctions mathématiques usuelles.
`numpy` permet de _vectorialiser_ les fonctions usuelles (sin, cos, tan, exp, log...), c'est-à-dire de les appliquer sur chaque élément d'un vecteur numpy.

```{attention}
Attention : ces fonctions sont dans la bibliothèque `numpy`, pas la bibliothèque `math`. Si `numpy` est importé dans un espace nommé (`np` par exemple), il faut utiliser : `np.sin, np.cos, np.tan, np.exp, np.log,...`

```

```{code-cell}
:tags: [hide-output]
"""Essayer de prévoir ce que va afficher chaque print puis afficher le résultat (croix à droite en bas)
np.pi est une variable de la bibliothèque numpy égale à pi
"""

L1 = np.array([0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi])
L2 = np.linspace(0, 1, 5)  

print(L2)
print(np.sin(L1))
print(np.cos(L1))
print(np.tan(L1))
print(np.exp(L2))
print(np.log(L2))  # on écrit log mais il s'agit de la fonction ln classique.

```

### Concaténation, ajout d'une valeur

```{attention}
Les vecteurs `numpy` sont __de taille fixe__. On ne peut pas changer leur taille avec la méthode classique `append`.
```

Il existe par contre une fonction __dans le module `numpy`__ qui s'appelle `append` (il faut suivre...) qu'on appellera donc ici `np.append` (cf. l'[importation](import-np)). __Cette fonction ne modifie pas le vecteur initial mais crée un nouveau vecteur qu'il va falloir enregistrer dans une variable__.

```{code-cell}
L1 = np.array([1, 2, 3])

L2 = np.append(L1, 4)  # On ajoute la valeur 4
print(L1)  # L1 est toujours array([1, 2, 3])
print(L2)  # L2 est array([1, 2, 3, 4])

L3 = np.append(L1, L2)  # On peut aussi concatener deux vecteurs numpy.
print(L3)
```

### Sélection d'une partie d'un vecteur numpy
Le principe est le même que pour la sélection d'une partie d'une liste. Si on sélectionne un seul élément, ce sera un entier ou un flottant. Si on sélectionne plusieurs éléments, ce sera un vecteur numpy.

## Créer un tableau numpy
On peut créer un tableau `numpy` à 2 dimensions. C'est très utile pour traiter des données expérimentales. 

````{tabbed} D'une liste de liste
On utilise à nouveau `array`. Attention : les lignes doivent avoir toutes le même nombre d'éléments. De même pour les colonnes. Mais le tableau peut-être rectangulaire.

```{code-block} ipython3
L = [[2.3, 2.5], [3.14, 3.16]]  # Ne mettre QUE des nombres
Ln = np.array(L)  # Ln est un tableau numpy

"""ou directement"""
Ln = np.array([[2.3, 2.5], [3.14, 3.16]])  # Note : on entre le tableau ligne par ligne
```
````

````{tabbed} D'une fonction
Même principe que précédemment, certaines fonctions renverront un tableau numpy.

````

````{tabbed} zeros() et ones()
A connaître : `zeros((N, m))` est une fonction de `numpy` qui crée un tableau de `N` éléments par `m` éléments nuls.
A connaître : `ones((N, m))` est une fonction de `numpy` qui crée un tableau de `N` éléments par `m` éléments tous égaux à 1.

```{margin}
La double parenthèse est importante : il y a un seul argument (5, 3) (on parle de _tuple_) et non deux argument 5 et 3.
```


```{code-block} ipython3
y = np.zeros((5, 3))  # Crée un tableau de 0 contenant 5 lignes et 3 colonnes
y2 = np.ones((5, 3))  # Crée un tableau de 1 contenant 5 lignes et 3 colonnes
```
````

## Manipuler un tableau numpy.

### Opérations usuelles et concaténation
* Les opérations usuelles et fonctions mathématiques présentées précédemment s'appliquent à nouveau pour les tableaux numpy.
* Concaténation : La fonction `append` doit prendre un troisième argument si on concatène deux tableaux (ou un tableau et un vecteur). L'argument `axis = `:
    * `axis = 0` : Concaténer _verticalement_ : les deux tableaux doivent donc avec le même nombre de colonnes. Sinon Python renverra une erreur.
    * `axis = 1` : Concaténer _horizontalement_ : les deux tableaux doivent donc avec le même nombre de lignes. Sinon Python renverra une erreur.
    * Pas de `axis` : Python va transformer les tableaux en vecteurs et concaténer les vecteurs.

```{code-cell}
L1 = np.array([[2.3, 2.5], [3.14, 3.16], [3.14, 4.17]])
L2 = np.array([[1.2, 3.7], [4.2, 3.1]])

L3 = np.append(L1, L2)  # On aplatit les tableaux pour obtenir un vecteur.
print(L3)

L4 = np.append(L1, L2, axis=0)  # On place L2 sous L1
print(L4)

L5 = np.append(L1, L2, axis=1)  # Renvoie un erreur car on ne peut placer L2 à droite de L1 : pas le même nombre de lignes
print(L5)
```

### Sélection d'une partie d'un tableau.
```{attention}
* La méthode présentée ne fonctionne pas avec une liste de listes classique. Uniquement avec des tableaux `numpy`
* Attention, l'__indexation commence toujours à 0__
```

#### Sélection d'un élément.

```{code-cell}

L1 = np.array([[2.3, 2.5], [3.14, 3.16], [3.14, 4.17]])
print(L1[2, 1])
"""
2 : sélection de la troisième ligne
1 : sélection de la deuxième colonne

On affiche donc 4.17
"""
```

#### Sélection d'une portion d'un tableau
```{margin}
On rappelle que L1[1:3] sélectionne jusqu'à l'index 3 (quatrième élément ) __non inclus__.
```

```{code-cell}

L1 = np.array([[2.3, 2.5, 6.8], [3.14, 3.16, -1.3], [3.14, 4.17, 3.45], [1.6, 4.2,  -8.2]])
print("Première sélection", L1[0:2, 1:3])  # Sélection des deux premières lignes (0 et 1) et colonnes d'index (1 et 2). C'est un tableau.
print("Deuxième sélection", L1[:, 1])  # Permet de sélectionner la deuxième colonne en entier (c'est un vecteur)
print("Troisième sélection", L1[1, :])  # Permet de sélectionner la deuxième ligne en entier (c'est un vecteur)
print("Quatrième sélection", L1[1])  # Permet de sélectionner la deuxième ligne en entier aussi (c'est un vecteur)
print("Cinquième sélection", L1[-1, :])  # Permet de sélectionner la dernière ligne en entier (c'est un vecteur)
```

## Si j'ai bien compris...
```{admonition} Exercice
:class: tip
Essayer de prévoir ce qu'affichera chaque `print` puis observer le résultat en cliquant sur la croix en bas à droite.
```

```{code-cell}
:tags: [hide-output]
L1 = np.arange(0, 4)
L2 = L1 * L1
L3 = np.ones(4)

"""
Ci-dessous : une méthode pour créer un tableau à partir de 3 vecteurs de même taille.
Les L1, L2, L3 sont les lignes du tableau.
"""
M1 = np.array([L1, L2, L3])

print("Cas 1 :", M1)
print("Cas 2 :", M1[2, 3])
print("Cas 3 :", M1[-1, 2])
print("Cas 4 :", M1[1])
print("Cas 5 :", M1[2, :])
print("Cas 6 :", M1[:, -1])
print("Cas 7 :", M1[:, -2])
print("Cas 8 :", M1[1:2, 0:1])
print("Cas 9 :", M1[:-1, 1])  # Un peu plus compliqué...
print("Cas 10 :", M1[-3:-1, -3:3])  # On se concentre bien...
```

```{code-cell}
:tags: [remove-input, remove-output]
L1 = np.arange(0, 4)
L2 = L1 * L1
L3 = np.ones(4)

"""
Ci-dessous : une méthode pour créer un tableau à partir de 3 vecteurs de même taille.
Les L1, L2, L3 sont les lignes du tableau.
"""
M1 = np.array([L1, L2, L3])

glue("cas_1", M1)
glue("cas_2", M1[2, 3])
glue("cas_3", M1[-1, 2])
glue("cas_4", M1[1])
glue("cas_5", M1[2, :])
glue("cas_6", M1[:, -1])
glue("cas_7", M1[:, -2])
glue("cas_8", M1[1:2, 0:1])
glue("cas_9", M1[:-1, 1])  # Un peu plus compliqué...
glue("cas_10", M1[-3:-1, -3:3])  # On se concentre bien...
```

```{dropdown} Explication des réponses
__Les vecteurs__ :
* L1 crée une liste d'entier partant de 0 jusqu'à 4 __exclus__ soit `[0, 1, 2, 3]`
* L2 est la multiplication terme à terme L1 par lui-même, on passe chaque élément au carré soit `[0, 1, 4, 9]`
* L3 est une liste de 4 éléments composées uniquement de 1.

__Le tableau__ :
* M est le tableau composé des trois vecteurs. Il est donc de taille 3*4 (3 lignes, 4 colonnes).

__Les affichage__:
* Cas 1 : Affichage du tableau complet
* Cas 2 : Affichage de l'élément de la troisième ligne, quatrième colonne, soit {glue:}`cas_2`
* Cas 3 : Afficahge de l'élément de la dernière ligne, troisième colonne soit {glue:}`cas_3`
* Cas 4 : Affichage de la __deuxième ligne__ soit {glue:}`cas_4`
* Cas 5 : Affichage de la troisième ligne soit {glue:}`cas_5`
* Cas 6 : Affichage de la dernière colonne soit {glue:}`cas_6`
* Cas 7 : Affichage de l'avant-dernière colonne soit {glue:}`cas_7`
* Cas 8 : Affichage du tableau limité à la deuxième ligne (de l'indice 1 à l'indice 2 __exclus__) et à la première colonne soit {glue:}`cas_8` (de l'indice 0 à l'indice 1 __exclus__). _C'est bien un tableau mais avec une seule valeur !_
* Cas 9 : Affichage de la deuxième colonne mais sans la dernière ligne (on va jusqu'au -1 __exclus__) {glue:}`cas_9`
* Cas 10 : Affichage du tableau limité aux deux premières lignes (de l'indice -3 soit indice 0 - puisqu'il y a 3 lignes - jusqu'à l'indice -1 (dernier) __exclus__) et à la deuxième et troisième (de l'indice - 3 soit la colonne d'indice 1 - puisque qu'il y a 4 colonnes - jusqu'à l'indice 3 soit la dernière colonne __exclus__) soit {glue:}`cas_10`
```

## Fonctions de la bibliothèque
`numpy` embarque de nombreuses fonctions très utiles qui seront présentées durant l'année. Vous apprendrez ainsi la syntaxe associée à ces fonctions au fur et à mesure. Citons à titre d'exemple :
* `polyfit` :  Permet l'ajustement d'un modèle (linéaire en général)
* Sous-module `random` : contient plusieurs fonctions permettant des tirages aléatoires (`uniform`, `normal`) : utile pour les simulations de Monte-Carlo en physique-chimie.



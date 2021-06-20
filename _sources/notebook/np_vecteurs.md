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

C'est pourquoi on utilisera le module `numpy` qui propose un objet particulier : le __tableau numpy__ (ou _vecteur numpy_ quand il n'a qu'une dimension). Les opérations sur un tableau numpy sont __différentes des opérations sur une liste classiques__, il est important de différencier les deux pour ne pas se tromper.

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
V1 = np.array(L)  # V1 est un vecteur numpy

"""ou directement"""
V1 = np.array([1, 6, 2, 5, 3.4])  #
```
````


````{tabbed} arange
_A connaître_ : `arange(start, stop, step)` est une fonction de `numpy` qui va créer une liste de _flottants_ partant de `start` (inclus) et allant jusqu'à `stop` (__exclus__) par pas de valeur `step`.

`start`, `stop` et `step` peuvent être des flottants contrairement à `range`.

```{code-block} ipython3
y = np.arange(2, 6, 0.5)
```
`y` est un vecteur numpy contenant {glue:}`ex_v2`

````

````{tabbed} linspace
_A connaître_ : `linspace(start, stop, N)` est une fonction de `numpy` qui crée un vecteur de `N` éléments réparties uniformément entre les valeurs `start` et `stop` (cette fois `stop` est inclus, il faut la compter dans les N valeurs !).

```{code-block} ipython3
y = np.linspace(2, 6, 9)
```
`y` est un vecteur numpy contenant {glue:}`ex_v3`

````

````{tabbed} zeros et ones
_A connaître_ : `zeros(N)` est une fonction de `numpy` qui crée un vecteur de `N` éléments nuls.  
_A connaître_ : `ones(N)` est une fonction de `numpy` qui crée un vecteur de `N` éléments tous égaux à 1.

```{code-block} ipython3
y = np.zeros(5)  # Crée un vecteur [0, 0, 0, 0, 0]
y = np.ones(5)  # Crée un vecteur [1, 1, 1, 1, 1]
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

```{margin}
La fonction `type(y)` renverra {glue:}`t_v`.
```


## Opérations sur les vecteurs numpy

### Opérations générales
`numpy` permet de faire des opérations classiques : `+, -, /, *`
*  __termes à termes__ entre deux vecteurs (_Il faut que les deux vecteurs aient la même taille_)
* __appliqué à chaque élément d'un vecteur__ entre un vecteur et une valeur seule (un _scalaire_) (cf. exemple ci-dessous)


```{attention}
L'opérateur `+`  na pas la même fonction que pour des listes classiques.
```


```{code-cell}
:tags: [hide-output]
"""Essayer de prévoir ce que va afficher chaque print puis afficher le résultat (croix à droite en bas)"""

V1 = np.array([1, 2, 3, 4])
V2 = np.array([2, 3, 4, 5])

print(V1 + V2)
print(V1 * V2)
print(V1 - V2)
print(V1 / V2)

V3 = np.ones(5) * 4  # Méthode très utile
print(V3)

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

V1 = np.array([0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi])
V2 = np.linspace(0, 1, 5)  

print(V2)
print(np.sin(V1))
print(np.cos(V1))
print(np.tan(V1))
print(np.exp(V2))
print(np.log(V2))  # on écrit log mais il s'agit de la fonction ln classique.

```

### Concaténation, ajout d'une valeur

```{attention}
Les vecteurs `numpy` sont __de taille fixe__. On ne peut pas changer leur taille avec la méthode classique `append`.
```

Il existe par contre une fonction __dans le module `numpy`__ qui s'appelle `append` (il faut suivre...) qu'on appellera donc ici `np.append` (cf. l'[importation](import-np)). __Cette fonction ne modifie pas le vecteur initial mais crée un nouveau vecteur qu'il va falloir enregistrer dans une variable__.

```{code-cell}
V1 = np.array([1, 2, 3])

V2 = np.append(V1, 4)  # On ajoute la valeur 4
print(V1)  # V1 est toujours array([1, 2, 3])
print(V2)  # V2 est array([1, 2, 3, 4])

V3 = np.append(V1, V2)  # On peut aussi concatener deux vecteurs numpy.
print(V3)
```

### Sélection d'une partie d'un vecteur numpy
Le principe est le même que pour la sélection d'une partie d'une liste. Si on sélectionne un seul élément, ce sera un entier ou un flottant. Si on sélectionne plusieurs éléments, ce sera un vecteur numpy.

## Créer un tableau numpy
On peut créer un tableau `numpy` à 2 dimensions. C'est très utile pour traiter des données expérimentales. 

````{tabbed} D'une liste de liste
On utilise à nouveau `array`. Attention : Le tableau doit être rectangulaire (ou carré) : les lignes doivent avoir toutes le même nombre d'éléments.

```{code-block} ipython3
L = [[2.3, 2.5], [3.14, 3.16]]  # Ne mettre QUE des nombres
V1 = np.array(L)  # V1 est un tableau numpy

"""ou directement"""
V1 = np.array([[2.3, 2.5], [3.14, 3.16]])  # Note : on entre le tableau ligne par ligne
```
````

````{tabbed} zeros() et ones()
A connaître : `zeros((N, m))` est une fonction de `numpy` qui crée un tableau de `N` éléments par `m` éléments nuls.
A connaître : `ones((N, m))` est une fonction de `numpy` qui crée un tableau de `N` éléments par `m` éléments tous égaux à 1.

```{margin}
La double parenthèse est importante : il y a un seul argument (5, 3) (on parle de _tuple_) et non deux arguments 5 et 3.
```


```{code-block} ipython3
y = np.zeros((5, 3))  # Crée un tableau de 0 contenant 5 lignes et 3 colonnes
y2 = np.ones((5, 3))  # Crée un tableau de 1 contenant 5 lignes et 3 colonnes
```
````

## Manipuler un tableau numpy.

La manipulation des tableaux numpy est plus délicate que les vecteurs. Une [présentation](tableau_numpy) est proposée pour ceux qui se sentent à l'aise avec le concept de vecteurs.

On pourra déjà retenir que les opérations terme à terme sont réalisable aussi avec deux tableaux de même taille.

## Fonctions de la bibliothèque
`numpy` embarque de nombreuses fonctions très utiles qui seront présentées durant l'année. Vous apprendrez ainsi la syntaxe associée à ces fonctions au fur et à mesure. Citons à titre d'exemple :
* `polyfit` :  Permet l'ajustement d'un modèle (linéaire en général)
* Sous-module `random` : contient plusieurs fonctions permettant des tirages aléatoires (`uniform`, `normal`) : utile pour les simulations de Monte-Carlo en physique-chimie.



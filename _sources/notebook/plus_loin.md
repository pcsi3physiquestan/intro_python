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

Certains points abordés ici serviront très peu ou seront abordés pendant l'année. Il s'agit d'approfondissement pour ceux qui ont déjà bien compris les parties précédentes.

```{code-cell}
"""On importe les bibliothèques scientifiques car elles seront utilisées ensuite"""
import numpy as np
import matplotlib.pyplot as plt

```

```{tip}
L'ordre des parties va du plus important au moins important.

```


(tableau_numpy)=
# Tableaux numpy : manipulations

## Opérations usuelles et concaténation
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

## Sélection d'une partie d'un tableau.
```{attention}
* La méthode présentée ne fonctionne pas avec une liste de listes classique. Uniquement avec des tableaux `numpy`
* Attention, l'__indexation commence toujours à 0__
```

### Sélection d'un élément.

```{code-cell}

L1 = np.array([[2.3, 2.5], [3.14, 3.16], [3.14, 4.17]])
print(L1[2, 1])
"""
2 : sélection de la troisième ligne
1 : sélection de la deuxième colonne

On affiche donc 4.17
"""
```

### Sélection d'une portion d'un tableau
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

Voici le tableau `L1` avec les indices des lignes et colonnes pour mieux comprendre.

![Tableau numpy 2](./images/tableau_2.png)

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
from myst_nb import glue

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
* M est le tableau composé des trois vecteurs. Il est donc de taille 3*4 (3 lignes, 4 colonnes). Soit :

![Tableau numpy](./images/matrice_numpy.png)

__Les affichage__:
* Cas 1 : Affichage du tableau complet
* Cas 2 : Affichage de l'élément de la troisième ligne (indice 2), quatrième colonne (indice 3), soit {glue:}`cas_2`
* Cas 3 : Afficahge de l'élément de la dernière ligne (indice -1), troisième colonne (indice 2) soit {glue:}`cas_3`
* Cas 4 : Affichage de la __deuxième ligne (indice 1)__ soit {glue:}`cas_4`
* Cas 5 : Affichage de la troisième ligne (indice 2) soit {glue:}`cas_5`
* Cas 6 : Affichage de la dernière colonne (indice -1) soit {glue:}`cas_6`
* Cas 7 : Affichage de l'avant-dernière colonne (indice -2) soit {glue:}`cas_7`
* Cas 8 : Affichage du tableau limité à la deuxième ligne (de l'indice 1 à l'indice 2 __exclus__) et à la première colonne soit {glue:}`cas_8` (de l'indice 0 à l'indice 1 __exclus__). _C'est bien un tableau mais avec une seule valeur !_
* Cas 9 : Affichage de la deuxième colonne mais sans la dernière ligne (on va jusqu'au -1 __exclus__) {glue:}`cas_9`
* Cas 10 : Affichage du tableau limité aux deux premières lignes (de l'indice -3 soit indice 0 - puisqu'il y a 3 lignes - jusqu'à l'indice -1 (dernier) __exclus__) et à la deuxième et troisième colonne (de l'indice - 3 soit la colonne d'indice 1 - puisque qu'il y a 4 colonnes - jusqu'à l'indice 3 soit la dernière colonne __exclus__) soit {glue:}`cas_10`
```

+++


# La compréhension des listes

+++

## Position du problème
On a vu que les listes classiques ne permettaient pas d'appliquer une fonction `f` à chaque élément en écrivant `f(L)` (`L` étant une liste classique). Cela a motivé l'utilisation des listes numpy. Néanmoins :
* On est quelques fois obligé d'utiliser les listes classiques
* La syntaxe `f(L)` ne fonctionne que si `f` est _vectorialisable_, c'est-à-dire programmée pour pouvoir s'appliquer à chaque élément.

```{admonition} Fonctions et opérateurs vectorialisables
:class: tip
Non exhaustif :
* Les opérateurs classiues `+,-,/,*` sont vectorialisables
* Les fonctions de la bibliothèques numpy (sin, cos, exp...) sont vectorialisables.
```

Un exemple fréquent est quand la fonction `f` fait appelle à un structure conditionnelle qui n'est pas vectorialisable. On reprend la fonction définie par morceau :

```{code-cell}
def fonction_morceau(x):
  if x < 3:  # Si x < 3
    a = x - 4
  elif x < 5:  # Si 3 <= x < 5
    a = x ** 2
  else:  # Si x >=5
    b = x + 4
    a = b ** 3
  return a
```

Impossible de l'appliquer directement à un vecteur numpy (pour la tracer par exemple:

```{code-cell}
x = np.linspace(0, 15, 1000)  # Vecteur de 1000 valeurs entre 0 et 15
y = fonction_morceau(x)

```

`The truth value of an array with more than one element is ambiguous.` : Le message est clair : il cherche à travailler sur le vecteur et non sur chaque élément.

+++

## Première solution
On peut écrire une boucle classique qui parcout tous les éléments de `x` et leur applique `fonction_morceau` :

```{code-cell}
x = np.linspace(0, 15, 1000)
y = np.zeros(len(x))  # On crée un vecteur nul qu'on va remplir

for i in range(len(x)):  # i prendra les indices de 0 à len(x) - 1
  y[i] = fonction_morceau(x[i])  # On remplit l'élément d'indice i


"""On peut même le tracer"""
f, ax = plt.subplots()  # On ne va pas légender le graphique puisqu'il n'y a pas de contexte ici.
ax.plot(x, y)

plt.show()

```

+++

## Utilisation de la compréhension des listes.
Une __liste en compréhension__ est une liste qui est obtenue par action (et/ou filtrage) de chaque élément d'une autre liste. La syntaxe est simple :

```python
L1 = [fonction(x) for x in L]  # L est une liste et L1 la nouvelle liste en compréhension
```

La syntaxe est assez clair : "Applique la `fonction` à `x` pour `x` qui parcourt chaque élément de `L`"

```{margin}
Ca fonction si `L` est un vecteur `numpy`

```

````{attention}
Cette méthode renvoie une liste __classique__, pas un vecteur `numpy`. Si vous voulez obtenir un vecteur numpy, il faudra utiliser la fonction `array` :
```python
L1 = np.array([fonction(x) for x in L])
```
````

A titre d'exemple :

```{code-cell}
x = np.linspace(0, 15, 1000)
y = np.array([fonction_morceau(i) for i in x])


"""On peut même le tracer"""
f, ax = plt.subplots()  # On ne va pas légender le graphique puisqu'il n'y a pas de contexte ici.
ax.plot(x, y)

plt.show()

```

```{margin}
On peut aussi filtrer la liste pour n'appliquer `fonction(x)` que sous certaines conditions :

`[fonction(x) for x in L if x > 1]`

```

+++

# Les dictionnaires
Les __dictionnaires__ sont des objets python un peu plus complexes que vous pourriez rencontrer comme retour d'une fonction native. Un dictionnaire, comme une liste possède un ensemble d'éléments (des entiers, flottants, chaine de caractère, listes, vecteur numpy ...) mais ces éléments sont rangés dans des _champs nommés_. Un exemple avec les différentes manipulations utiles :

```{code-cell}
""" On définit un dictionnaire. 
- Il a 4 champs : fruit, couleur, nombre, prix.
- On a assigné à ces champs les valeurs respectives : "Pomme", "Rouge", 234, 3.45

"""
a = {'fruit': "Pomme", 'couleur': "Rouge", 'nombre': 234, 'prix': 3.45}

b = a['fruit']  # Pour accéder au contenu du champ "fruit" du dictionnaire a

print('Nom des champs :')
for truc in a:  # truc prendra les différents nom des champs
  print(truc)  # On affiche le nom des champs

print('Contenu des champs :')
for truc in a:  # truc prendra les différents nom des champs
  print(a[truc])  # On affiche le contenu de chaque champ

```

```{tip}
Attention:
* Si `fruit` est le nom d'un champ, on écrit `a["fruit"]` avec des `""` : c'est bien une chaine de caractère
* Si `truc` est le nombre de la variable qui contient la chaine de caractère `"fruit"`, on écrit `a[truc]`
```

+++

# Fonctions et méthodes
Vous avez peut-être remarqué que la façon d'appeler les "fonctions" pouvait un peu différer suivant les cas :
* `print(f)` : La __fonction__ `print` agit sur _l'argument_ `f` en l'affichant.
* `f.suptitle()` : la __méthode__  `suptitle` agit sur _l'objet_ `f` en le modifiant

On ne revient pas sur les fonctions qui ont déjà été présentées précédemment.

Les __méthodes__ sont comme des fonctions mais qui sont étroitement associées à certains types de variables (on parle d'objets). Cette notion est reliée à la __Programmation Orientée Objet__ (POO) rendue possibles par de nombreux langages informatiques (comme Python). 

Sans rentrer dans les détails, on peut en POO définir des _objets_ qui vont posséder certaines caractéristiques (des _attributs_) et certaines fonctions propres (des _méthodes_) qui permettent de les modifier ou d'obtenir certaines caractéristiques.

Par exemple, si je crée un vecteur numpy `v0` : c'est un objet. 
* Il possède certaines attributs. Ex : `shape` donnera sa taille. Pour obtenir cet attribut, on écrira `v0.shape` (l'attribut `shape` de l'objet `v0`)
* Il possède des méthodes. Ex : `fill(valeur)` remplit le vecteur avec la même `valeur`. Pour appliquer cette méthode à `v0`, on écrira `v0.fill(3)`.

Il n'est pas nécessaire de maîtriser complètement la POO. Il suffira surtout de repérer quand on utilise la syntaxe d'une fonction (ex : `round(3.4)`) et qu'on il s'agit de la syntaxe d'une méthode (ex : `ax.set_xlabel('Légende des x')`).





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

Il n'est pas nécessaire de maîtriser complètement la POO. Il suffira surtout de repérer quand on utilise la syntaxe d'une fonction (ex : `round(3.4)`) et quand il s'agit de la syntaxe d'une méthode (ex : `ax.set_xlabel('Légende des x')`).





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


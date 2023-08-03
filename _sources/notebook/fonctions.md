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

# Les fonctions.
Une fonction est une série d'instructions préenregistrées qui prend en entrée une ou plusieurs données (on parle __d'arguments__) et qui après exécution des instructions peut renvoyer une ou plusieurs données (les __retours__).

## Exécuter une fonction
Syntaxe : `nomfonction(argument1, argument2)`

```{margin}
Pensez à assigner ce que renvoie la fonction dans une variable pour une utilisation ultérieure.
```

## Signature d'une fonction
La signature d'une fonction définit les entrées et les sorties d'une fonction. Un signature définit ainsi :
* le nombre d'arguments en entrée et le type attendus de chaque argument.
* les valeurs en sortie et le type de chaque retour.

La signature est en général donnée dans l'aide de la fonction pour savoir comment l'utiliser.

````{sidebar} Types usuelles
* `float` : flotant
* `int` : entier
* `string` : chaine de caractère
* `list` : liste
* `dict` : dictionnaire
* `callable` : fonction
* `numpy.ndarray` : vecteur numpy (cf. suite)
````
````{important}
On trouvera souvent par la suite la syntaxe suivante pour préciser le type des arguments et du retour de la fonction :

`nomfonction(argument1:type1, argument2:type2) -> typeretour`

Ainsi la fonction:

`associe(a:float, b:float) -> list`

prend comme arguments deux flottants `a`et `b` et renvoie une liste.
````


## Les fonctions existantes

### La bibliothèque standard
Il existe de nombreuses fonctions natives dans Python qui peuvent être utilisées directement (on parle de _bibliothèque standard_). On a déjà présenté quelques fonctions utiles ainsi que les types auxquels elles s'appliquaient.

```{code-cell}
"""Exemple d'utilisation d'une fonction native"""
a = min(3, 5, 6, 12)  # Fonction qui renvoie le minimum de l'ensemble des arguments données.
print(a)
```

Quelques fonctions utiles (non exhaustif):
* `print(a)` : Affiche la variable
* `len(l)` : Longueur d'une liste ou d'une chaine de caractères
* `round(nombre, digits)` : Arrondi le flottant au nombre de digits après la virgule.
* `min(a, b, ...)` et `max(a, b, ...)`
* `int(b)` : Transforme en entier (utile pour transformer une chaine de caractère `"1"` en entier `1`)
* `float(b)` : Transforme en flottant (utile pour transformer une chaine de caractère `"1.23"` en flottant `1.23`)
* `str(a)` : Transforme (un nombre) en une chaine de caractère. Utile pour afficher un nombre dans une phrase.
* `type(a)` : Renvoie le type de la variable.
* `help(nom_fonction)` : Affiche l'aide d'une fonction.

### Les modules supplémentaires
Des fonctions supplémentaires (entre autre) sont contenus dans des modules (ou bibliothèques) qui nécessitent d'être importés. Les méthodes d'importations sont :

`````{tab-set}
````{tab-item} Importation dans l'espace standard
```{code-block} ipython3
from math import *
```

On importe toutes les fonctions du modules dans l'espace standard. La fonction `sin` du module `math` est appelé par :
```{code-block} ipython3
sin(3.14)
```
````

````{tab-item} Importation dans un espace nommé
```{code-block} ipython3
import math as mt
```

On importe toutes les fonctions du modules dans un espace nommé (ici `mt`). La fonction `sin` du module `math` est appelé par :
```{code-block} ipython3
mt.sin(3.14)
```
````
`````

```{important}
Si on vous donne un programme déjà partiellement écrit, pensez à regarder comment le module a été importé pour utiliser correctement les fonctions du module.
```

## Créer sa fonction

### Un exemple

```{code-cell}
def somme3(arg1, arg2, arg3):
	"""
	Ici le corps de la fonction, par exemple on ajoute les trois arguments
	et on assigne le résultat à la variable total
	"""
	total = arg1 + arg2 + arg3
	# Ne pas oublier de renvoyer la valeur
	return total
```

### Structure d'une fonction
Les instructions ci-dessous sont le minimum dans une fonction :
* le mot-clé `def` est indispensable et le `:` à la fin aussi.
* __L'indentation (décalage du corps de la fonction vers la droite) est obligatoire en Python.__
* choisir un nom de fonction (ici `somme3`) explicite (les règles de nommage sont les mêmes que pour les variables)
* Le nom des arguments `arg1, ...` sera celui utilisé _dans_ le bloc d'instruction de la fonction. `arg1` n'existe pas en dehors de la fonction.
* pensez au `return` sans quoi votre fonction ne renverra rien... Ici, on pourra accéder contenu de la variable `total` en dehors de la fonction.

```{margin}
Il peut arriver qu'une fonction ne prennent aucun argument ou ne renvoie rien. Mais c'est prévu par le concepteur de la fonction.
```


### Pensez à commenter votre fonction pour qu'elle soit compréhensible.
Tout est dans le titre...

### Créer n'est pas appeler.
La cellule précédente a créé la fonction mais n'a pas _utilisé la fonction_ (on dit __appeler la fonction__). Une fois créée, il faut encore l'appeler pour qu'elle soit exécuter dans un programme. L'appel de la fonction se fait comme pour une fonction native.

```{code-cell}
""" On appelle la fonction qu'on vient de créer précédemment"""
toti = somme3(1, 2, 3)  # On appelle la fonction et on stocke son retour dans la variable toti
print(toti)
```

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

# Instructions itératives.
Les boucles permettent de réaliser plusieurs fois la même série d'instructions. On distingue deux types de boucles :
* les boucles bornées (`for`) : on sait quand la boucle va s'arrêter.
* les boucles non bornées (`while`) : on impose une condition d'arrêt de la boucle sans savoir quand elle sera réalisée.

## Boucle for

### Un exemple 

(exemple-for)=
```{code-cell}
L = [1, 3, 5, 7, 9]

for n in L:
	print(n**2)
```

### Syntaxe
* Début de la boucle : `for variable_d_iteration in objet_iterable:` : le `:` est obligatoire
* Eviter d'utiliser comme `variable_d_iteration` une variable qui existe déjà.
* Un `objet_iterable` est un objet qui contient plusieurs éléments et qui peut être parcouru (itéré). La `variable_d_iteration` prendre comme valeur à chaque tour de boucle les différents éléments. Pour nous les `objet_iterable` sont :
    * les listes
    * les vecteurs numpy (cf. suite)
    * des objets créés pour la boucle comme `range(5)` qui va créer une suire de nombres (0, 1, 2, 3, 4)
* Le corps de la boucle (série d'instruction à exécuter en boucle) doit __être indenté__.

### Exemple
* A partir d'une liste déjà créée : cf. l'[exemple précédent](exemple-for).
* On utilise `range(n)` :

```{code-cell}
for n in range(4):  # On aura donc n=0, n=1, n=2, n=3 et on sort de la boucle
	print(n**2)
```

```{important}
La "liste" créée par `range(n)` commence à 0 et s'arrête à __n-1__
```

### Aller plus loin avec range
La syntaxe précédente `range(n)` suffit en général. Mais il existe une autre syntaxe plus complète : `range(start, stop, step)` qui créé une suite de nombre partant de `start` inclus jusqu'à `stop` __non inclus__ par pas de `step`

````{admonition} Exemples
:class: note
* `range(3, 8, 2)` crée la suite de nombres `3, 5, 7`
* `range(3, 9, 2)` crée la suite de nombres `3, 5, 7` (pas le 9 qui est exclus)
* `range(3, 5, 2)` crée la suite de nombres `3`
* `range(7, 5, 1)` crée une suite vide puisque 7 est plus grand que 5

On peut aussi utiliser un pas négatif (avec `start` > `stop`) :
* `range(8, 4, -1)` crée la suite de nombres `8, 7, 6, 5`
````

```{margin}
C'est un détail pour nous mais `range(n)` ne crée pas une liste (on peut s'en rendre compte en demande `print(range(3))` ou `type(range(3))`). Si on veut le transformer en liste, il faudra écrire `list(range(3))`. Cela dit, il est conseillé alors d'utiliser un vecteur numpy (cf. suite).

```

## Boucle while
### Un exemple
```{code-cell}
i = 0  # Initialisation
while i < 4:
	print(i ** 3)
	i = i + 1  # Incrémentation
```

### Syntaxe
* __Initialisation__ : Une boucle nécessite en général une initialisation (précédemment définir le `i` et lui donner une valeur de départ).
* `while condition:` : le `:` est obligatoire
* le bloc d'instruction __doit être indenté__.
* le bloc d'instruction __doit contenir une instruction qui modifie la condition__ : sinon on crée un boucle infinie ! (précédemment, l'incrémentation modifie la valeur de `i` et donc la condition).

```{margin}
Un microcontrolleur (comme Arduino) est programmé avec une boucle infinie pour réaliser sans fin les instructions qu'on lui a donné. Dans nos programmes, on le fera rarement...
```

### Exemple
```{code-cell}
"""On va remplir une liste avec le carré des entiers n tant que n^2 - 2n < 5.
On ne peut utiliser une boucle for car on ne sait pas jusqu'à quel n aller.
"""
N = 5

""" Initialisation de la boucle """
l = []  # Création de la liste qu'on va remplir
n = 0  # Initialisation des entiers.

while n**2 - 2*n < N:  # Condition à vérifier
	l.append(n**2)  # Importance de l'initialisation : sinon on ne peut ajouter de valeur à la liste.
	n = n + 1  # Sans l'incrémentation, n garde la même valeur et la boucle est infinie.

print(l)
```


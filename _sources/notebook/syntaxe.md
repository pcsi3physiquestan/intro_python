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
# Syntaxe : récapitulation

## Opérations de base
```{tabbed} Les entiers
Type : `int`
Exemple : `3` ou `123543654` ou `-3465`

Opérateur/Fonctions usuelles :
* Addition : `3 + 4`
* Soustraction : `3 - 4`
* Multiplication : `3 * 4`
* Division entière : `5 // 3` (renvoie 1)
* Reste de la division euclidienne : `5 % 3` (renvoie 2)
* Puissance : `3 ** 2` (renvoie 9)
```

````{tabbed} Les flottants (nombres réels)
Type : `float`
Exemple : `3.45`ou `3254.234` ou `-23.34` ou `3.` ou `-5.` (observez le `.` après le 3 et le 5).

Opérateur/Fonctions usuelles :
* Addition : `3.2 + 4.1`
* Soustraction : `3. - 4.5`
* Multiplication : `3.1 * 4.2`
* Division : `3.3 / 4.`
* Puissance : `3.3 ** 4.`

```{note}
Une opération entre un entier et un flottant est possible, elle renverra un flottant.
```
````


```{tabbed} Les booléens (Vrai ou Faux)
Type : `bool`
Exemple : `True`, `False`

Opérateur/Fonctions usuelles :
* Négation : `not`
* Ou : `or`
* Et : `and`
```

````{tabbed} Les chaines de caractères
Type : `str`
Exemple : `"Hello World !"`, `"Les guillemets sont importantes"`, `'Ou des apostrophes'`

Opérateur/Fonctions usuelles :
* Concaténation : `"Hello" + "World !"` renvoie `"HelloWorld !"`
* Longueur : `len("Hello")` renvoie `5`
* Transformer un nombre en chaine de caractère : `str(345.2)` renvoie la chaine de caractère `"345.2"`
* Accès à un caractère : `"Hello"[0]` renvoie `"H"`, `"Hello"[4]` renvoie `"o"`,

```{attention}
La position des caractères commence à 0 et non à 1
```

````

## Les listes
* Créer une liste vide : `l = []`
* Créer une liste à partir de données : `l = [3, "phrase", 4.2, "teste"]`
* Ajouter un élément à une liste `l2` __à la fin__ : `l2.append("Un ajout")`
* Concaténer deux listes `l1` et `l3` : `l1 + l3`
* Longueur de la liste : `len(l)`
* Extraction : `a = l[k]` ($0 <= k < len(l)$)

## Fonctions
### Fonctions standards utiles (non exhaustif)
* `print(a)` : Affiche la variable
* `len(l)` : Longueur d'une liste ou d'une chaine de caractères
* `min(a, b, ...)` et `max(a, b, ...)`
* `int(b)` : Transforme en entier (utile pour transformer une chaine de caractère `"1"` en entier `1`)
* `str(a)` : Transforme (un nombre) en une chaine de caractère. Utile pour afficher un nombre dans une phrase.

### Importer un module

```{code-block}
import nom_module as alias_module
```

### Définir une fonction
```{code-block}
def nom_fonction(arg1, arg2, ...):
	corps _de_la_fonction
	...
	return objets_à_retourner
```

## Blocs
````{tabbed} Condition
```{code-block}
if condition:
	...
elif condition:
	...
else:
	...
```
````
````{tabbed} Boucle for
```{code-block}
for var in obj_iter:
	corps_de_boucle
	...
```

`range(N)` permet d'itérer de 0 à __N-1__
````
````{tabbed} Boucle while
```{code-block}
Initialisation
while condition:
	...
	...
	Modification_de_la_condition
```
````

## Numpy
````{tabbed} Création
* `V1 = np.array(L)`  : L est une liste et V1 est un vecteur numpy
* `arange(start, stop, step)` : une vecteur de _flottants_ partant de `start` (inclus) et allant jusqu'à `stop` (__exclus__) par pas de valeur `step`.
* `linspace(start, stop, N)` : un vecteur de `N` éléments réparties uniformément entre les valeurs `start` et `stop` (__inclus__)
* `zeros(N)` : un vecteur de `N` éléments nuls.  
* `ones(N)` : un vecteur de `N` éléments tous égaux à 1.
````

````{tabbed} Extraction
Comme pour les listes
````

````{tabbed} Opérations
* Opérations termes à termes `+,-,/,*`
* Fonctions classiques __à partir de la bibliothèque numpy__ `sin, cos, tan, exp, log`
````

````{tabbed} Régression linéaire
```{code-block}
import numpy as np
coefs = np.polyfit(x, y, 1) 
""" 
coefs : vecteur numpy contenant
les coefficients du polynome par ordre croissant
"""
```
````

````{tabbed} Numpy.random
```{code-block}
import numpy.random as rd
X = rd.uniform(a, b, N)
"""
X : Vecteur numpy de taille N contenant
N tirages aléatoires uniformes entre a et b
"""
Y = rd.normal(m, s, N)
"""
X : Vecteur numpy de taille N contenant
N tirages aléatoires suivant une loi normale
d'espérance m et d'écart-type s
"""
```
````

````{tabbed} Importer depuis un fichier
```{code-block}
"""Extraire tout sous forme de tableau numpy"""
T = loadtxt('chemin_vers_fichier', skiprows=3, delimiter=',')

"""Extraire chaque colonne dans un vecteur différent"""
V1, V2, V3 = loadtxt('chemin_vers_fichier', skiprows=3, delimiter=',', unpack=False)

"""Extraire une colonne particulière"""
V1 = loadtxt('chemin_vers_fichier', skiprows=3, delimiter=',', usecols=2)
```
````

## Matplotlib.pyplot
### Structure générale
```{code-block}
import matplotlib.pyplot as plt

#...
#Creation des vecteurs x et y
#...
#...

f, ax = plt.subplots()  # On crée la fenêtre graphique et les axes (= zone de tracé, ici une seule zone)

f.suptitle("Titre du graphique")  # On donne un titre au graphique

ax.set_xlabel("Temps (s)")  # On légende les abscisses de la zone de tracé
ax.set_ylabel("Position (m)")  # On légende les ordonnées de la zone de tracé

ax.plot(x, y, label="Légende")  # On trace la courbe voulue

ax.legend()  # On affiche la légende de la zone de tracé

ax.grid()  # Optionnel : permet d'afficher une grille sur le graphique

plt.show()  # On demande d'afficher le graphique.
```

### Fonctions de tracés
* Nuage de points simples : `ax.plot(x, y, marker='+', linestyle='', color='red', label='Nuages')`
* Avec incertitudes : `ax.errorbar(x, y, xerr=incert_x, yerr=incert_y, marker='+', linestyle='', color='red', label='Nuages')`
* Histogramme : `hist(liste_valeurs, bins='rice')`


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

# Blocs d'instructions.
On distingue deux blocs d'instructions :
* les instructions conditionnelles
* les instructions itératives (boucles)

## Instructions conditionnelles

### Un exemple pas très utile...
```{code-cell}
L = [1, 4, 9, 2, 1]

if len(L) < 3:
	print("La longueur de la liste est inférieure à 3")
else:
	print("La longueur de la liste est supérieure à 3")
```

### Syntaxe
La syntaxe :
* Premiere condition à tester : `if condition :`. Le `:` est obligatoire.
* Deuxième condition à tester : `elif condition :`
* Bloc de fin : `else:` (facultatif)
* Sous chaque condition : __le bloc d'instruction à exécuter sous condition doit être indenté__.

### Les conditions
Un condition renvoie un booléen `True ` ou `False`. Exemples de condition :
* Egalité : `==` (__pas `=`__)
* Inégalité : `<, >, <=, >=`
* Appartenance à une liste : `3 in L` (renvoie True si `3` est un élément de la liste `L`)

### Un autre exemple

```{code-cell}
""" 
On va utiliser les instructions conditionnelles pour créer une fonction mathématique par morceaux
"""
def fonction_morceau(x):
	if x < 3:  # Si x < 3
		a = x - 4
		return a
	elif x < 5:  # Si 3 <= x < 5
		return x ** 2
	else:  # Si x >=5
		a = x + 4
		b = a ** 3
		return b

print(fonction_morceau(3))
print(fonction_morceau(4))
print(fonction_morceau(6))
```

```{margin}
L'exemple précédent montre que si le premier test est vrai, les tests suivant n'auront pas lieu, on sort du bloc conditionnel. Idem si le second test est vrai, on n'exécute pas le `else`.
```

## Instructions itératives.
Les boucles permettent de réaliser plusieurs fois la même série d'instructions. On distingue deux types de boucles :
* les boucles bornées (`for`) : on sait quand la boucle va s'arrêter.
* les boucles non bornées (`while`) : on impose une condition d'arrêt de la boucle sans savoir quand elle sera réalisée.

### Boucle for

#### Un exemple 

(exemple-for)=
```{code-cell}
L = [1, 3, 5, 7, 9]

for n in L:
	print(n**2)
```

#### Syntaxe
* Début de la boucle : `for variable_d_iteration in objet_iterable:` : le `:` est obligatoire
* Eviter d'utiliser comme `variable_d_iteration` une variable qui existe déjà.
* Un `objet_iterable` est un objet qui contient plusieurs éléments et qui peut être parcouru (itéré). La `variable_d_iteration` prendre comme valeur à chaque tour de boucle les différents éléments. Pour nous les `objet_iterable` sont :
    * les listes
    * les vecteurs numpy (cf. suite)
    * des objets crée pour la boucle comme `range(5)` qui va créer une sorte de "liste" [0, 1, 2, 3, 4]
* Le corps de la boucle (série d'instruction à exécuter en boucle) doit __être indenté__.

#### Exemple
* A partir d'une liste déjà créée : cf. l'[exemple précédent](exemple-for).
* On utilise `range(n)` :

```{code-cell}
for n in range(4):  # On aura donc n=0, n=1, n=2, n=3 et on sort de la boucle
	print(n**2)
```

```{margin}
On pourra utiliser `range(start, stop, pas)` qui crée une succession de valeur de `start` (inclus) jusqu'à `stop` (non inclus) avec un incrément de `pas` (facultatif, il vaut alors 1). _Cela ne fonctionne qu'avec des __entiers__._ 
```

```{important}
La "liste" créée par `range(n)` commence à 0 et s'arrête à __n-1__
```



### Boucle while
#### Un exemple
```{code-cell}
i = 0  # Initialisation
while i < 4:
	print(i ** 3)
	i = i + 1  # Incrémentation
```

#### Syntaxe
* __Initialisation__ : Une boucle nécessite en général une initialisation (précédemment définir le `i` et lui donner une valeur de départ).
* `while condition:` : le `:` est obligatoire
* le bloc d'instruction __doit être indenté__.
* le bloc d'instruction __doit contenir une instruction qui modifie la condition__ : sinon on crée un boucle infinie ! (précédemment, l'incrémentation modifie la valeur de `i` et donc la condition).

```{margin}
Un microcontrolleur (comme Arduino) est programmé avec une boucle infinie pour réaliser sans fin les instructions qu'on lui a donné. Dans nos programmes, on le fera rarement...
```

#### Exemple
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


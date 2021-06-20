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

# Instructions conditionnelles

## Un exemple pas très utile...
```{code-cell}
L = [1, 4, 9, 2, 1]

if len(L) < 3:
	print("La longueur de la liste est inférieure à 3")
else:
	print("La longueur de la liste est supérieure à 3")
```

## Syntaxe
La syntaxe :
* Premiere condition à tester : `if condition :`. Le `:` est obligatoire.
* Deuxième condition à tester : `elif condition :`
* Bloc de fin : `else:` (facultatif)
* Sous chaque condition : __le bloc d'instruction à exécuter sous condition doit être indenté__.

## Les conditions
Un condition renvoie un booléen `True ` ou `False`. Exemples de condition :
* Egalité : `==` (__pas `=`__)
* Inégalité : `<, >, <=, >=`
* Appartenance à une liste : `3 in L` (renvoie True si `3` est un élément de la liste `L`)
* Négation : `not(condition_a_nier)`
* Ou et Et : `(condition 1) and (condition 2)`; `(condition 1) or (condition 2)`

## Un autre exemple

```{code-cell}
""" 
On va utiliser les instructions conditionnelles pour créer une fonction mathématique par morceaux
"""
def fonction_morceau(x):
	if x < 3:  # Si x < 3
		a = x - 4
	elif x < 5:  # Si 3 <= x < 5
		a = x ** 2
	else:  # Si x >=5
		b = x + 4
		a = b ** 3
	return a

print(fonction_morceau(3))
print(fonction_morceau(4))
print(fonction_morceau(6))
```

```{margin}
L'exemple précédent montre que si le premier test est vrai, les tests suivant n'auront pas lieu, on sort du bloc conditionnel. Idem si le second test est vrai, on n'exécute pas le `else`.
```


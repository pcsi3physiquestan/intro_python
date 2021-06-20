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

# Fonctions, conditions boucles

L'utilisation de fichiers pour écrire son écrire permet facilement de créer des __blocs__ d'instructions qui vont avoir un comportement particulier :
* les __fonctions__ qui enregistrent une série d'instructions à partir de certains données (arguments) et qui pourront être utilisées et réutilisées ultérieurement.
* les __structures conditionnelles__ qui permet au programme d'adapter l'instruction à réaliser en fonction d'une condition.
* les __boucles__ qui permettent de répéter plusieurs fois les mêmes instructions.

En Python, on rappelle que __toutes les instructions d'un même bloc doivent être indentés avec le même décalage__.

```{code-block}
"""La structure d'un bloc sera donc : """
declaration_du_bloc:  # Les : sont obligatoires. Python renverra une erreur sinon.
    a = 1  # On observe l'indentation : début du bloc
    b = a
    c = b * a  # On est toujours dans le bloc

a = 23  # Cette instruction n'est plus dans le bloc.
```

```{margin}
N'hésitez à sautez des lignes avant et après un bloc pour plus de lisibilité.

```
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

```{note}
Il existe des fonctions de la bibliothèque standard (`open`) qui permettent d'accéder au contenu d'un fichier puis d'écrire dans un fichier. Nous ne présentons pas ici ces fonctions pour ne pas alourdir la présentation bien qu'elles puissent être très utiles pour un utilisateur de Python. On se concentre ici sur l'importation et l'exportation de données expérimentales qui seront des nombres (entiers ou flottants) directement dans des vecteurs et tableaux `numpy`.
```

+++

# Traiter des données expérimentales avec numpy

Lorsqu'il y a peu de données expérimentales, on peut les importer "à la main" mais si ces données sont nombreuses, il devient utile de stocker ces données dans un fichier (extension `.txt`, `.dat` ou autre) pour les importer dans Python lorsqu'on veut les utiliser. Cela permettra en plus de réutiliser un script Python déjà écrit sur de nouvelles données expérimentales en ne changeant que le nom du fichier.

+++

## Préambule : Répertoire et chemin vers un fichier

En général les données expérimentales proviendront :
* d'un système d'acquisition tiers qui a enregistré les données dans un fichier localisé ailleurs sur l'ordinateur. Dans ce cas, il faudra localiser l'endroit où est stocké le fichier : la suite de répertoires et sous-répertoires dans lequel se trouve le fichier, c'est-à-dire le __chemin__ vers ce fichier.
* d'un fichier partagé par l'enseignant ou par des camarades. Il faudra en général le télécharger sur votre ordinateur (ou celui du TP) en choisissant correctement le répertoire de destination (_il est conseillé de le déplacé dans le même répertoire que le fichier contenant votre script_).

Dans tous les cas, Python n'est pas censé connaître où est le fichier de données, il faudra donc lui donner le __chemin__ d'accès au fichier.

+++

### Chemin absolu

Si vous avez déjà utilisé un explorateur de fichier, vous avez peut-être remarqué qu'en remontant d'un dossier parent à un autre on finit par arrive à un répertoire _racine_ (`C:/` sous `Windows` et `/` ou le nom de l'ordinateur sous `MacOs` et `Linux`). On peut ainsi localiser un fichier par la succession de répértoire dans lesquels il est inclus depuis ce répertoire racine, on parle de __chemin absolu__.

> Exemple sous Windows : Si vous créez un fichier `exercice.py` dans le répertoire `python_prepa` que nous vous avons conseillé de créer. Il est situé dans la succession de répertoire (sous Windows) `Users > nom_utilisateur > Documents > python_prepa` (`nom_utilisateur` est à remplacer par le nom que vous avez choisi à la création de votre compte sur votre ordinateur). On note alors le __chemin absolu vers le fichier__ :
> `C:/Users/nom_utilisateur/Documents/python_prepa/exercice.py`

```{margin}
* On donne ici la syntaxe utilisée sous Python. Sous Windows (dans un terminal par exemple), il faudra remplacer les `/` par des `\`.
* Le répertoire racine peut-être une autre lettre (`D:/`, `P:/`) en cas de disques durs multiples ou d'utilisation d'un réseau.
```


> Exemple sous MacOS et Linux : Si vous créez un fichier `exercice.py` dans le répertoire `python_prepa` que nous vous avons conseillé de créer. Il est situé dans la succession de répertoire (sous MacOs ou Linux) `home > nom_utilisateur > python_prepa` (`nom_utilisateur` est à remplacer par le nom que vous avez choisi à la création de votre compte sur votre ordinateur). On note alors le __chemin absolu vers le fichier__ :
> `/home/nom_utilisateur/python_prepa/exercice.py`

+++

### Chemin relatif
Les chemins absolus sont pratiques car ils ne souffrent d'aucune ambiguité. Il ont par contre le défaut d'être long et nécessite la connaissance du chemin complet (logique !). Il est souvent pratique d'utiliser uniquement le chemin depuis le fichier qu'on est en train d'utiliser. On parle de __chemin relatif__.

> Exemple : On utilise le fichier `exercice.py` précédent mais on a besoin de données expérimentales situées dans le fichier `/home/nom_utilisateur/python_prepa/donnees/data.txt`. Pour "passer" du répertoire de `exercice.py` au répertoire de `data.txt`, il suffit simplement d'entrer dans le répertoire `donnees`. Le __chemin relatif de `exercice.py` vers `data.txt` est alors :__
> `donnees/data.txt`
> _On remarquera l'absence de `C:/` ou `/` au début qui signifie qu'il s'agit d'un chemin relatif._

La plupart du temps, l'utilisation de chemin relatif sera préfèrée.

````{note}
Il arrive qu'on doivent remonter d'un répertoire, on utilise alors `..`

> Exemple : le chemin relatif de `data.txt` vers `exercice.py` est `../exercice.py`

````

### os.chdir
Il existe un module `os` dans l'une des fonctions `chdir(path:string)` permet de __définir le répertoire de travail de Python pour la suite du script.__ C'est dans le répertoire de travail que Python va, par défaut recherche les fichiers que l'on veut importer. __Il est vivement conseillé, lorsqu'on travaille avec des fichiers de données de définir un répertoire de travail.__

> Exemple : Ouverture du fichier `donnees.txt` situé dans le répertoire `C:/Users/nom_utilisateur/Documents/python_prepa/`
```{code-block}
from os import chdir  # Importation de chdir
chdir("C:/Users/nom_utilisateur/Documents/python_prepa/")  # Le nom de repertoire doit être une chaine de caractère

open("donnees.txt", 'r')  # Exemple de façon d' 'ouvrir un fichier (non expliqué ici)
```

+++

## Préparer l'importation
Avant d'importer un fichier de données, il est important :
* de le localiser (cf. supra)
* de l'ouvrir pour regarder la forme du fichier (_pyzo permet en général d'ouvrir un fichier texte contenant des données_)

Un fichier de données expérimentales contient en général :
1. (Facultatif) un en-tête décrivant les données expérimentales recueillies (expérience réalisée et conditions expérimentales, date, ...)
2. Un tableau de valeurs avec 
	* une ligne (ou plusieurs) d'en-tête donnant les grandeurs mesurées et leur unité
	* des lignes contenant les données expérimentales
	* sur une ligne, chaque donnée (chaque "colonne") est séparée par un séparateur (`,` ou `;`)

Pour préparer l'importation, il est important de vérifier le nombre de lignes de commentaires (elles ne seront pas importées), la ligne titre qui donne l'ordre des grandeurs mesurées, l'unité... et le séparateur utilisé. Dans [l'exemple ci-après](donnees_exp), on observe :
* 6 lignes de commentaires et une ligne d'en-tête
* les mesures dans l'ordre : Température en Kelvin, Pression en bar, Volume en $cm^3$ puis un entier représentant l'état du système (Liquide, Gaz, Liquide+Gaz, Fluide supercritique).
* Une séparation des colonnes par des `,`

```{figure} ./images/fichier_donnees.png
:name: donnees_exp
:align: center
Exemple de fichier de données affichés dans Pyzo
```

+++

## Importation dans Python
```{margin}
Il existe d'autres fonctions d'importation comme `fromfile` qui ne sont pas présentée ici pour ne pas alourdir la présentation.
```

+++

### Syntaxe
La bibliothèque `numpy` propose la fonction `loadtxt` qui permet d'importer des données __numériques__ dans un tableau numpy. Sa signature est :

```{code-block}
loadtxt('chemin_vers_fichier', skiprows=3, delimiter=',', usecols=(0, 1, 4), unpack=False)
```
Il renvoie un tableau numpy.

* `'chemin_vers_fichier'` est __une chaine de caractère__ (entre `""` donc) donnant le chemin (relatif ou absolu) vers le fichier à lire
* `skiprows=3` donne le nombre de ligne au début du fichier à sauter (commentaires et ligne de titre)
* `delimiter=','` est le caractère (ne pas oublier les `''`autour de la virgule) qui sépare les colonnes dans le fichier de données.
* `usecols=(0, 1, 4)` indique les colonnes à importer. Cet argument est __optionnel__, si rien n'est donné, toutes les colonnes sont importées.
```{margin}
* Si vous voulez importer une seul colonne, utiliser la syntaxe `usecols=3`. Vous obtiendrez alors un vecteur numpy.
* L'indice des colonnes commence à 0 comme pour un tableau numpy.
```
* `unpack`: (Optionnel). 
	* `False` (par défaut), la fonction renvoie les données sous forme de tableau numpy avec la même structure que dans le fichier. 
	* `True`, renvoie les données sous plusieurs variables (vecteurs numpy) et chaque variable contient une colonne
* _D'autres arguments optionnels existent mais ils ont moins d'utilité._

```{attention}
L'importation ne pourra fonctionner que si le tableau de données expérimentales est bien formé, c'est-à-dire si chaque ligne de données possède le même nombre de colonnes.
```

+++

### Exemple
Le script suivant est dans un fichier. Les données précédentes sont situées dans le fichier `Sf6/sf6.dat`.

```{dropdown} C'est un chemin...
... relatif par rapport au fichier où se trouve le script. On se permet donc de ne pas utiliser `chdir`.
```

```{code-cell}
:tags: [output_scroll]
import numpy as np  # On n'oublie pas d'importer numpy

nom_fichier = "Sf6/sf6.dat"  # Bien penser aux "" car ce doit être une chaine de caractère

"""Importation complète sous forme de tableau"""
datas1 = np.loadtxt(nom_fichier, skiprows=7, delimiter=',')  # Il y a bien 6 lignes de commentaires et une ligne de titre.
print("Tableau complet : ")
print(datas1)  # Le tableau étant volumineux, Python n'en affiche qu'une partie.

print("Première colonne du tableau :")
print(datas1[:, 0])  # On peut extraire une colonne, ici la colonne des températures.


"""Importation complète avec un vecteur par colonne"""
T, P, V, Etat = np.loadtxt(nom_fichier, skiprows=7, delimiter=',', unpack=True)  # On récupère ainsi des vecteurs séparés

print("Vecteur de pressions :")
print(P)  # Vecteur des données de Pression en bar


"""Importation partielle d'une colonne
Il est déconseillé, pour des raison d'efficacité d'importer une à une les colonnes si vous les voulez toutes.
"""
V1 = np.loadtxt(nom_fichier, skiprows=7, delimiter=',', usecols=2)  # On importe la colonne d'indice 2, soit les volumes

print("Vecteur des volumes :")
print(V1)  # Vecteur de données de Volume en m^3

```

+++

## Sauvegarder des données depuis Python
Vous l'utiliserez peut-être moins souvent, mais vous pouvez sauvegarder un tableau numpy dans un fichier avec `savetxt` de la bibliothèque `numpy` :

```{code-block}
np.savetxt(nom_fichier, tableau_a_sauver, delimiter=',', comments='', header='')
```

* `comments` : Commentaires à ajouter au début du fichier
* `header` : Chain de caractère ajoutant une ligne d'en-tête (par exemple `"T(K),P(bar),V(m^3)"`)
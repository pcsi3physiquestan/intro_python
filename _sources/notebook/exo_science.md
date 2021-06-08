---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  formats: ipynb, md:myst, py
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

# A vous de coder
Voici quelques exercices pour vous entraîner à coder. Il est conseillé d'utiliser Pyzo pour créer un fichier par exercice puis les exécuter. Quelques consignes :
* Créér un répertoire dans lequel vous mettrez vos fichiers contenant vos scripts.
* N'oubliez pas de bien indenter votre code
* Pensez à commenter votre code
* N'oubliez pas d'importer les bibliothèques numpy et matplotlib.pyplot lorsqu'elles sont utiles.
* Essayer de lire les messages d'erreurs quand il y en a pour comprendre où vous avez un problème (numéro de ligne) et quel est le problème. Si vous avez du mal à les lire, quelques conseils sont données [ici](erreurs).

## Exercices

```{admonition} Exercice 1
Vous devez tracer le graphique de la fonction $f(x) = 2x + 3 * sin(x)$ sur l'intervalle $[0, 4\pi]$ en prenant $N = 2000$ points pour le tracé.

On suppose qu'il s'agit du mouvement d'une "Masse sur un axe". Le tracé de la courbe devra être rouge (trait plein). Les abscisses seront des temps (en secondes) et les ordonnées des positions (en mètre). Utilisez ces informations pour légender et titrer votre graphique.
```

```{toggle} Quelques indices pour s'organiser
* Vous avez besoin des deux bibliothèques. N'oubliez pas de les importer.
* Vous n'allez pas créer un vecteur de 2000 points manuellement, pensez aux fonction qui permettent de créer automatiquement des vecteurs.
* Seule la fonction `sin` de la bibliothèque `numpy` est vectorialisable.
* Essayer d'avoir une organisation claire de vos instructions pour créer le graphique. Vous pourrez la réutiliser souvent.
```

Vous devriez obtenir le graphique suivant :

```{code-cell}
:tags: [remove-input]
import matplotlib.pyplot  as plt
import numpy as np

def f(x):
  return 2*x + np.sin(x)

x = np.linspace(0, 2 * np.pi, 2000)
y = f(x)

f, ax = plt.subplots()
f.suptitle("Masse sur un axe")

ax.set_xlabel("t(s)")
ax.set_ylabel("Position(m)")

ax.plot(x, y, color='red', label='x(t)')

ax.legend()
```



```{admonition} Exercice 2
La fonction `uniform(a, b, N)` de la bibliothèque `numpy.random` permet de réaliser N tirages aléatoires suivant une loi uniforme (toutes les valeurs ont la même probabilité) entre a et b. La fonction renvoie un vecteur numpy de taille N contenant les tirages.

Vous devez :
1. importer la sous-bibliothèque `numpy.random` dans l'espace nommé `rd` pour l'utiliser ensuite.
2. Utiliser la fonction `uniform` pour créer un vecteur contenant $N = 1000000$ de tirages aléatoires entre 0 et 5.
3. Obtenir alors le vecteur X contenant le carré de chaque valeur tirée précédemment.
4. Tracer l'histogramme des valeurs de X. Le titre sera "Distribution statistique", il n'est pas utile d'afficher la légende.

```

```{toggle} Quelques indices pour s'organiser
* Vous avez besoin des deux bibliothèques scientifiques en plus de `numpy.random`. N'oubliez pas de les importer aussi.
* Bien comprendre le fonctionnement de `uniform` pour l'utiliser.
```

Vous devriez obtenir le graphique suivant (cela peut un peu varier puisqu'on fait des tirages aléatoires) :

```{code-cell}
:tags: [remove-input]
import matplotlib.pyplot  as plt
import numpy as np
import numpy.random as rd

N = 1000000

x = rd.uniform(0, 5, N)
y = x * x

f, ax = plt.subplots()
f.suptitle("Distribution statistique")

ax.set_xlabel("Valeurs accessibles")
ax.set_ylabel("Fréquences")

ax.hist(y, bins='rice')

plt.show()
```
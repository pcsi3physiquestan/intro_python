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

# Le langage Python

Python est langage de programmation dit __interprété__ : il va lire une à une la série d'instruction écrite par l'utilisateur et les "interpréter" en langage machine pour exécuter les dites instructions.

```{margin}
On le différencie des langages dits __compilés__ qui lise l'ensemble du programme pour le transformer (le compiler) en un fichier exécutable en langage machine qui sera le programme qui réalisera la tâche qui lui est demandée.
```

On utilise Python de deux manières :
* Dans une console Python (appelé aussi Shell) : on rentre alors une instruction. Python l'interprête et l'exécute. On peut alors en rentrer une autre et ainsi de suite.
* Dans un fichier contenant une série d'instruction les unes à la suite des autres. La commande python va alors lire et exécuter l'ensemble des instructions les unes après les autres. On peut ainsi écrire des programmes plus complexes et c'est pourquoi on privilégiera cette façon de procéder.

```{margin}
Les environnements de développements comme Spyder ou Pyzo permettent d'interprêter un fichier par simple clic mais on peut aussi demander à python d'interprêter un fichier depuis un terminal en ligne de commande.
```

````{admonition} Installation de Python
:class: note
Python existe sous plusieurs versions et possède des bibliothèques supplémentaires proposant des fonctionnalités utiles (calcul scientifique, tracé graphique, analyse numérique...). Dans le cadre des classes préparatoires, il est important d'installer la verion 3.X de Python ainsi que les bibliothèques scientifiques comme numyp, scipy, matplotlib...

Pour une installation simple et identique à ce qui sera utilisé dans l'année, il est demandé d'installer la suite Anaconda Python et l'environnement Pyzo. Vous trouverez [ici un tutoriel détaillé pour l'installation des deux](https://filedn.eu/l2bpHGgwv4dYLpu8bJBwYM7/Stan/Pyzo_Anaconda/Python_pyzo_insta_gen_auroraW/co/Python_Installation.html).
````

Remarque : A plusieurs reprises des éléments de code sont proposés. Vous pouvez les tester vous même en les copiant dans un fichier script que vous exécuterez. En haut à droite de chaque cellule de code, vous trouverez un petit bouton ![Bouton](./images/copie_code.png) qui vous permettra de copier le code pour le coller ensuite dans votre fichier.
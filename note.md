# TP2

### Questions : - Quelle est l’importance de mélanger les données avant de les diviser en ensembles d’entraînement et de test ?

Mélanger les données avant de les diviser en ensembles d'entraînement et de test revêt une importance cruciale dans le domaine de l'apprentissage machine. Cette pratique garantit une répartition aléatoire des observations, offrant ainsi plusieurs avantages essentiels pour la formation et l'évaluation des modèles.

Tout d'abord, le mélange aléatoire prévient tout biais lié à la distribution initiale des données. Si les données sont ordonnées de manière particulière (par exemple, toutes les instances d'une classe sont regroupées), une simple division pourrait conduire à des ensembles d'entraînement et de test non représentatifs, introduisant un biais indésirable dans l'évaluation du modèle.

Ensuite, mélanger les données assure une généralisation robuste du modèle. Un modèle bien généralisé doit être capable de faire des prédictions précises sur des données qu'il n'a pas vues pendant l'entraînement. Le mélange aléatoire garantit que le modèle est exposé à une variété de scénarios dès le départ, favorisant ainsi sa capacité à généraliser efficacement sur de nouvelles données.

Enfin, cette pratique contribue à la reproductibilité des résultats. En garantissant un mélange aléatoire avec une graine de générateur de nombres aléatoires fixée, les expériences peuvent être reproduites de manière cohérente. Cela est particulièrement crucial pour des projets de recherche et de développement où la reproductibilité des résultats est une considération essentielle.

En résumé, le mélange aléatoire des données avant de les diviser en ensembles d'entraînement et de test est une étape fondamentale pour garantir des évaluations justes et représentatives des performances des modèles d'apprentissage machine.

----

### Pourquoi devons-nous créer une variable cible binaire ?

La création d'une variable cible binaire est nécessaire pour adapter le problème de classification à un modèle de machine learning, en l'occurrence le modèle SGDClassifier utilisé dans ce cas. Dans la classification binaire, le modèle doit apprendre à distinguer entre deux classes, par exemple, prédire si une image correspond ou non à un chiffre spécifique (comme le chiffre 5 dans cet exemple). En attribuant la valeur True à la classe d'intérêt (correspondant au chiffre choisi) et False à toutes les autres classes, nous transformons le problème en une tâche de classification binaire.

### Comment entraînons-nous le modèle SGDClassifier pour cette tâche de classification binaire ?

Nous utilisons l'ensemble d'entraînement préalablement créé avec la variable cible binaire pour entraîner le modèle SGDClassifier. Le modèle est initié à l'aide de la classe SGDClassifier de scikit-learn, puis entraîné à l'aide de la méthode fit en fournissant les données d'entraînement (X_train) et la variable cible binaire correspondante (y_train_binary). Pendant l'entraînement, le modèle ajuste ses paramètres afin de minimiser la fonction de perte, permettant ainsi de faire des prédictions précises sur de nouvelles données. Une fois le modèle entraîné, il peut être utilisé pour faire des prédictions sur de nouvelles images, comme illustré dans l'exemple où il prédit si une image donnée correspond au chiffre choisi (par exemple, le chiffre 5).

-----

### Quelle est la signification de la précision et du rappel dans le contexte de la classification ?

La précision et le rappel sont deux métriques essentielles utilisées pour évaluer la performance d'un modèle de classification, notamment dans un contexte binaire où l'on cherche à distinguer deux classes (positif et négatif, par exemple). 

- **Précision :** La précision mesure la proportion d'instances correctement classées comme positives parmi toutes les instances classées comme positives par le modèle. En d'autres termes, elle évalue la qualité des prédictions positives du modèle. Une précision élevée signifie que le modèle a fait peu d'erreurs en classant des instances négatives comme positives.

- **Rappel :** Le rappel, également appelé sensibilité ou taux de vrais positifs, mesure la proportion d'instances positives correctement classées parmi toutes les instances réellement positives. Il évalue la capacité du modèle à ne pas manquer d'instances positives. Un rappel élevé indique que le modèle a réussi à capturer la plupart des instances positives.

Ces deux métriques sont souvent en tension l'une avec l'autre. Améliorer la précision peut souvent conduire à une diminution du rappel et vice versa, ce qui est connu sous le nom de compromis précision-rappel.

### Comment calculons-nous la précision et le rappel ?

- **Précision :** La précision se calcule en divisant le nombre de vrais positifs par la somme des vrais positifs et des faux positifs.
  
    Précision = Vrais positifs /  Vrais positifs + Faux positifs

- **Rappel :** Le rappel se calcule en divisant le nombre de vrais positifs par la somme des vrais positifs et des faux négatifs.

    Précision = Vrais positifs /  Vrais positifs + Faux positifs

Ces formules expriment le pourcentage d'exactitude des prédictions positives par rapport à différentes références, offrant ainsi une évaluation approfondie de la capacité du modèle à identifier les instances de la classe positive.

----

### Différence entre le SGDClassifier et le RandomForestClassifier :

Le **SGDClassifier (Stochastic Gradient Descent Classifier)** et le **RandomForestClassifier** sont deux algorithmes de classification utilisés dans le domaine de l'apprentissage automatique, mais ils se basent sur des principes différents.

- **SGDClassifier :** Il s'agit d'un classificateur linéaire qui utilise la descente de gradient stochastique pour minimiser une fonction de coût. Il est particulièrement adapté aux grands ensembles de données en raison de son apprentissage itératif et de sa capacité à traiter les données par lots. Il fonctionne bien pour des tâches de classification binaire et multiclasse.

- **RandomForestClassifier :** En revanche, le RandomForestClassifier appartient à la famille des méthodes ensemblistes et est basé sur la construction d'arbres de décision multiples. Il crée un ensemble de nombreux arbres de décision indépendants, chaque arbre votant pour la classe la plus probable, et agrège ensuite ces votes pour obtenir une prédiction finale. Les forêts aléatoires sont souvent utilisées pour des ensembles de données complexes et présentant des interactions non linéaires.

### Comparaison des performances :

Pour comparer les performances du **SGDClassifier** et du **RandomForestClassifier**, nous utilisons généralement des métriques d'évaluation telles que la précision, le rappel, la F-mesure, l'aire sous la courbe ROC, etc.

Dans le code fourni, nous utilisons les métriques de précision et de rappel pour évaluer les performances de chaque modèle sur un ensemble de test. La précision mesure la proportion d'instances positives correctement classées parmi toutes les instances classées comme positives, tandis que le rappel mesure la proportion d'instances positives correctement classées parmi toutes les instances réellement positives.

En observant les résultats de ces métriques pour chaque modèle, nous pouvons avoir une idée de la capacité de chaque classificateur à faire des prédictions précises et à capturer efficacement les instances positives. Une comparaison des valeurs de précision et de rappel entre le SGDClassifier et le RandomForestClassifier nous permettra de déterminer lequel des deux modèles fonctionne mieux pour la tâche spécifique de classification binaire du chiffre choisi (par exemple, le chiffre 8).

----

### Comment adaptons-nous notre modèle pour effectuer une classification multiclasse ?

Pour effectuer une classification multiclasse, nous devons adapter notre modèle pour prendre en charge la prédiction de plusieurs classes plutôt que d'une seule. Deux approches courantes pour ce faire sont l'utilisation de classificateurs spécifiques à la classification multiclasse et la transformation de problèmes multiclasse en problèmes binaires.

1. **Classificateurs Multiclasse :** Certains algorithmes, tels que le `SGDClassifier` utilisé dans le code fourni, peuvent être utilisés directement pour effectuer une classification multiclasse. Ces classificateurs intègrent la gestion des multiples classes et peuvent attribuer une instance à une classe parmi plusieurs.

   ```python
   sgd_clf_multiclass = SGDClassifier(random_state=42)
   sgd_clf_multiclass.fit(X_train, y_train)
   ```

2. **Transformation de Problèmes :** Une autre approche consiste à transformer le problème multiclasse en plusieurs problèmes binaires. Par exemple, la méthode "One-vs-All" (ou "One-vs-Rest") consiste à entraîner un classificateur binaire pour chaque classe, en le distinguant des autres classes. Le modèle final prend alors la classe avec la prédiction la plus confiante.

   ```python
   from sklearn.multiclass import OneVsRestClassifier
   sgd_clf_multiclass_ovr = OneVsRestClassifier(SGDClassifier(random_state=42))
   sgd_clf_multiclass_ovr.fit(X_train, y_train)
   ```

### Quelles métriques d’évaluation devrions-nous utiliser pour évaluer la performance de ce modèle ?

Pour évaluer la performance d'un modèle de classification multiclasse, plusieurs métriques d'évaluation peuvent être utilisées en fonction des caractéristiques spécifiques du problème. Voici quelques-unes des métriques couramment utilisées :

1. **Précision (Precision) :** Mesure la proportion d'instances correctement classées parmi celles classées comme positives. Pour une classification multiclasse, la précision peut être calculée pour chaque classe, et une moyenne pondérée est souvent utilisée.

2. **Rappel (Recall) :** Mesure la proportion d'instances positives correctement classées parmi toutes les instances réellement positives. Comme la précision, le rappel peut être calculé pour chaque classe avec une moyenne pondérée.

3. **F1-score :** C'est la moyenne harmonique de la précision et du rappel. Il fournit une mesure équilibrée entre la précision et le rappel.

4. **Matrice de Confusion :** Une matrice qui montre le nombre d'instances de chaque classe classées correctement et incorrectement. Elle offre une vue détaillée des performances du modèle pour chaque classe.

5. **Précision Globale :** La proportion totale d'instances correctement classées parmi toutes les instances.

En utilisant ces métriques, nous pouvons obtenir une évaluation complète de la capacité du modèle à effectuer une classification multiclasse précise et cohérente sur un ensemble de données.

----
AUTHOR  : YURTSEVEN Huseyin
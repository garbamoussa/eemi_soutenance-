



# Customer-Intelligence-de-la-Data-au-Machine-Learning 



##  Soutenance et livrable final


### Cas pratique : Système de recommandation de films

#### Consignes de l'évaluation 
Présentation du projet final basé sur une étude de cas
Soutenance orale : Présentation des résultats et recommandations
Critères d'évaluation : rigueur, profondeur d'analyse et créativité
Retour et conseils pour l'amélioration future
Discussion autour des tendances futures en Customer Intelligence



##################  Cas pratique : Système de recommandation de films #################################


###  Contexte

Une plateforme de streaming inspirée de
👉 MovieLens
souhaite améliorer l’expérience utilisateur grâce à un moteur de recommandation personnalisé.

L’entreprise dispose des données suivantes issues du projet de recherche de
👉 GroupLens Research :

films et genres

notes utilisateurs

tags

liens vers IMDb/TMDB

Objectif : développer un système capable de recommander des films pertinents.

🎯 Objectifs du projet

Le système devra :

proposer des films personnalisés

recommander des films similaires

exploiter genres, notes et tags

améliorer la rétention utilisateur

optimiser les campagnes marketing

#2. Problématique

Comment concevoir un moteur de recommandation :

fiable

scalable

explicable

temps réel

tout en exploitant les données MovieLens ?

🛠️ Travail demandé : Cahier des charges

L’équipe data doit rédiger un cahier des charges fonctionnel et technique.

✅ Partie 1 — Analyse des besoins

Questions :

Qui sont les utilisateurs du système ?

Quels types de recommandations sont attendus ?

personnalisées

similaires

populaires

Quels indicateurs métier doivent être améliorés ?

engagement

durée de visionnage

conversion

✅ Partie 2 — Données

Questions :

Quelles tables du dataset seront utilisées ?

Comment traiter les genres ?

Comment exploiter les tags ?

Comment gérer les données manquantes ?

Comment gérer le cold start ?

✅ Partie 3 — Choix du modèle

Questions :

Filtrage collaboratif ou content-based ?

Pourquoi un modèle hybride ?

Quels algorithmes utiliser ?

KNN

Matrix factorization

Deep learning

Comment évaluer le modèle ?

✅ Partie 4 — Architecture technique

Questions :

Pipeline de données ?

Stockage des embeddings ?

API de recommandation ?

Batch vs temps réel ?

Scalabilité ?

✅ Partie 5 — Évaluation

Questions :

Métriques offline :

RMSE

Precision@K

Recall@K

Métriques business :

CTR

watch time

A/B testing ?

✅ Partie 6 — Contraintes

Questions :

Temps de réponse acceptable ?

Volume d’utilisateurs cible ?

Explicabilité requise ?

Protection des données ?

⭐ Extension avancée

Demander aux étudiants :

proposer un prototype Python

créer dashboard KPI

simuler cold start

proposer stratégie hybride

🧠 Questions critiques (niveau mémoire)

Pourquoi Netflix utilise hybride ?

Comment éviter biais de popularité ?

Comment intégrer contexte (heure, device) ?

Comment expliquer une recommandation ?

Quel compromis précision vs diversité ?

✍️ Travail demandé aux étudiants


Rédiger :

1.1   cahier des charges complet

1.2 choix du modèle

1.3 pipeline technique

1.4 plan d’évaluation

1.5roadmap projet

##################  Questions sur le notebook #################################


✅ Questions sur le nettoyage des données

Combien de lignes sont chargées après l’instruction pd.read_csv(..., nrows=100001) ?
👉 Donnez la forme du DataFrame.

Combien d’utilisateurs uniques (n_users) et d’items (n_items) avez-vous après le mapping des IDs ?

Pourquoi a-t-on créé User_ID_new et Item_ID_new ?
👉 Vérifiez avec un exemple concret d’utilisateur.

Quel est le niveau de sparsity affiché par le notebook ?
👉 Interprétez ce résultat.

✅ Questions Memory-based (User / Item)

Quelle est la taille de la matrice train_data_matrix ?

Quelle métrique de similarité est utilisée dans pairwise_distances ?

Quelle est la différence conceptuelle entre :

user-based

item-based

Après exécution, quel modèle donne le RMSE le plus faible ?

Expliquez pourquoi ce modèle est meilleur sur ce dataset.

✅ Questions sur la comparaison des prédictions

Comparez R et R_pred pour un utilisateur donné.
👉 Donnez un exemple d’item correctement prédit.

Trouvez un cas où la prédiction est mauvaise.
👉 Expliquez pourquoi.

✅ Questions recommandation personnalisée

Exécutez :

getrecom_membased_for_item(...)

👉 Quels sont les 3 premiers items recommandés ?

Quelle différence entre :

ch="all"

ch="discover"

Pourquoi exclure les items déjà notés ?

✅ Questions SVD (Model-based)

Quel est le rôle de svds ?

Combien de dimensions latentes sont utilisées ?

Quel est le RMSE obtenu avec SVD ?
👉 Comparez avec memory-based.

Pourquoi faut-il normaliser la matrice prédite ?

✅ Questions SGD-WR

Quel est le rôle des matrices P et Q ?

Que représente le paramètre :

k

lambda

Comment évolue le RMSE au fil des steps ?
👉 Décrivez la courbe.

Quel RMSE final est obtenu avec SGD ?
👉 Est-il meilleur que SVD ?

✅ Questions de réflexion (très efficaces pour forcer l’exécution)

Testez plusieurs valeurs de k.
👉 Comment change le RMSE ?

Testez un autre utilisateur que 18.
👉 Les recommandations changent-elles ?

Quel modèle recommanderiez-vous en production et pourquoi ?

⭐ Questions bonus (niveau master)

Pourquoi la sparsity pose problème aux modèles memory-based ?

Expliquez le cold-start dans ce notebook.

Comment hybrider memory-based et model-based ici ?





Description
=======

This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.

Users were selected at random for inclusion. All selected users had rated at least 20 movies. No demographic information is included. Each user is represented by an id, and no other information is provided.

The data are contained in the files `links.csv`, `movies.csv`, `ratings.csv` and `tags.csv`. More details about the contents and use of all these files follows.

This is a *development* dataset. As such, it may change over time and is not an appropriate dataset for shared research results. See available *benchmark* datasets if that is your intent.

This and other GroupLens data sets are publicly available for download at <http://grouplens.org/datasets/>.


Usage License
=============

Neither the University of Minnesota nor any of the researchers involved can guarantee the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:

* The user may not state or imply any endorsement from the University of Minnesota or the GroupLens Research Group.
* The user must acknowledge the use of the data set in publications resulting from the use of the data set (see below for citation information).
* The user may redistribute the data set, including transformations, so long as it is distributed under these same license conditions.
* The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from a faculty member of the GroupLens Research Project at the University of Minnesota.
* The executable software scripts are provided "as is" without warranty of any kind, either expressed or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. The entire risk as to the quality and performance of them is with you. Should the program prove defective, you assume the cost of all necessary servicing, repair or correction.

In no event shall the University of Minnesota, its affiliates or employees be liable to you for any damages arising out of the use or inability to use these programs (including but not limited to loss of data or data being rendered inaccurate).

If you have any further questions or comments, please email <grouplens-info@umn.edu>


Citation
========

To acknowledge use of the dataset in publications, please cite the following paper:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>


Further Information About GroupLens
===================================

GroupLens is a research group in the Department of Computer Science and Engineering at the University of Minnesota. Since its inception in 1992, GroupLens's research projects have explored a variety of fields including:

* recommender systems
* online communities
* mobile and ubiquitious technologies
* digital libraries
* local geographic information systems

GroupLens Research operates a movie recommender based on collaborative filtering, MovieLens, which is the source of these data. We encourage you to visit <http://movielens.org> to try it out! If you have exciting ideas for experimental work to conduct on MovieLens, send us an email at <grouplens-info@cs.umn.edu> - we are always interested in working with external collaborators.


Content and Use of Files
========================

Formatting and Encoding
-----------------------

The dataset files are written as [comma-separated values](http://en.wikipedia.org/wiki/Comma-separated_values) files with a single header row. Columns that contain commas (`,`) are escaped using double-quotes (`"`). These files are encoded as UTF-8. If accented characters in movie titles or tag values (e.g. Misérables, Les (1995)) display incorrectly, make sure that any program reading the data, such as a text editor, terminal, or script, is configured for UTF-8.


User Ids
--------

MovieLens users were selected at random for inclusion. Their ids have been anonymized. User ids are consistent between `ratings.csv` and `tags.csv` (i.e., the same id refers to the same user across the two files).


Movie Ids
---------

Only movies with at least one rating or tag are included in the dataset. These movie ids are consistent with those used on the MovieLens web site (e.g., id `1` corresponds to the URL <https://movielens.org/movies/1>). Movie ids are consistent between `ratings.csv`, `tags.csv`, `movies.csv`, and `links.csv` (i.e., the same id refers to the same movie across these four data files).


Ratings Data File Structure (ratings.csv)
-----------------------------------------

All ratings are contained in the file `ratings.csv`. Each line of this file after the header row represents one rating of one movie by one user, and has the following format:

    userId,movieId,rating,timestamp

The lines within this file are ordered first by userId, then, within user, by movieId.

Ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).

Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.


Tags Data File Structure (tags.csv)
-----------------------------------

All tags are contained in the file `tags.csv`. Each line of this file after the header row represents one tag applied to one movie by one user, and has the following format:

    userId,movieId,tag,timestamp

The lines within this file are ordered first by userId, then, within user, by movieId.

Tags are user-generated metadata about movies. Each tag is typically a single word or short phrase. The meaning, value, and purpose of a particular tag is determined by each user.

Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.


Movies Data File Structure (movies.csv)
---------------------------------------

Movie information is contained in the file `movies.csv`. Each line of this file after the header row represents one movie, and has the following format:

    movieId,title,genres

Movie titles are entered manually or imported from <https://www.themoviedb.org/>, and include the year of release in parentheses. Errors and inconsistencies may exist in these titles.

Genres are a pipe-separated list, and are selected from the following:

* Action
* Adventure
* Animation
* Children's
* Comedy
* Crime
* Documentary
* Drama
* Fantasy
* Film-Noir
* Horror
* Musical
* Mystery
* Romance
* Sci-Fi
* Thriller
* War
* Western
* (no genres listed)


Links Data File Structure (links.csv)
---------------------------------------

Identifiers that can be used to link to other sources of movie data are contained in the file `links.csv`. Each line of this file after the header row represents one movie, and has the following format:

    movieId,imdbId,tmdbId

movieId is an identifier for movies used by <https://movielens.org>. E.g., the movie Toy Story has the link <https://movielens.org/movies/1>.

imdbId is an identifier for movies used by <http://www.imdb.com>. E.g., the movie Toy Story has the link <http://www.imdb.com/title/tt0114709/>.

tmdbId is an identifier for movies used by <https://www.themoviedb.org>. E.g., the movie Toy Story has the link <https://www.themoviedb.org/movie/862>.

Use of the resources listed above is subject to the terms of each provider.


Cross-Validation
----------------

Prior versions of the MovieLens dataset included either pre-computed cross-folds or scripts to perform this computation. We no longer bundle either of these features with the dataset, since most modern toolkits provide this as a built-in feature. If you wish to learn about standard approaches to cross-fold computation in the context of recommender systems evaluation, see [LensKit](http://lenskit.org) for tools, documentation, and open-source code examples.




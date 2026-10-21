# Détection d'anomalies dans le trafic réseau — UNSW-NB15

**Code source et artefacts expérimentaux du Travail de Fin de Cycle (TFC)**  
**Université de Kinshasa — Faculté Polytechnique, Génie électrique et informatique — Année académique 2025–2026**  
**Sujet :** *Détection d’anomalies dans le trafic réseau par autoencodeur profond dans un cadre one-class : apprentissage du trafic normal et évaluation sur un jeu de données public d’intrusion.*  
**Auteure :** Sharon MUKANYA BUKASA.

> **Périmètre scientifique.** La comparaison principale oppose **Deep Dense Autoencoder (DDAE)** et **Isolation Forest (IF)** sur un **test indépendant commun**. Les résultats du **Convolutional Autoencoder (CAE)** et du **LSTM Autoencoder (LSTM-AE)** proviennent de **protocoles exploratoires distincts** et ne constituent pas une comparaison appariée des quatre modèles.

## 1. Objectif de l'étude

Modéliser le trafic réseau **normal** par apprentissage *one-class* puis signaler comme atypiques les observations dont le score dépasse un seuil. Pour les autoencodeurs, ce score correspond à l'erreur quadratique moyenne de reconstruction (MSE). Pour Isolation Forest, on utilise `-score_samples` afin qu'un score plus élevé signifie une plus grande atypicité.

Les étiquettes `label` (0 = normal ; 1 = attaque) servent à sélectionner les observations normales d'entraînement et à **évaluer** les détecteurs. Elles ne servent pas à apprendre de façon supervisée les familles d'attaques.

## 2. Organisation du dépôt

Les chemins ci-dessous sont relatifs à la racine du dépôt :

| Dossier / fichier | Contenu |
| --- | --- |
| [`Dense_Autoencodeur/Dense_Autoencodeur.ipynb`](Dense_Autoencodeur/Dense_Autoencodeur.ipynb) | Notebook du DDAE ; des variantes antérieures du notebook sont également conservées dans le dossier. |
| [`Dense_Autoencodeur/resultats_DDAE/`](Dense_Autoencodeur/resultats_DDAE/) | Modèles `.keras`, partitions, données transformées, scores `.npy`, graphiques, configurations et audits du DDAE. |
| [`Isolation_Forest/Isolation_Forest.ipynb`](Isolation_Forest/Isolation_Forest.ipynb) | Notebook d'Isolation Forest ; un ancien notebook est conservé séparément. |
| [`Isolation_Forest/resultats_Isolation_Forest/`](Isolation_Forest/resultats_Isolation_Forest/) | Modèle `.joblib`, préprocesseur, partitions, scores, figures, tableaux et audits IF. |
| [`CNN_Autoencodeur/CNN_Autoencodeur.ipynb`](CNN_Autoencodeur/CNN_Autoencodeur.ipynb) | Expérience exploratoire du CAE, modèle `.h5` et figures. |
| [`CNN_Autoencodeur/figures_corrigees_CAE/`](CNN_Autoencodeur/figures_corrigees_CAE/) | Visualisations du CAE issues des scores historiques et fichier de scores `.npz`. |
| [`LSTM_Autoencodeur/LSTM_Autoencodeur.ipynb`](LSTM_Autoencodeur/LSTM_Autoencodeur.ipynb) | Expérience exploratoire LSTM-AE, figures et [modèle sauvegardé](LSTM_Autoencodeur/models/lstm_autoencoder_best.keras). |
| [`Image/`](Image/) | Figures et illustrations complémentaires. |
| [`GRU_Autoencodeur/`](GRU_Autoencodeur/) | Essai GRU conservé comme archive de travail ; **non retenu dans la comparaison du mémoire**. |
| [`New/`](New/) et autres notebooks à la racine | Fichiers de travail ou visualisations complémentaires ; ne remplacent pas les résultats audités. |

**Pour les chiffres définitifs**, privilégier les rapports de synthèse et d'audit plutôt que les sorties de notebooks ou figures d'expériences antérieures.

## 3. Jeu de données

- Source : [UNSW-NB15, University of New South Wales](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
- Fichier utilisé : **`UNSW-NB15_1.csv`**, composé de **700 001 observations** : **677 786 normales** et **22 215 attaques**.
- Le CSV original **n'est pas versionné** dans ce dépôt (`*.csv` figure dans `.gitignore`). Il doit être obtenu séparément auprès de sa source et placé à l'emplacement attendu par les notebooks.
- Empreinte SHA-256 du fichier source consignée dans le rapport de comparabilité :  
  `7d851bbeabd27894ce39c8e78835c73341fc946652fb7743b9eff193b55eb511`.

**Vérification locale du fichier source sous PowerShell :**
```powershell
Get-FileHash -Algorithm SHA256 .\UNSW-NB15_1.csv
```
Le résultat doit correspondre à l'empreinte ci-dessus pour reproduire **ce fichier exact**. Les autres fichiers/versions d'UNSW-NB15 ne sont pas interchangeables sans vérification.

### Prétraitement des expériences principales (DDAE et IF)

1. Lecture du CSV brut et attribution explicite des 49 noms de colonnes (fichier sans en-tête).
2. Exclusion des deux cibles `label`, `attack_cat`, puis de `srcip`, `dstip`, `Stime`, `Ltime`.
3. Conservation de **43 variables** : 40 numériques et 3 catégorielles (`proto`, `state`, `service`).
4. Séparation des observations **avant** l'ajustement du prétraitement ; imputation médiane, standardisation, encodage *one-hot* ajustés uniquement sur les **normales d'entraînement**.
5. Obtention de **70 caractéristiques numériques encodées**.

La séparation repose sur des groupes de signatures d'observation, avec `GroupShuffleSplit` et les graines **42** et **43**. Elle limite la dispersion de doublons exacts entre partitions mais **ne démontre pas l'indépendance temporelle des sessions réseau**.

| Partition | Total | Population mobilisée |
| --- | ---: | --- |
| Entraînement | 490 140 | 474 263 observations normales |
| Validation | 104 990 | 102 016 observations normales |
| Test indépendant commun | 104 871 | 101 507 normales, 3 364 attaques |

Le test commun DDAE–IF a été audité : même empreinte du fichier source, mêmes identifiants du test, même ordre et mêmes étiquettes. Cette vérification du **test** n'est pas une démonstration d'identité intégrale de tous les traitements ou de toutes les données d'entraînement.

## 4. Modèles, paramètres et calibration

| Modèle | Configuration documentée | Statut |
| --- | --- | --- |
| **DDAE** | Architecture **70 → 32 → 16 → 32 → 70**, 5 654 paramètres ; Adam (`learning_rate=0.001`), MSE, batch 256, 50 époques ; meilleure validation à l'époque 45 | Comparaison principale |
| **Isolation Forest** | 200 arbres ; `max_samples=256`, `max_features=1.0`, `bootstrap=False`, `contamination='auto'`, `n_jobs=-1`, `random_state=42` | Comparaison principale |
| **CAE** | Convolutions 1D, entrée **(47, 1)**, 15 937 paramètres ; expérience d'origine distincte | Exploratoire |
| **LSTM-AE** | Fenêtres **(10, 48)** ; encodeur LSTM(64) → LSTM(32), puis `RepeatVector(10)`, LSTM(32), LSTM(64), `TimeDistributed(Dense(48))` ; 77 616 paramètres | Exploratoire |

**Calibration des deux détecteurs principaux :** seuil égal au **95e percentile (P95)** des scores sur la **validation normale**, sans optimisation à partir du test.

| Détecteur | Score (sens croissant = plus atypique) | Seuil P95 |
| --- | --- | ---: |
| DDAE | MSE de reconstruction | `0.035257672634652557` |
| Isolation Forest | `-score_samples(x)` | `0.5262464252920358` |

Pour le CAE et le LSTM-AE, les seuils historiques documentés sont respectivement **0,03379321** et **0,00355848**. Leurs populations ou unités d'évaluation diffèrent du test commun ; les fenêtres LSTM ne garantissent pas une session réseau temporellement continue.

## 5. Résultats expérimentaux

**Comparaison principale — même test de 104 871 observations, attaques = classe positive :**

| Modèle | Accuracy | Précision | Rappel | F1 | FPR | ROC-AUC | Average Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DDAE | 95,11 % | 39,19 % | 95,30 % | 55,54 % | 4,90 % | 0,9851 | 0,5746 |
| Isolation Forest | 95,46 % | 40,30 % | 86,00 % | 54,89 % | 4,22 % | 0,9811 | 0,5455 |

| Modèle | Vrais négatifs (TN) | Faux positifs (FP) | Faux négatifs (FN) | Vrais positifs (TP) |
| --- | ---: | ---: | ---: | ---: |
| DDAE | 96 533 | 4 974 | 158 | 3 206 |
| Isolation Forest | 97 222 | 4 285 | 471 | 2 893 |

**Compromis observé :** le DDAE identifie **313 attaques supplémentaires** mais génère **689 faux positifs supplémentaires** par rapport à IF. Ce résultat concerne le jeu de test décrit et ne démontre pas une supériorité universelle.

**Expériences exploratoires, non appariées au test commun :**

| Modèle | Accuracy | Précision | Rappel | F1 | FPR | ROC-AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CAE | 93,27 % | 20,97 % | 40,48 % | 27,63 % | 5,00 % | Non rapportée | Non rapportée |
| LSTM-AE (séquences) | 86,01 % | 15,97 % | 79,95 % | 26,62 % | 13,79 % | 0,8769 | 0,4418 |

**Attention :** ne pas classer les quatre architectures en utilisant directement ce tableau. Les protocoles CAE/LSTM-AE ne sont pas strictement comparables au protocole DDAE–IF.

## 6. Reproduire ou contrôler les résultats

### Environnement documenté

Relevé de l'environnement Python des expériences principales / de leur vérification : **Python 3.10.21**, Windows 10.0.26300, NumPy **1.26.4**, Pandas **2.3.3**, scikit-learn **1.7.2**, TensorFlow **2.15.0**, Matplotlib **3.10.9**, Seaborn **0.13.2**, SciPy **1.15.3**, JupyterLab **4.6.3**, IPykernel **7.3.0**.

Voir [`Dense_Autoencodeur/environnement_logiciel.txt`](Dense_Autoencodeur/environnement_logiciel.txt) et les fichiers `environnement_DDAE.json` / `configuration_environnement_IF.json`. Au moment du relevé, TensorFlow ne détectait aucun GPU. Le matériel utilisé pour toutes les expériences historiques n'est pas entièrement documenté.

**Exemple de préparation sous Windows / PowerShell :**
```powershell
git clone https://github.com/SharoBks/Codes_TFC_Sharon.git
cd Codes_TFC_Sharon
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4 pandas==2.3.3 scikit-learn==1.7.2 tensorflow==2.15.0 matplotlib==3.10.9 seaborn==0.13.2 scipy==1.15.3 jupyterlab==4.6.3 ipykernel==7.3.0 joblib openpyxl
jupyter lab
```

Ces commandes décrivent un **environnement de départ** ; elles ne constituent pas une garantie de reproduction bit à bit sur toute machine. Si le dépôt est privé, le clonage nécessite un accès GitHub autorisé. Certains notebooks peuvent conserver des **chemins absolus Windows** : adapter les chemins du CSV et des sorties à sa propre machine avant d'exécuter les cellules.

### Contrôle rapide *sans réentraînement* (DDAE + IF)

Depuis la **racine du dépôt**, après installation de NumPy et de scikit-learn, exécuter :

```python
import numpy as np
from sklearn.metrics import confusion_matrix

y = np.load("Dense_Autoencodeur/resultats_DDAE/artefacts/matrices_communes/y_test.npy")

experiences = {
    "DDAE": (
        "Dense_Autoencodeur/resultats_DDAE/scores/scores_MSE_test_DDAE.npy",
        0.035257672634652557,
    ),
    "Isolation Forest": (
        "Isolation_Forest/resultats_Isolation_Forest/scores/scores_test_Isolation_Forest.npy",
        0.5262464252920358,
    ),
}

for modele, (fichier_scores, seuil) in experiences.items():
    scores = np.load(fichier_scores)
    assert len(y) == len(scores) == 104871, "Effectifs de test inattendus"
    predictions = (scores > seuil).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    print(f"{modele}: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
```

Les fichiers de scores et étiquettes sont conservés dans un ordre commun, documenté par l'audit de comparabilité. Pour une vérification rigoureuse, consulter ce rapport et les tableaux ci-dessus. **Le script de contrôle ne réentraîne pas les modèles.**

### Réexécution des notebooks

1. Télécharger le fichier source `UNSW-NB15_1.csv` et en vérifier l'empreinte SHA-256.
2. Installer les dépendances, lancer JupyterLab et adapter les chemins de fichiers nécessaires.
3. Pour la comparaison principale, se référer aux notebooks `Dense_Autoencodeur/Dense_Autoencodeur.ipynb` et `Isolation_Forest/Isolation_Forest.ipynb`, en distinguant les cellules de génération d'artefacts et celles d'audit.
4. Conserver la séparation entraînement/validation/test, le prétraitement ajusté sur les seules normales d'entraînement, la calibration sur validation normale et **l'absence d'utilisation du test pour choisir les hyperparamètres**.
5. Utiliser les artefacts et rapports archivés pour comparer les nouvelles sorties aux résultats documentés.

**Prudence :** les notebooks conservés peuvent inclure différentes versions de travail. Éviter d'écraser les modèles, matrices, seuils et rapports audités lors d'une nouvelle exécution.

## 7. Sources de vérité et audits

Les résultats du mémoire sont principalement traçables dans les fichiers suivants :

- [Synthèse définitive DDAE](Dense_Autoencodeur/resultats_DDAE/synthese_finale_DDAE/resultats_DDAE_pour_comparaison.json)
- [Rapport d'audit final DDAE](Dense_Autoencodeur/resultats_DDAE/audit_final_DDAE/rapport_audit_final_DDAE.json)
- [Synthèse expérimentale IF](Isolation_Forest/resultats_Isolation_Forest/audit/audit_synthese_experimentale_IF.json)
- [Rapport de comparabilité DDAE–IF](Isolation_Forest/resultats_Isolation_Forest/audit/comparabilite_experimentale_IF_DDAE.json)
- [Seuil DDAE documenté](Dense_Autoencodeur/resultats_DDAE/artefacts/seuil_DDAE_95e_percentile.json)
- [Seuil IF documenté](Isolation_Forest/resultats_Isolation_Forest/artefacts/seuil_Isolation_Forest_95e_percentile.json)
- [Modèle DDAE retenu](Dense_Autoencodeur/resultats_DDAE/modeles/DDAE_meilleur_modele.keras)
- [Modèle Isolation Forest](Isolation_Forest/resultats_Isolation_Forest/modeles/Isolation_Forest_one_class.joblib)

Le rapport de synthèse IF et le rapport de **comparabilité** ont des objectifs distincts : le premier documente l'évaluation d'IF ; le second établit explicitement la concordance avec le test DDAE. Les chemins locaux éventuellement inclus dans certains rapports sont des traces historiques de l'environnement Windows d'origine et doivent être adaptés en cas de réexécution.

## 8. Limites et bonnes pratiques

- **Indépendance temporelle non prouvée :** le partitionnement par signatures n'assure pas que les flux ou sessions ne se recoupent pas dans le temps.
- **Généralisation non démontrée :** il s'agit d'une évaluation hors ligne sur un fichier d'UNSW-NB15, pas d'un déploiement IDS validé en conditions réelles.
- **Protocoles distincts :** les résultats CAE/LSTM-AE sont exploratoires et non directement comparables à ceux du test commun DDAE–IF.
- **Ressources de calcul :** les temps d'entraînement, d'inférence et la mémoire n'ont pas été comparés selon un protocole matériel homogène.
- **Faux positifs :** un seuil P95 calibré sur validation normale ne garantit pas 5 % de faux positifs sur tout autre trafic.
- **Données et autorisations :** respecter les conditions d'utilisation du jeu de données UNSW-NB15 et ne pas publier de fichiers sensibles.
- **Licence :** aucune licence logicielle n'est explicitement définie ici ; la visibilité du dépôt n'est pas, à elle seule, une autorisation de réutilisation.

## 9. Contact et citation

Pour référencer ce projet dans un travail académique :

> MUKANYA BUKASA, S., *Détection d'anomalies dans le trafic réseau par autoencodeur profond dans un cadre one-class : apprentissage du trafic normal et évaluation sur un jeu de données public d'intrusion*, Travail de Fin de Cycle, Université de Kinshasa, 2025–2026. Dépôt de code : https://github.com/SharoBks/Codes_TFC_Sharon

**Remarque :** les modèles sérialisés (notamment `.joblib` ou `.h5` / `.keras`) doivent être chargés uniquement dans un environnement de confiance.

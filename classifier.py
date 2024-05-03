import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import re

class Classifier():

    # Voert k-fold cross validation uit
    def kfold(self, X, y, clf, labels=None, min_prob=0):
        kf = StratifiedKFold(n_splits=12)
        true = []
        pred = []
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            clf.fit(X.loc[train_index], y.loc[train_index])
            try:
                pred_probs = clf._predict_proba_lr(X.loc[test_index])
            except:
                pred_probs = clf.predict_proba(X.loc[test_index])
            for j in range(len(pred_probs)):
                max_prob = np.argmax(pred_probs[j])
                if pred_probs[j][max_prob] >= min_prob:
                    pred += [clf.classes_[max_prob]]
                    true += [list(y.loc[test_index])[j]]
        print(classification_report(true, pred))
        matrix = pd.DataFrame(confusion_matrix(true, pred))
        matrix = matrix.rename(index={0: labels[0], 1: labels[1], 2: labels[2], 3: labels[3]},
                               columns={0: labels[0], 1: labels[1], 2: labels[2], 3: labels[3]})
        ax = sns.heatmap(matrix, annot=True, xticklabels=True, yticklabels=True, square=True, fmt='g')
        ax.figure.tight_layout()
        plt.show()

    # Bouwt een random forest model
    def random_forest(self, X, y, Z=None, labels=None, min_prob=0):
        parameters = {
            "n_estimators": (15, 20, 25),
            "max_depth": (5, 10, 20, 25),
            "min_samples_split": (2, 4, 6, 8),
            "max_features": ("sqrt", "log2")
        }
        clf = RandomForestClassifier(class_weight="balanced", random_state=42)
        grid = GridSearchCV(clf, parameters)
        grid.fit(X, y)
        clf = grid.best_estimator_
        return clf

    # Bouwt een support vector machine model
    def svm(self, X, y, Z=None, labels=None, min_prob=0):
        parameters = {
            "C": (0.0001, 0.001, 0.01, 0.1, 0.2),
            "tol": (0.5, 0.1, 0.01, 0.001)
        }
        clf = svm.LinearSVC(dual=True, class_weight="balanced", max_iter=10000, random_state=42)
        grid = GridSearchCV(clf, parameters)
        grid.fit(X, y)
        clf = grid.best_estimator_
        return clf

    # Bouwt een model en maakt een voorspelling op de trainingsdata, dan wel de volledige dataset
    def build_predict(self, X, y, Z=None, labels=None, min_prob=0, model="RF"):
        if model == "RF":
            clf = self.random_forest(X, y)
        elif model == "SVM":
            clf = self.svm(X, y)
        X = X[np.intersect1d(Z.columns, X.columns)]
        Z = Z[np.intersect1d(Z.columns, X.columns)]
        if labels is None:
            self.kfold(X, y, clf, labels=y.unique(), min_prob=min_prob)
        else:
            self.kfold(X, y, clf, labels=labels, min_prob=min_prob)
        clf.fit(X, y)
        if model == "RF":
            predicted_probabilities = clf.predict_proba(Z)
        else:
            predicted_probabilities = clf._predict_proba_lr(Z)
        predicted_classes = []
        for i in range(len(predicted_probabilities)):
            max_prob = np.argmax(predicted_probabilities[i])
            if predicted_probabilities[i][max_prob] >= min_prob:
                predicted_classes.append(clf.classes_[max_prob])
            else:
                predicted_classes.append("undefined")
        return predicted_classes

    # Bouwt een rule-based model voor bestuursorganen
    def rule_based(self, dataset):
        organen = []
        for i in range(len(dataset)):
            try:
                pattern = re.search(r'(.*)\nen\n(.*)', dataset[i], re.M)
                orgaan = pattern.group(1)
                if orgaan == '':
                    pattern = re.search(r'(.*)\n\nen\n\n(.*)', dataset[i], re.M)
                    orgaan = pattern.group(1)
                if orgaan.replace('.', '') not in ["de staatssecretaris van Veiligheid en Justitie",
                                  "de staatssecretaris van Justitie en Veiligheid",
                                  "de minister van Veiligheid en Justitie",
                                  "de minister van Sociale zaken en Werkgelegenheid",
                                  "het college van burgemeester en wethouders van Amsterdam",
                                  "het college van burgemeester en wethouders van Rotterdam",
                                  "het college van burgemeester en wethouders van Den Haag",
                                  "het college van burgemeester en wethouders van Utrecht",
                                  "de korpschef van de politie",
                                  "het bestuur van de raad voor rechtsbijstand (hierna: de raad)",
                                  "de Belastingdienst/Toeslagen",
                                  "de directie van het Centraal Bureau Rijvaardigheidsbewijzen (hierna: het CBR)"]:
                    pattern = re.search(r'(.*)\nen\n(.*)', dataset[i], re.M)
                    orgaan = pattern.group(2)
                    if orgaan == '':
                        pattern = re.search(r'(.*)\n\nen\n\n(.*)', dataset[i], re.M)
                        orgaan = pattern.group(2)
            except:
                orgaan = "undefined"
            organen.append(orgaan)
        counts = Counter(organen)
        for i in range(len(organen)):
            if (counts[organen[i]] < 50 or organen[i] == "de staatssecretaris" or organen[i] == 'e'
                    or organen[i] == "het college" or organen[i] == "de minister"
                    or organen[i] == '' or organen[i] == "de raad"
                    or organen[i] == "de burgemeester" or organen[i] == "appellant"):
                organen[i] = "undefined"
            if (organen[i] == "de staatssecretaris van Veiligheid en Justitie"):
                organen[i] = "de staatssecretaris van Justitie en Veiligheid"
            if (organen[i] == "het bestuur van de raad voor rechtsbijstand (hierna: de raad)"):
                organen[i] = "het bestuur van de raad voor rechtsbijstand"
        return organen

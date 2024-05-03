from collector import Collector
from cleaner import Cleaner
from classifier import Classifier
from scipy.stats import chi2_contingency
import pandas as pd

# Verzamelt data
def collect(file):
    collector = Collector()
    ids = collector.get_ids(max=1000, general=True)
    data = collector.ids_to_uitspraken(ids)
    collector.update_csv(file, data)

# Schoont data op
def clean(type):
    cleaner = Cleaner()
    if type is None:
        Z = cleaner.read_data('uitspraken_large.csv')["uitspraak"]
    else:
        Z = cleaner.read_data('uitspraken_large.csv')
        Z, y = cleaner.clean_dataset(Z, type="all")
    if type == "rechtsgebied":
        data = cleaner.read_data('uitspraken_rechtsgebieden.csv')
    else:
        data = cleaner.read_data("uitspraken_small.csv")
    X, y = cleaner.clean_dataset(data, type=type)
    return X, y, Z

# Doet voorspelling op data
def classify(X, y, Z, heatmap_labels=None, model="RF", min_prob=0):
    classifier = Classifier()
    if model == "RF":
        predicted = classifier.build_predict(X, y, Z=Z, labels=heatmap_labels, min_prob=min_prob, model="RF")
    elif model == "SVM":
        predicted = classifier.build_predict(X, y, Z=Z, labels=heatmap_labels, min_prob=min_prob, model="SVM")
    else:
        predicted = classifier.rule_based(Z)
    return predicted

# Voert een chi-kwadraattest uit
def chi_squared(predicted_1, predicted_2):
    c = 0
    for i in range(len(predicted_1)):
        if predicted_1[c] == "undefined" or predicted_2[c] == "undefined":
            predicted_1.pop(c)
            predicted_2.pop(c)
        else:
            c += 1
    crosstab = pd.crosstab(predicted_1, predicted_2, margins=True, margins_name="subtotal",
                           normalize=False)
    print(crosstab.to_string())
    chi, p, dof, expected = chi2_contingency(crosstab)
    print("p-waarde: " + str(p))

if __name__ == '__main__':
    X, y, Z = clean("rechtmatigheid")
    print("rechtmatigheid")
    predicted_rechtmatigheid = classify(X, y, Z, heatmap_labels=["rechtmatig", "onrechtmatig", None, None], model="SVM")

    X, y, Z = clean("rechtsgebied")
    print("rechtsgebied")
    predicted_rechtsgebied = classify(X, y, Z, min_prob=0.3, model="SVM")

    X, y, Z = clean("aangehaalde normen")
    for i in ["zorgvuldigheidsbeginsel", "motiveringsbeginsel", "wabo"]:
        print(i + " aangehaald")
        predicted_aangehaald = classify(X, y[i], Z, heatmap_labels=["niet_aangehaald", "aangehaald", None, None],
                                        model="SVM")

    X, y, Z = clean("geschonden normen")
    for i in ["zorgvuldigheidsbeginsel", "motiveringsbeginsel", "wabo"]:
        print(i + " geschonden")
        predicted_geschonden = classify(X, y[i], Z, heatmap_labels=["niet_geschonden", "geschonden", None, None],
                                        model="SVM")

    X, y, Z = clean(None)
    predicted_orgaan = classify(X, y, Z, model="RB")

    chi_squared(predicted_rechtmatigheid.copy(), predicted_rechtsgebied)
    chi_squared(predicted_rechtmatigheid, predicted_orgaan)
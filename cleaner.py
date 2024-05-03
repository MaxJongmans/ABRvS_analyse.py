import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer

class Cleaner():

    # Laadt het uitsprakenbestand
    def read_data(self, csv):
        data = pd.read_csv(csv)
        return data

    # Verwijdert ongelabelde data uit het uitsprakenbestand
    def delete_non_labelled(self, data):
        data = data.dropna(subset=["onrechtmatig"], ignore_index=True)
        return data

    def get_new_columns(self, data, col):
        column_names = data[col].dropna(ignore_index=True).tolist()
        for i in range(len(column_names)):
            column_names[i] = column_names[i].split(", ")
        column_names = list(set([item for row in column_names for item in row]))
        column_names = pd.DataFrame(columns=column_names)
        return column_names

    # One hot encodes column
    def one_hot_encode(self, data, col):
        new_columns = self.get_new_columns(data, col)
        data = pd.concat([data, new_columns])
        data = data.fillna(0)
        for i in range(len(data)):
            cell = data[col][i]
            if cell != 0:
                cell = cell.split(", ")
                for column_name in cell:
                    data.loc[i, column_name] += 1
        data = data.drop(columns=["id", col])
        return data


    def group_rows(self, data):
        rechtsgebieden = [0, "vreemdelingenrecht", "bouwrecht", "sociale zekerheidsrecht",
                          "milieurecht", "ruimtelijke ordeningsrecht"]
        bestuursorganen = [0, "staatssecretaris van justitie en veiligheid", "gemeenteraad",
                           "minister van sociale zaken en werkgelegenheid", "college van burgemeester en wethouders",
                           "minister van infrastructuur en waterstaat"]
        for i in range(len(data)):
            if data["rechtsgebied"][i] not in rechtsgebieden:
                data.loc[i, "rechtsgebied"] = "overig"
            if data["orgaan"][i] not in bestuursorganen:
                data.loc[i, "orgaan"] = "overig"
        return data

    # Maakt kolom voor rechtmatigheidsoordelen
    def encode_bin(self, data):
        onrechtmatig = []
        for i in range(len(data)):
            if pd.isnull(data.loc[i, "geschonden normen"]):
                onrechtmatig.append(0)
            else:
                onrechtmatig.append(1)
        onrechtmatig = pd.Series(onrechtmatig)
        return onrechtmatig

    # Verwijdert stopwoorden uit uitspraken
    def remove_stop_words(self, data):
        stop_words = set(stopwords.words('dutch'))
        word_tokenizer = RegexpTokenizer(r'\w+')
        for i in range(len(data)):
            try:
                tokens = word_tokenizer.tokenize(data[i])
                filtered = [w for w in tokens if not w.lower() in stop_words]
                data[i] = filtered
            except:
                data[i] = []
        return data

    # Splitst uitspraken op in losse woorden
    def tokenize(self, data):
        data = self.remove_stop_words(data)
        mlb = MultiLabelBinarizer()
        tokens = pd.DataFrame(mlb.fit_transform(data), columns=mlb.classes_, index=data.index)
        data = pd.concat([data, tokens], axis=1)
        data = data.drop(data.columns[0], axis=1)
        return data

    # Schoont de dataset op afhankelijk van gekozen type
    def clean_dataset(self, data, type):
        for c in data:
            data.loc[:, c] = data[c].str.lower()
        if type == "rechtmatigheid":
            data = data[data["aangehaalde normen"].notna()].reset_index()
            labels = self.encode_bin(data)
        elif type == "aangehaalde normen":
            data = data[data["aangehaalde normen"].notna()].reset_index()
            labels = self.one_hot_encode(data, "aangehaalde normen")
        elif type == "geschonden normen":
            data = data[data["aangehaalde normen"].notna()].reset_index()
            labels = self.one_hot_encode(data, "geschonden normen")
        elif type == "rechtsgebied":
            labels = data["rechtsgebied"]
        else:
            labels = None
        data = data["uitspraak"]
        data = self.tokenize(data)
        return data, labels

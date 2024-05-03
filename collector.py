import requests
import pandas as pd
import xmltodict
from bs4 import BeautifulSoup


class Collector:

    # Haalt data op uit de API van de Raad voor de Rechtspraak en verwerkt het
    def request(self, subject, creator, max, last_date):
        url = ("https://data.rechtspraak.nl/uitspraken/zoeken?subject=" + subject + "&creator=" + creator + "&max=" +
               str(max) + "&return=DOC" + "&date=" + last_date + "&date=2023-12-31")
        response = requests.get(url).content
        xml = xmltodict.parse(response)["feed"]["entry"]
        last_date = xml[len(xml) - 1]["title"]["#text"].split(", ")[2]
        last_date = last_date[6:] + last_date[2:6] + last_date[:2]
        ids = pd.DataFrame(xml)["id"]
        return ids, last_date

    # Haalt de ECLI's van uitspraken op via de API van de Raad voor de Rechtspraak
    def get_ids(self, max=1000, general=True, date="2022-01-01"):
        if general == True:
            rechtsgebieden = {"bestuursrecht": "http://psi.rechtspraak.nl/rechtsgebied%23bestuursrecht"}
        else:
            rechtsgebieden = {"europees_bestuursrecht":
                                  "http://psi.rechtspraak.nl/rechtsgebied%23bestuursrecht_europeesBestuursrecht",
                              "omgevingsrecht":
                                  "http://psi.rechtspraak.nl/rechtsgebied%23bestuursrecht_omgevingsrecht",
                              "socialezekerheidsrecht":
                                  "http://psi.rechtspraak.nl/rechtsgebied%23bestuursrecht_socialezekerheidsrecht",
                              "vreemdelingenrecht":
                                  "http://psi.rechtspraak.nl/rechtsgebied%23bestuursrecht_vreemdelingenrecht"}
        creator = "http://standaarden.overheid.nl/owms/terms/Raad_van_State"
        df = pd.DataFrame(columns=["id", "rechtsgebied"])
        for subject in rechtsgebieden:
            last_date = date
            ids, last_date = self.request(rechtsgebieden[subject], creator, max, last_date)
            ids_copy = ids
            while len(ids) == 1000:
                ids, last_date = self.request(rechtsgebieden[subject], creator, max, last_date)
                ids_copy = pd.concat([ids_copy, ids], ignore_index=True, axis=0)
            ids = ids_copy.values.tolist()
            subject = [subject] * len(ids)
            df = pd.concat([df, pd.DataFrame(data={"id": ids, "rechtsgebied": subject})], ignore_index=True,
                           axis=0)
        return df

    # Haalt tektsen en metadata van uitspraken op via de API van de Raad van de Rechtspraak
    def get_uitspraak(self, id, meta=False):
        if meta:
            url = "http://data.rechtspraak.nl/uitspraken/content?return=META&id=" + id
        else:
            url = "http://data.rechtspraak.nl/uitspraken/content?&id=" + id
        response = requests.get(url).content
        soup = BeautifulSoup(response, 'xml')
        try:
            uitspraak = soup.find("uitspraak").text
            procedure = soup.find("psi:procedure").text
        except:
            uitspraak = ""
            procedure = ""
        return uitspraak, procedure

    # Converteert ECLI's naar teksten en metadata
    def ids_to_uitspraken(self, ids):
        uitspraak_list = []
        for i in range(len(ids)):
            uitspraak, procedure = self.get_uitspraak((ids["id"][i]))
            if procedure == "Hoger beroep":
                uitspraak_list.append([ids["id"][i], uitspraak, ids["rechtsgebied"][i]])
        df = pd.DataFrame(data=uitspraak_list, columns=["id", "uitspraak", "rechtsgebied"])
        return df

    # Updatet het uitsprakenbestand
    def update_csv(self, filename, df):
        df.to_csv(filename, index=False)

import json
from collections import Counter
with open(r"C:\Users\ASUS\Documents\UIS\Trabajo_de_grado\Analisis_Bandas_de_soporte\analisis_bandas\features_espectrales.json") as f:
    data = json.load(f)
print(Counter((d["lot_id"], d["class"]) for d in data))
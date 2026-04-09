"""
Algoritmo: Naive Bayes
Dataset: Gripe
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Carregar dados ──────────────────────────────────────────────────────────
url = "https://docs.google.com/spreadsheets/d/1g1aQ61vijh6uHJuc8sijeBEMsoIQ2a5yLwUK04Wptlg/export?format=csv"
df = pd.read_csv(url)

mapping = {
    "Carimbo de data/hora": "timestamp",
    "Você ficou gripado no ano passado ?": "gripe_ano_passado",
    "Você tomou vacina da gripe no ano passado?": "vacina",
    "  Você frequentou no ano passado,  semanalmente ambientes com muitas pessoas? (salas cheias, ônibus, eventos, etc.)  ": "ambientes_cheios",
    "  Você viajou no ano passado mais de 100 km de distância?  ": "viajou",
    "  Você tem alergia nas vias aéreas (rinite, sinusite, etc.)?  ": "alergia",
    "Quantas horas você dormiu em média por noite no ano passado?": "horas_sono",
    "Você praticou atividade física no ano passado?": "exercicio",
    "Você se alimentou de forma balanceada no ano passado?": "alimentacao",
    "Em média, quantas vezes você lavou as mãos por dia no ano passado?": "lavagem_maos",
    "Na sua percepção, o seu nível de estresse no ano passado foi:": "estresse",
}

df = df.rename(columns=mapping).drop(columns=["timestamp"], errors="ignore").dropna()

# ── 2. Codificar variáveis categóricas ────────────────────────────────────────
le = LabelEncoder()
for col in df.select_dtypes(exclude=["number"]).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# ── 3. Separar features e target ──────────────────────────────────────────────
X = df.drop(columns=["gripe_ano_passado"])
y = df["gripe_ano_passado"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ── 4. Treinar Naive Bayes ────────────────────────────────────────────────────
model = CategoricalNB()
model.fit(X_train, y_train)

# ── 5. Avaliar ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print("=== Naive Bayes — Dataset Gripe ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}")
print()
print(classification_report(y_test, y_pred))

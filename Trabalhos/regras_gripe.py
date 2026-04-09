"""
Algoritmo: Regras de Associação (Apriori)
Dataset: Gripe
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

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

# ── 2. Converter cada célula para "coluna=valor" ──────────────────────────────
# Regras de associação trabalham com itens, então transformamos cada valor
# em um item único do tipo "vacina=Sim", "gripe_ano_passado=Sim", etc.
transactions = []
for _, row in df.iterrows():
    items = [f"{col}={val}" for col, val in row.items()]
    transactions.append(items)

# ── 3. Codificar as transações em formato binário ─────────────────────────────
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# ── 4. Minerar itemsets frequentes com Apriori ────────────────────────────────
frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False, inplace=True)

print("=== Regras de Associação (Apriori) — Dataset Gripe ===")
print(f"\nItemsets frequentes encontrados: {len(frequent_itemsets)}")
print(frequent_itemsets.head(10).to_string(index=False))

# ── 5. Gerar regras ───────────────────────────────────────────────────────────
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules = rules.sort_values("lift", ascending=False)

print(f"\nRegras geradas (confiança >= 70%): {len(rules)}")
print("\nTop 10 regras por lift:")
print(
    rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    .head(10)
    .to_string(index=False)
)

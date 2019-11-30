import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import matplotlib.pyplot as plt

original_df = pd.read_csv('prods_pedidos.csv', names=['Pedido','Producto'])
# print(original_df)
q = original_df.Pedido.value_counts().count()

# Pivots data, in order to get one column per product type.
df = original_df.pivot( index='Pedido', columns='Producto', values='Producto')

# ********** IMPORTANT *****************
# Could not remove Pedido col from code so I dit manually from outside.
df = pd.read_csv('Pivoted_without_Pedido_col.csv')
# Dumps new DF to csv

# print(df.head())
items  = df.nunique().keys()

print(f"Q de Pedidos: {q}")
print(f"Uniques Productos: {len(items)}")

productos = []

for item in items:
    productos.append(item)
# print(productos)
# print(len(productos))

# # replace 'NaN' with 0
df = df.fillna(False)

# # replace 'products' with 1
df = df.replace( productos, True)
# print(df.head(10))

df.to_csv('encoded_vals.csv', sep=',')

print("Frequent Items:")
freq_items = apriori(df, min_support=0.15, use_colnames=True, max_len=None, verbose=1, low_memory=False)
print(freq_items.sort_values(by=['support'], ascending=False).head(25))
freq_items.to_csv('freq_items.csv', sep=',')

rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)
print("Association rules:")
print(rules.head(25))

rules.to_csv('rules.csv', sep=',')

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


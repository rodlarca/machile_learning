# Importar librerías necesarias
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Cargar el conjunto de datos de iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Selección de características utilizando el método de chi-cuadrado
chi2_selector = SelectKBest(chi2, k=2)
X_kbest = chi2_selector.fit_transform(X, y)
print(f"Características seleccionadas por chi-cuadrado: {X.columns[chi2_selector.get_support(indices=True)]}")

# Selección de características utilizando Recursive Feature Elimination (RFE)
model = LogisticRegression(max_iter=200)
rfe_selector = RFE(model, n_features_to_select=2, step=1)
rfe_selector = rfe_selector.fit(X, y)
print(f"Características seleccionadas por RFE: {X.columns[rfe_selector.get_support(indices=True)]}")

# Selección de características utilizando Random Forest
forest = RandomForestClassifier(random_state=42)
forest.fit(X, y)
importances = forest.feature_importances_
important_features = X.columns[importances > np.percentile(importances, 75)]
print(f"Características seleccionadas por Random Forest: {important_features}")
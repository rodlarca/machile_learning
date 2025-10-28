from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Crear un Pipeline
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Ajustar el Pipeline a los datos de entrenamiento
pipeline.fit(X_train, y_train)

# Hacer predicciones utilizando el Pipeline
y_pred = pipeline.predict(X_test)
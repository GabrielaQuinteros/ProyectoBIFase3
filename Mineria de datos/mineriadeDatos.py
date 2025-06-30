
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules

# Cargar datos
clientes = pd.read_csv("clientes.csv")
cuentas = pd.read_csv("cuentas.csv")
tarjetas = pd.read_csv("tarjetas.csv")
transacciones = pd.read_csv("transacciones.csv")
riesgo = pd.read_csv("riesgo_crediticio.csv")
sucursales = pd.read_csv("sucursales.csv")
canales = pd.read_csv("canales_digitales.csv")
fechas = pd.read_csv("fechas.csv")

# 1. Análisis Descriptivo
print("Resumen de clientes:")
print(clientes.describe())

# 2. Análisis de Segmentación (Clustering)
X = riesgo[['Score', 'DiasMora']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
riesgo['Cluster'] = kmeans.fit_predict(X_scaled)

# 3. Clasificación (Random Forest para predecir Estado de Mora)
y = riesgo['EstadoMora'].map({'Al día': 0, 'En mora': 1})
X = riesgo[['Score', 'DiasMora']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Reporte de Clasificación:", classification_report(y_test, y_pred))

# 4. Reducción de Dimensionalidad (PCA)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
riesgo['PCA1'] = pca_result[:, 0]
riesgo['PCA2'] = pca_result[:, 1]

# 5. Detección de Valores Atípicos (Z-score)
riesgo['Zscore_Score'] = (riesgo['Score'] - riesgo['Score'].mean()) / riesgo['Score'].std()
outliers = riesgo[riesgo['Zscore_Score'].abs() > 3]
print(f"Número de outliers en Score: {len(outliers)}")

# 6. Asociación (Apriori)
# Preparar datos binarizados por canal
canales_bin = canales.copy()
canales_bin['App'] = canales_bin['Canal'] == 'App'
canales_bin['Web'] = canales_bin['Canal'] == 'Web'
canales_bin['Cajero'] = canales_bin['Canal'] == 'Cajero Virtual'
canales_grp = canales_bin.groupby('ID_Usuario')[['App', 'Web', 'Cajero']].max().reset_index()
frequent_items = apriori(canales_grp[['App', 'Web', 'Cajero']], min_support=0.1, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
print("Reglas de Asociación:", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# 7. Correlación de Variables
cor_matrix = riesgo[['Score', 'DiasMora']].corr()
sns.heatmap(cor_matrix, annot=True)
plt.title("Correlación entre Score y Días de Mora")
plt.savefig("correlacion_score_mora.png")

# 8. Análisis de Distribuciones
sns.histplot(clientes['Edad'], bins=15, kde=True)
plt.title("Distribución de Edad de Clientes")
plt.savefig("distribucion_edad.png")

# Guardar datasets procesados para Power BI
riesgo.to_csv("riesgo_procesado.csv", index=False)
clientes.to_csv("clientes_procesado.csv", index=False)

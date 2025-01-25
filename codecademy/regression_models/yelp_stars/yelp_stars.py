import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# JSON-Dateien laden
businesses = pd.read_json("jsons/yelp_business.json", lines=True)
checkins = pd.read_json("jsons/yelp_checkin.json", lines=True)
photos = pd.read_json("jsons/yelp_photo.json", lines=True)
reviews = pd.read_json("jsons/yelp_review.json", lines=True)
tips = pd.read_json("jsons/yelp_tip.json", lines=True)
users = pd.read_json("jsons/yelp_user.json", lines=True)

# Daten zusammenführen
merged_data = pd.merge(businesses, checkins, on="business_id", how="left")
merged_data = pd.merge(merged_data, photos, on="business_id", how="left")
merged_data = pd.merge(merged_data, reviews, on="business_id", how="left")
merged_data = pd.merge(merged_data, tips, on="business_id", how="left")
merged_data = pd.merge(merged_data, users, on="business_id", how="left")

# Nur numerische und boolesche Spalten behalten
merged_filtered = merged_data.select_dtypes(include=['number', 'bool'])

# Korrelationen berechnen
correlation_matrix = merged_filtered.corr()

target = "stars"
correlation = correlation_matrix[target].sort_values(ascending=False)

# Heatmap erstellen
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Features mit starker Korrelation (Threshold = 0.1)
threshold = 0.05
top_correlated_features = correlation[abs(correlation) > threshold].index.tolist()

# Zielvariable entfernen
if target in top_correlated_features:
    top_correlated_features.remove(target)

# Speichern der wichtigsten Features
pd.DataFrame(top_correlated_features, columns=["Top Features"]).to_csv("../../top_features.csv", index=False)

with open("../../top_features.txt", "w") as f:
    for feature in top_correlated_features:
        f.write(feature + "\n")

print("Top korrelierte Features wurden gespeichert in 'top_features.csv' und 'top_features.txt'")

# Daten in Trainings- und Testdaten aufsteilen sowie bereinigen
x_train, x_test, y_train, y_test = train_test_split(merged_filtered[top_correlated_features], merged_filtered[target],
                                                    test_size=0.2, random_state=1)
x_train = x_train.dropna()
y_train = y_train.loc[x_train.index]  # Die gleichen Zeilen im y-Datensatz behalten

x_test = x_test.dropna()
y_test = y_test.loc[x_test.index]

# Regressionsmodell aufsetzen
model = LinearRegression()
model.fit(x_train, y_train)

# Modell evaluieren
stars_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, stars_pred)
mse = mean_squared_error(y_test, stars_pred)
rmse = np.sqrt(mean_squared_error(y_test, stars_pred))
r2 = r2_score(y_test, stars_pred)

print(f"📊 Modellbewertung:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Visuelle Darstellung der Abweichung
plt.figure(figsize=(8, 6))
plt.scatter(y_test, stars_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="dashed")
plt.xlabel("Echte Sterne-Bewertung")
plt.ylabel("Vorhergesagte Sterne-Bewertung")
plt.title("Modellvorhersagen vs. Tatsächliche Werte")
plt.show()

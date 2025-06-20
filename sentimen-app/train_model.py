import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# 1. Load Dataset
df = pd.read_csv("Tweet_PPKM.csv", delimiter='\t', on_bad_lines='skip')

# 2. Ambil kolom penting dan bersihkan data
df = df[['Tweet', 'sentiment']]
df.dropna(inplace=True)

# 3. Preprocessing teks
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+", "", text)  # hapus URL
    text = re.sub(r"@\w+|#\w+", "", text)  # hapus mention dan hashtag
    text = re.sub(r"[^a-z\s]", "", text)  # hapus angka & simbol
    return text

df['clean_tweet'] = df['Tweet'].apply(clean_text)

# 4. Cek dan seimbangkan dataset
print("Distribusi awal:")
print(df['sentiment'].value_counts())

df_0 = df[df['sentiment'] == 0]  # Negatif
df_1 = df[df['sentiment'] == 1]  # Netral
df_2 = df[df['sentiment'] == 2]  # Positif

min_count = min(len(df_0), len(df_1), len(df_2))

df_balanced = pd.concat([
    df_0.sample(min_count, random_state=42),
    df_1.sample(min_count, random_state=42),
    df_2.sample(min_count, random_state=42)
])

print("\nSetelah penyeimbangan:")
print(df_balanced['sentiment'].value_counts())

# 5. Split Data
X = df_balanced['clean_tweet']
y = df_balanced['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Vektorisasi Teks
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# 8. Evaluasi
print("\nAkurasi:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Visualisasi Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix")
plt.show()

# 10. Simpan model dan vectorizer
with open("logistic_regression.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

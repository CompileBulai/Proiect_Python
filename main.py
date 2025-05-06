import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier  # Importăm RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# Încărcare set de date
df = pd.read_csv('IMDB Dataset.csv')

# Afișare primele 5 rânduri
print(df.head())

# Verificare informațiilor despre setul de date
print(df.info())

# Verificarea distribuție etichete
print(df['sentiment'].value_counts())


# Funcție de preprocesare a textului
def preprocess_text(text):
    # Transformare text în litere mici
    text = text.lower()
    # Eliminare caractere speciale și cifre
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenizare text
    tokens = word_tokenize(text)
    # Eliminare cuvinte de legătură
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lematizare cuvinte
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# preprocesarea asupra coloanei de recenzii
df['clean_review'] = df['review'].apply(preprocess_text)

# Vectorizare text folosind TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review']).toarray()
print(X)

# Transformă etichetele în valori numerice
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Aplică PCA pentru a reduce dimensionalitatea la 100 de componente


# Vizualizează variația explicată de fiecare componentă principală
plt.plot(range(1, 101), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('Numărul de componente')
plt.ylabel('Variația explicată cumulată')
plt.title('Variația explicată de componentele principale')
plt.show()

# Împărțire date în seturi de antrenament și test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pca = PCA(n_components=100)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced  = pca.transform(X_test)

# model SVM pe datele originale
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# performanța SVM
print("Performanța pe datele originale (SVM):")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# model SVM pe datele cu dimensionalitate redusă
svm_reduced = SVC(kernel='linear')
svm_reduced.fit(X_train_reduced, y_train)
y_pred_reduced = svm_reduced.predict(X_test_reduced)

# performanța SVM pe datele reduse
print("Performanța pe datele cu dimensionalitate redusă (SVM):")
print(confusion_matrix(y_test, y_pred_reduced))
print(classification_report(y_test, y_pred_reduced))

# model Random Forest pe datele originale
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# performanța Random Forest
print("Performanța pe datele originale (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# model Random Forest pe datele cu dimensionalitate redusă
rf_reduced = RandomForestClassifier(n_estimators=100)
rf_reduced.fit(X_train_reduced, y_train)
y_pred_rf_reduced = rf_reduced.predict(X_test_reduced)

# performanța Random Forest pe datele reduse
print("Performanța pe datele cu dimensionalitate redusă (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf_reduced))
print(classification_report(y_test, y_pred_rf_reduced))

# acuratețea modelelor
accuracy_svm = svm.score(X_test, y_test)
accuracy_svm_reduced = svm_reduced.score(X_test_reduced, y_test)
accuracy_rf = rf.score(X_test, y_test)
accuracy_rf_reduced = rf_reduced.score(X_test_reduced, y_test)

print(f"Acuratețea pe datele originale (SVM): {accuracy_svm:.2f}")
print(f"Acuratețea pe datele cu dimensionalitate redusă (SVM): {accuracy_svm_reduced:.2f}")
print(f"Acuratețea pe datele originale (Random Forest): {accuracy_rf:.2f}")
print(f"Acuratețea pe datele cu dimensionalitate redusă (Random Forest): {accuracy_rf_reduced:.2f}")
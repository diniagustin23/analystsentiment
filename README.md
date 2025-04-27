Import package and the dataset
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Reshape
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.optimizers import Adam
!pip install imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
# Read CSV into a DataFrame
df = pd.read_csv('datafix.csv')

# Display the first few rows of the DataFrame to check if the data was loaded correctly
df.head()
# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Alternatively, fill missing values with a specific value
# df = df.fillna('your_value')
# Check for and remove duplicates
print("Number of duplicates:", df.duplicated().sum())
df = df.drop_duplicates()
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download data NLTK untuk proses stemming
import nltk
nltk.download('punkt')

# Fungsi untuk membersihkan, melakukan tokenisasi, dan stemming pada teks
def clean_tokenize_and_stem(text):
    # Hapus emoji dan simbol
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    # Hapus karakter newline
    cleaned_text = cleaned_text.replace('\n', '')
    
    # Lakukan tokenisasi pada teks
    words = word_tokenize(cleaned_text)
    
    # Inisialisasi Porter Stemmer
    stemmer = PorterStemmer()
    
    # Terapkan proses stemming pada setiap kata
    stemmed_words = [stemmer.stem(word) for word in words]
    
    # Gabungkan kata-kata yang telah di-stemming menjadi satu kalimat
    stemmed_text = ' '.join(stemmed_words)
    
    return stemmed_text

# Terapkan pembersihan, tokenisasi, dan stemming pada kolom "Tweet"
df['Tweet'] = df['Tweet'].apply(clean_tokenize_and_stem)

# Terapkan pembersihan, tokenisasi, dan stemming pada kolom "Nama"
df['Nama'] = df['Nama '].apply(clean_tokenize_and_stem)

# Assuming df is your DataFrame
class_distribution = df['Label'].value_counts()

print("Class Distribution:")
print(class_distribution)
# Assume df is your cleaned DataFrame

# Text Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Tweet'])
sequences = tokenizer.texts_to_sequences(df['Tweet'])
max_len = 100  # Adjust as needed
padded_sequences = pad_sequences(sequences, maxlen=max_len)
# One-hot encode the labels
labels = to_categorical(df['Label'], num_classes=3)  # assuming 3 classes
# Build and Compile the CNN Model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(Conv1D(256, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # 3 classes for contra, neutral, and pro

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the Model
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42, stratify=labels)
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))
# Evaluate the Model (optional)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
# Make Predictions (on new data)
new_texts = ["Formulae bagus banget!", "Saya tidak setuju dengan formulae"]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_len)

predictions = model.predict(new_padded_sequences)
print("Predicted Probabilities:", predictions)
predicted_labels = np.argmax(predictions, axis=1)
print("Predicted Labels:", predicted_labels)
class_mapping = {0: 'pro', 1: 'neutral', 2: 'contra'}
predicted_labels_mapped = [class_mapping[label] for label in predicted_labels]
print("Mapped Predicted Labels:", predicted_labels_mapped)
df_oversampling = df.copy()
# Assume df_oversampling is your cleaned DataFrame

# Text Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_oversampling['Tweet'])
sequences = tokenizer.texts_to_sequences(df_oversampling['Tweet'])
max_len = 100  # Adjust as needed
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# One-hot encode the labels
labels = to_categorical(df_oversampling['Label'], num_classes=3)  # assuming 3 classes

# Train the Model
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42, stratify=labels)

# Assuming X_train and y_train are your training data and labels
rus = RandomOverSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# Build and Compile the CNN Model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(Conv1D(256, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # 3 classes for contra, neutral, and pro

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model with Undersampled Data
model.fit(X_resampled, y_resampled, epochs=10, batch_size=32, validation_data=(X_test, y_test))
# Evaluate the Model (optional)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
print(len(y_resampled))
print(len(X_resampled))
# Assuming y_resampled is your oversampled labels
label_counts = np.sum(y_resampled, axis=0)

print("Label Counts in Oversampled Data:")
print("Class 0:", label_counts[0])
print("Class 1:", label_counts[1])
print("Class 2:", label_counts[2])
# Make Predictions (on new data)
new_texts = ["Formulae bagus banget!", "Saya tidak setuju dengan formulae"]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_len)

predictions = model.predict(new_padded_sequences)
print("Predicted Probabilities:", predictions)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Predict the classes
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate and plot the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

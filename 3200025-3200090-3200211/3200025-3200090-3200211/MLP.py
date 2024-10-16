import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Ορίζουμε τον αριθμό των λέξεων που θέλουμε να κρατήσουμε
num_words = 1000

# Φορτώνουμε τα δεδομένα IMDB
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

# Εφαρμόζουμε zero-padding στα δεδομένα εισόδου
X_train_padded = pad_sequences(X_train, maxlen=100)
X_test_padded = pad_sequences(X_test, maxlen=100)

# Υλοποιούμε Μοντέλο Πολλαπλών Επιπέδων (MLP) με Ενσωματωμένα Λεξικά

# Δημιουργούμε ένα νέο μοντέλο Sequential το οποίο θα περιέχει τα επόμενα επίπεδα
# Oι διαφορετικές στοίβες ή επίπεδα νευρώνων συνδέονται σειριακά
# , με τα δεδομένα να περνάνε από κάθε επίπεδο μία φορά
mlp_model = Sequential()
# Προσθέτουμε ένα επίπεδο ενσωμάτωσης (embedding) για τη μετατροπή των ακολουθιών λέξεων σε αναπαραστάσεις διανύσματος
# Το επίπεδο ενσωμάτωσης θα έχει διάσταση εξόδου (output_dim) 100, όπως και μήκος εισόδου (input_length) 100
mlp_model.add(Embedding(input_dim=num_words, output_dim=32, input_length=100))
# Προσθέτουμε ένα επίπεδο Flatten για να μετατρέψουμε τις διαστάσεις του διανύσματος ενσωμάτωσης σε μια διάνυσμα επίπεδου
mlp_model.add(Flatten())
# Προσθέτουμε ένα πλήρως συνδεδεμένο επίπεδο (Dense) με 64 νευρώνες και συνάρτηση ενεργοποίησης ReLU
mlp_model.add(Dense(64, activation='relu'))
# Προσθέτουμε ένα πλήρως συνδεδεμένο επίπεδο (Dense) με ένα νευρώνα και συνάρτηση ενεργοποίησης sigmoid (για δυαδική ταξινόμηση)
mlp_model.add(Dense(1, activation='sigmoid'))
# Μεταγλωττίζουμε το μοντέλο χρησιμοποιώντας τον βελτιστοποιητή Adam, συνάρτηση απώλειας binary crossentropy και μετρικές ακρίβειας
mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Εκπαιδεύουμε το μοντέλο
mlp_history = mlp_model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_split=0.5)

# Υπολογισμός ακρίβειας για τα σύνολα train και test για το MLP
mlp_train_accuracy = mlp_model.evaluate(X_train_padded, y_train)[1]
mlp_test_accuracy = mlp_model.evaluate(X_test_padded, y_test)[1]
print("MLP Train Accuracy:", mlp_train_accuracy)
print("MLP Test Accuracy:", mlp_test_accuracy)

# Προβλέπουμε τις κλάσεις για τα δεδομένα εκπαίδευσης
y_train_pred = (mlp_model.predict(X_train_padded) > 0.5).astype(int)
# Υπολογίζουμε το classification report για τα δεδομένα εκπαίδευσης
print("Classification Report for Training Data:")
print(classification_report(y_train, y_train_pred))

# Προβλέπουμε τις κλάσεις για τα δεδομένα ελέγχου
y_test_pred = (mlp_model.predict(X_test_padded) > 0.5).astype(int)
# Υπολογίζουμε το classification report για τα δεδομένα ελέγχου
print("Classification Report for Testing Data:")
print(classification_report(y_test, y_test_pred))

# Κατασκευάζουμε καμπύλες εκπαίδευσης
plt.figure(figsize=(12, 6))

# Καμπύλες MLP
plt.subplot(1, 2, 1)
plt.plot(mlp_history.history['accuracy'], label='MLP Training Accuracy')
plt.plot(mlp_history.history['val_accuracy'], label='MLP Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('MLP Training and Validation Accuracy')
plt.legend()

plt.show()

# Κατασκευή καμπύλων σφάλματος (loss) για MLP
plt.figure(figsize=(12, 6))

# Καμπύλες MLP
plt.subplot(1, 2, 1)
plt.plot(mlp_history.history['loss'], label='MLP Training Loss')
plt.plot(mlp_history.history['val_loss'], label='MLP Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLP Training and Validation Loss')
plt.legend()

plt.show()

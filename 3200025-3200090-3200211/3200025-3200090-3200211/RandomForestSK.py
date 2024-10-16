from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
import time

# Load IMDB dataset
from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=1000)


def preprocess_text(text):
    if isinstance(text, list):
        text = ' '.join([str(word) for word in text])
    return text.lower()


# Υπερπαράμετροι
m = 1000
n = 0.9
k = 50
n_estimators = 10
max_depth = 3

# Μετατροπή των δεδομένων σε κείμενο
X_train_text = [preprocess_text(sample) for sample in X_train]
X_test_text = [preprocess_text(sample) for sample in X_test]

vectorizer = CountVectorizer(max_features=m, max_df=n, min_df=k, binary=True)
X_train_binary = vectorizer.fit_transform(X_train_text).toarray()
X_test_binary = vectorizer.transform(X_test_text).toarray()

# Εκπαίδευση του Random Forest Classifier
random_forest_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
random_forest_classifier.fit(X_train_binary, y_train)

# Προβλέψεις
y_train_pred_rf = random_forest_classifier.predict(X_train_binary)
y_test_pred_rf = random_forest_classifier.predict(X_test_binary)


# Εκτύπωση των μετρικών
print(f"Random Forest - Train Accuracy: {accuracy_score(y_train, y_train_pred_rf):.5f}")
print(f"Random Forest - Test Accuracy: {accuracy_score(y_test, y_test_pred_rf):.5f}")

# Εκτύπωση του Classification Report
classification_report_str = classification_report(y_train, y_train_pred_rf)
print("Random Forest - Classification Report(train):")
print(classification_report_str)

classification_report_str = classification_report(y_test, y_test_pred_rf)
print("Random Forest - Classification Report(test):")
print(classification_report_str)

# Λίστες για την αποθήκευση των μετρικών
# Εκπαιδεύει έναν ταξινομητή Random Forest σε διάφορα μεγέθη υποσυνόλων των δεδομένων εκπαίδευσης και
# στη συνέχεια αξιολογεί την απόδοσή του στα σύνολα εκπαίδευσης και δοκιμής.
# Οι μετρικές (ακρίβεια, ακρίβεια, ανάκληση και F1 score) για κάθε μέγεθος υποσυνόλου αποθηκεύονται σε λίστες.
# Αυτός ο κώδικας αξολογεί πόσο καλά αποδίδει ο μοντέλο Random Forest σε διάφορα μεγέθη δεδομένων εκπαίδευσης.
# Βοηθά στην κατανόηση του πώς γενικεύει το μοντέλο καθώς το μέγεθος του συνόλου εκπαίδευσης αυξάνεται.

train_accuracies = []

# Αξιολόγηση απόδοσης του Random Forest Classifier σε διάφορα μεγέθη εκπαίδευσης
train_accuracies = []
test_accuracies = []
precisions = []
recalls = []
f1_scores = []

training_sizes = [100, 500, 1000, 2000, 5000, 10000, 25000]

for size in training_sizes:
    start_time = time.time()
    random_forest_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    random_forest_classifier.fit(X_train_binary[:size], y_train[:size])

    y_train_pred_rf = random_forest_classifier.predict(X_train_binary[:size])
    y_test_pred_rf = random_forest_classifier.predict(X_test_binary[:size])

    train_accuracy = accuracy_score(y_train[:size], y_train_pred_rf)
    test_accuracy = accuracy_score(y_test[:size], y_test_pred_rf)
    precision = precision_score(y_test[:size], y_test_pred_rf)
    recall = recall_score(y_test[:size], y_test_pred_rf)
    f1 = f1_score(y_test[:size], y_test_pred_rf)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Training size: {size}, Runtime: {runtime} seconds")

# Σχεδίαση καμπύλων μάθησης
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, train_accuracies, label='Training Accuracy')
plt.plot(training_sizes, test_accuracies, label='Test Accuracy')
plt.xlabel('Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves:RFsk')
plt.legend()
plt.show()

# Σχεδίαση πινάκων μετρικών απόδοσης
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, precisions, label='Precision')
plt.plot(training_sizes, recalls, label='Recall')
plt.plot(training_sizes, f1_scores, label='F1 Score')
plt.xlabel('Size')
plt.ylabel('Score')
plt.title('Performance Metrics:RFsk')
plt.legend()
plt.show()
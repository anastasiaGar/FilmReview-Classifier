import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from keras.datasets import imdb


# Ορισμός μιας συνάρτησης για προεπεξεργασία
def preprocess_text(text):
    if isinstance(text, list):
        text = ' '.join([str(word) for word in text])
    return text.lower()


# Φόρτωση των δεδομένων IMDB
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)


# Μετατροπή των δεδομένων
X_train_text = [preprocess_text(sample) for sample in x_train]
X_test_text = [preprocess_text(sample) for sample in x_test]


# Υπερπαράμετροι για την επιλογή των λέξεων
m = 1000  # m most frequent words to include
n = 0.9  # n most frequent words to skip.Words that occur in more than 90% of the documents will be excluded.
k = 50   # k most rare words to skip


vectorizer = CountVectorizer(max_features=m, max_df=n, min_df=k, binary=True)

# Fit και transform για τα training data
X_train_binary = vectorizer.fit_transform(X_train_text)

# Transform για τα testing data
X_test_binary = vectorizer.transform(X_test_text)


class CustomNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_probs = None
        self.feature_probs = None

    def fit(self, x, y):
        num_samples, num_features = x.shape
        self.classes = np.unique(y)
        len(self.classes)

        # Calculate class probabilities
        self.class_probs = {c: np.sum(y == c) / num_samples for c in self.classes}

        # Calculate feature probabilities
        self.feature_probs = {}
        for c in self.classes:
            class_indices = np.where(y == c)
            class_data = x[class_indices]
            self.feature_probs[c] = np.sum(class_data, axis=0) / np.sum(class_data)

    def predict(self, x):
        predictions = []
        for sample in x:
            class_scores = {c: np.log(self.class_probs[c]) + np.sum(np.log(self.feature_probs[c][sample > 0] + 1e-10))
                            for c in self.classes}
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        return np.array(predictions)


# Εκπαίδευση του Custom Naive Bayes Classifier στο σύνολο εκπαίδευσης
custom_nb_classifier = CustomNaiveBayes()
custom_nb_classifier.fit(X_train_binary.toarray(), y_train)

# Προβλέψεις στο σύνολο εκπαίδευσης
y_train_dev_pred_custom_nb = custom_nb_classifier.predict(X_train_binary.toarray())

# Προβλέψεις στο σύνολο ελέγχου
y_test_dev_pred_custom_nb = custom_nb_classifier.predict(X_test_binary.toarray())

# Αποτελέσματα accuracy για δεδομένα εκπαίδευσης και ελέγχου
print(f"Custom Naive Bayes - Train Accuracy: {accuracy_score(y_train, y_train_dev_pred_custom_nb):.5f}")
print(f"Custom Naive Bayes - Test Accuracy: {accuracy_score(y_test, y_test_dev_pred_custom_nb):.5f}")


# Εκτύπωση του Classification Report
classification_report_nb = classification_report(y_train, y_train_dev_pred_custom_nb)
print("Custom Naive Bayes - Classification Report (train data):")
print(classification_report_nb)

# Εκτύπωση του Classification Report
classification_report_nb = classification_report(y_test, y_test_dev_pred_custom_nb)
print("Custom Naive Bayes - Classification Report (test data):")
print(classification_report_nb)

# Λίστες για την αποθήκευση των μετρικών
train_accuracies_custom_nb = []
test_accuracies_custom_nb = []
precisions_custom_nb = []
recalls_custom_nb = []
f1_scores_custom_nb = []

# Διαφορετικά μεγέθη υποσυνόλων εκπαίδευσης
sizes = [100, 500, 1000, 2000, 5000, 10000, 25000]

for size in sizes:
    # Εκπαίδευση του Custom Naive Bayes Classifier στο υποσύνολο
    custom_nb_classifier = CustomNaiveBayes()
    custom_nb_classifier.fit(X_train_binary[:size].toarray(), y_train[:size])

    # Προβλέψεις στο σύνολο εκπαίδευσης
    y_train_pred_custom_nb = custom_nb_classifier.predict(X_train_binary[:size].toarray())

    # Προβλέψεις στο σύνολο ελέγχου
    y_test_pred_custom_nb = custom_nb_classifier.predict(X_test_binary[:size].toarray())

    # Υπολογισμός μετρικών
    train_accuracy_custom_nb = accuracy_score(y_train[:size], y_train_pred_custom_nb)
    test_accuracy_custom_nb = accuracy_score(y_test[:size], y_test_pred_custom_nb)
    precision_custom_nb = precision_score(y_test[:size], y_test_pred_custom_nb)
    recall_custom_nb = recall_score(y_test[:size], y_test_pred_custom_nb)
    f1_custom_nb = f1_score(y_test[:size], y_test_pred_custom_nb)

    # Αποθήκευση μετρικών στις λίστες
    train_accuracies_custom_nb.append(train_accuracy_custom_nb)
    test_accuracies_custom_nb.append(test_accuracy_custom_nb)
    precisions_custom_nb.append(precision_custom_nb)
    recalls_custom_nb.append(recall_custom_nb)
    f1_scores_custom_nb.append(f1_custom_nb)

# Σχεδίαση καμπύλων μάθησης
plt.figure(figsize=(10, 6))
plt.plot(sizes, train_accuracies_custom_nb, label='Training Accuracy (Custom Naive Bayes)')
plt.plot(sizes, test_accuracies_custom_nb, label='Test Accuracy (Custom Naive Bayes)')
plt.xlabel('Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves (Custom Naive Bayes)')
plt.legend()
plt.show()

# Σχεδίαση καμπυλών μετρικών απόδοσης
plt.figure(figsize=(10, 6))
plt.plot(sizes, precisions_custom_nb, label='Precision (Custom Naive Bayes)')
plt.plot(sizes, recalls_custom_nb, label='Recall (Custom Naive Bayes)')
plt.plot(sizes, f1_scores_custom_nb, label='F1 Score (Custom Naive Bayes)')
plt.xlabel('Size')
plt.ylabel('Score')
plt.title('Performance Metrics / Test data (Custom Naive Bayes)')
plt.legend()
plt.show()

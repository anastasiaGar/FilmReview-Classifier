from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from keras.datasets import imdb


# Ορισμός μιας συνάρτησης για προεπεξεργασία των κειμένων
def preprocess_text(text):
    if isinstance(text, list):
        text = ' '.join([str(word) for word in text])
    return text.lower()


# Φόρτωση των δεδομένων IMDB
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=1000)

# Μετατροπή των δεδομένων σε κείμενο
X_train_text = [preprocess_text(sample) for sample in X_train]
X_test_text = [preprocess_text(sample) for sample in X_test]

# Υπερπαράμετροι για την επιλογή των λέξεων
m = 1000  # m most frequent words to include
n = 0.9  # n most frequent words to skip.Words that occur in more than 90% # of the documents will be excluded.
k = 50   # k most rare words to skip

# Initialize CountVectorizer with custom parameters
vectorizer = CountVectorizer(max_features=m, max_df=n, min_df=k, binary=True)

# Fit and transform the training data
X_train_binary = vectorizer.fit_transform(X_train_text)

# Transform the testing data
X_test_binary = vectorizer.transform(X_test_text)


# Εκπαίδευση του Naive Bayes Classifier στο σύνολο ανάπτυξης
naive_bayes_classifier = BernoulliNB()
naive_bayes_classifier.fit(X_train_binary.toarray(), y_train)

# Προβλέψεις στο σύνολο ανάπτυξης
y_train_pred_nb = naive_bayes_classifier.predict(X_train_binary.toarray())

# Προβλέψεις στο σύνολο ελέγχου
y_test_pred_nb = naive_bayes_classifier.predict(X_test_binary.toarray())

# Αποτελέσματα στο σύνολο εκπαίδευσης
print(f"Custom Naive Bayes - Train Accuracy: {accuracy_score(y_train, y_train_pred_nb):.5f}")
print(f"Custom Naive Bayes - Test Accuracy: {accuracy_score(y_test, y_test_pred_nb):.5f}")


# Εκτύπωση του Classification Report
classification_report_str_custom_nb = classification_report(y_train, y_train_pred_nb)
print("Custom Naive Bayes - Classification Report (train data):")
print(classification_report_str_custom_nb)

# Εκτύπωση του Classification Report
classification_report_str_custom_nb = classification_report(y_test, y_test_pred_nb)
print("Custom Naive Bayes - Classification Report (test data):")
print(classification_report_str_custom_nb)

# Λίστες για την αποθήκευση των μετρικών
train_accuracies_nb = []
test_accuracies_nb = []
precisions_nb = []
recalls_nb = []
f1_scores_nb = []

# Διαφορετικά μεγέθη υποσυνόλων εκπαίδευσης
sizes = [100, 500, 1000, 2000, 5000, 10000, 25000]

for size in sizes:
    # Εκπαίδευση του Custom Naive Bayes Classifier στο υποσύνολο
    nb_classifier = BernoulliNB()
    nb_classifier.fit(X_train_binary[:size].toarray(), y_train[:size])

    # Προβλέψεις στο σύνολο εκπαίδευσης
    y_train_pred_dev_nb = nb_classifier.predict(X_train_binary[:size].toarray())

    # Προβλέψεις στο σύνολο ελέγχου
    y_test_pred_dev_nb = nb_classifier.predict(X_test_binary[:size].toarray())

    # Υπολογισμός μετρικών
    train_accuracy_custom_nb = accuracy_score(y_train[:size], y_train_pred_dev_nb)
    test_accuracy_custom_nb = accuracy_score(y_test[:size], y_test_pred_dev_nb)
    precision_custom_nb = precision_score(y_test[:size], y_test_pred_dev_nb)
    recall_custom_nb = recall_score(y_test[:size], y_test_pred_dev_nb)
    f1_custom_nb = f1_score(y_test[:size], y_test_pred_dev_nb)

    # Αποθήκευση μετρικών στις λίστες
    train_accuracies_nb.append(train_accuracy_custom_nb)
    test_accuracies_nb.append(test_accuracy_custom_nb)
    precisions_nb.append(precision_custom_nb)
    recalls_nb.append(recall_custom_nb)
    f1_scores_nb.append(f1_custom_nb)

# Σχεδίαση καμπύλων μάθησης
plt.figure(figsize=(10, 6))
plt.plot(sizes, train_accuracies_nb, label='Training Accuracy (Custom Naive Bayes)')
plt.plot(sizes, test_accuracies_nb, label='Test Accuracy (Custom Naive Bayes)')
plt.xlabel('Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves (Custom Naive Bayes)')
plt.legend()
plt.show()

# Σχεδίαση πινάκων μετρικών απόδοσης
plt.figure(figsize=(10, 6))
plt.plot(sizes, precisions_nb, label='Precision (Custom Naive Bayes)')
plt.plot(sizes, recalls_nb, label='Recall (Custom Naive Bayes)')
plt.plot(sizes, f1_scores_nb, label='F1 Score (Custom Naive Bayes)')
plt.xlabel('Size')
plt.ylabel('Score')
plt.title('Performance Metrics / Test data (Custom Naive Bayes)')
plt.legend()
plt.show()

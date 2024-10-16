import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import time
import tensorflow as tf
from collections import Counter
import math

# Αρχικοποίηση ενός κόμβου στο δέντρο απόφασης
class Node:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):

        self.feature_index = feature_index  # Δείκτης του χαρακτηριστικού(λεξη) για διαίρεση
        self.threshold = threshold  # Κατώτατο όριο για διαίρεση
        self.value = value  # Ετικέτα κλάσης για φύλλο(0 ή 1)
        self.left = left  # Αριστερό υποδέντρο
        self.right = right  # Δεξί υποδέντρο

class ID3DecisionTree:
    # Αρχικοποίηση του δέντρου ID3
    def __init__(self, max_depth):

        self.max_depth = max_depth
        self.tree = None

    # Εκπαίδευση του δέντρου απόφασης στα δεδομένα εκπαίδευσης(κληση τις _fit)
    def fit(self, X, y):

        self.tree = self._fit(X, y, depth=0)

    # Ιδιωτική μεθοδος για αναδρομική κατασκευή του δέντρου απόφασης
    def _fit(self, X, y, depth):

        # εξάγει τις διαστάσεις του πίνακα X και τις αποθηκεύει στις μεταβλητές num_samples και num_features.
        # Εδώ, num_samples αναφέρεται στον αριθμό των δειγμάτων, και num_features αναφέρεται στον αριθμό των χαρακτηριστικών.
        num_samples, num_features = X.shape
        # unique_classes : oι μοναδικές τιμές που περιέχονται στον πίνακα y, counts : αριθμός των εμφανίσεών τους.
        unique_classes, counts = np.unique(y, return_counts=True) #υπολογίζει τις μοναδικές κατηγορίες (unique_classes) και τον αριθμό των εμφανίσεών τους (counts) στον πίνακα y.
        majority_class = unique_classes[np.argmax(counts)] #επιλέγει την κατηγορία με τον υψηλότερο αριθμό εμφανίσεων

        # Εάν ο κόμβος είναι καθαρός ή έχει φτάσει το μέγιστο βάθος, δημιουργήστε φύλλο
        if len(unique_classes) == 1 or depth == self.max_depth:
            return Node(value=majority_class)

        # Βρείτε τον καλύτερο διαχωρισμό
        best_feature, best_threshold = self._find_best_split(X, y)

        # Εάν δεν βρεθεί διαχωρισμός, δημιουργήστε φύλλο
        if best_feature is None:
            return Node(value=majority_class)

        # Διαίρεση των δεδομένων

        # Δημιουργεί μια μάσκα (boolean array) όπου οι θέσεις που ικανοποιούν τη συνθήκη
        # (το χαρακτηριστικό είναι μικρότερο ή ίσο με το καλύτερο κατώφλι) έχουν τιμή True, ενώ οι υπόλοιπες θέσεις έχουν τιμή False.
        # Αυτό δημιουργεί το αριστερό υποσύνολο των δεδομένων
        left_mask = X[:, best_feature] <= best_threshold
        #  Δημιουργεί το δεξί υποσύνολο των δεδομένων αντιστρέφοντας τη μάσκα.
        #  Δηλαδή, όπου η μάσκα έχει τιμή True, το right_mask θα έχει τιμή False, και αντιστρόφως
        right_mask = ~left_mask

        # Αναδρομική κατασκευή των υποδέντρων
        left_subtree = self._fit(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._fit(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_index=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    # Βρισκει τον καλύτερο διαχωρισμό για το δέντρο απόφασης με βαση το IG
    def _find_best_split(self, X, y):

        num_samples, num_features = X.shape
        best_feature = None
        best_threshold = None
        best_ig = -1  # Αρχικοποίηση με αρνητική τιμή, καθώς θέλουμε να μεγιστοποιήσουμε το IG

        # έλεγχο για κάθε δυνατή τιμή κατωφλίου και κάθε χαρακτηριστικού, ώστε να εντοπίσει τον καλύτερο διαχωρισμό

        for feature_index in range(num_features):
            feature_values = np.unique(X[:, feature_index])
            for threshold in feature_values:
                #Δημιουργία μάσκας ανάλογα με το όρισμα
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                ig = self.calculate_ig(y, X[:, feature_index])

                if ig > best_ig:
                    best_ig = ig
                    best_feature = feature_index
                    best_threshold = threshold


        return best_feature, best_threshold #τιμές που επιτυγχάνουν τον καλύτερο διαχωρισμό με βαση το IG

    # Υπολογισμός Κέρδους Πληροφορίας για το δέντρο απόφασης
    def calculate_ig(self, classes_vector, feature):

        classes = np.unique(classes_vector)
        num_samples = len(classes_vector)

        # Υπολογισμός Αρχικού Εντροπικού Κόστους (HC)
        HC = -np.sum((np.array([np.sum(classes_vector == c) for c in classes]) / num_samples) * np.log2(
            np.array([np.sum(classes_vector == c) for c in classes]) / num_samples))

        feature_values, feature_counts = np.unique(feature, return_counts=True)
        HC_feature = 0  # Αρχικοποίηση του Εντροπικού Κόστους για το Χαρακτηριστικό (HC_feature)

        # Υπολογισμός Εντροπίας για κάθε τιμή του Χαρακτηριστικού
        for value, count in zip(feature_values, feature_counts):
            pf = count / num_samples  # Υπολογισμός της P(X=x)
            indices = np.where(feature == value)[0]  # Εντοπισμός των γραμμών όπου X=x

            classes_of_feat = classes_vector[indices]  # Κατηγορία παραδειγμάτων που αντιστοιχούν στα παραπάνω δείγματα
            pcf = np.array(
                [np.sum(classes_of_feat == c) / len(classes_of_feat) for c in classes])  # Υπολογισμός της P(C=c|X=x)

            # Υπολογισμός του Εντροπικού Κόστους για το Χαρακτηριστικό (HC_feature)
            temp_H = -pf * pcf * np.nan_to_num(np.log2(pcf + 1e-10),
                                               nan=0)  # Προσθήκη μικρού epsilon (1e-10) για αποφυγή διαίρεσης με το μηδέν
            HC_feature += np.sum(temp_H)

        # Υπολογισμός Κέρδους Πληροφορίας
        ig = HC - HC_feature
        return ig

    def predict(self, X):
        # Κάντε προβλέψεις χρησιμοποιώντας το δέντρο απόφασης
        return np.array([self._predict_tree(x, self.tree) for x in X])

    # Για κάθε δείγμα, καλείται η ιδιωτική μέθοδος _predict_tree
    # Αναδρομική διάσχιση του δέντρου απόφασης για προβλέψεις
    def _predict_tree(self, x, node):

        # Αν ο κόμβος είναι φύλλο, επιστρέφεται η κλάση του φύλλου
        if node.value is not None:
            return node.value

        # Ανάλογα με τη συνθήκη χωρισμού, καλείται η μέθοδος _predict_tree για το αριστερό ή δεξί υποδέντρο
        if x[node.feature_index] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)


class myRandomForestClassifier:
    def __init__(self, n_estimators, max_depth=None):
        # Αρχικοποίηση του ταξινομητή τυχαίων δασών
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    # Εκπαίδευση του ταξινομητή τυχαίων δασών
    def fit(self, X, y):


        # Δημιουργία νέου δέντρου ID3DecisionTree για κάθε επανάληψη
        for i in range(self.n_estimators):
            tree = ID3DecisionTree(max_depth=self.max_depth)  # Χρήση της κλάσης ID3DecisionTree
            tree.fit(X, y)  # Εκπαίδευση του δέντρου στα δεδομένα εισόδου
            self.trees.append(tree)  # Προσθήκη του εκπαιδευμένου δέντρου στο δάσος

    # Προβλέψεις χρησιμοποιώντας τον ταξινομητή RF
    def predict(self, X):

        # Έλεγχος αν τα δεδομένα εισόδου είναι None
        if X is None:
            raise ValueError("Input data is None.")

        # Έλεγχος αν ο πίνακας έχει 0 χαρακτηριστικά
        if X.shape[1] == 0:
            raise ValueError("Array found with 0 features.")

        # Δημιουργία πίνακα προβλέψεων από κάθε δέντρο στο δάσος
        predictions = np.array([tree.predict(X) for tree in self.trees])

        # Επιστροφή πίνακα που περιέχει την τελική πρόβλεψη για κάθε παράδειγμα
        return np.array([np.argmax(np.bincount(predictions[:, i])) for i in range(len(X))])


def preprocess_text(text):
    if isinstance(text, list):
        text = ' '.join([str(word) for word in text])
    return text

# Φόρτωση του dataset IMDB
# Τα δεδομένα περιλαμβάνουν τις κριτικές των ταινιών,
# όπου η κάθε κριτική έχει αντιστοιχηθεί σε μια λίστα από ακέραιους αριθμούς.
# Ο ακέραιος αριθμός αντιστοιχεί στη θέση μιας λέξης στο λεξικό.
# Επιλέγονται οι κορυφαίες 1000 συχνότερες λέξεις.
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=1000)

# Υπερπαράμετροι
m = 1000  #  Ο μέγιστος αριθμός χαρακτηριστικών που θα χρησιμοποιηθούν από τον CountVectorizer.
n = 0.9 # Το όριο υψηλής συχνότητας για τις λέξεις κατά τον υπολογισμό του CountVectorizer. Αφαιρεί τις λέξεις που εμφανίζονται πολύ συχνά και δεν προσφέρουν πολλή πληροφορία
k = 50  # Το όριο χαμηλής συχνότητας για τις λέξεις κατά τον υπολογισμό του CountVectorizer.
n_estimators = 10 # Ο αριθμός των δέντρων που θα περιέχει το δάσος
max_depth = 3 # Ο μέγιστος βάθος κάθε δέντρου του δάσους. Περιορίζει τον αριθμό των επιπέδων του δέντρου

# Μετατροπή των δεδομένων σε κείμενο
X_train_text = [preprocess_text(sample) for sample in X_train]
X_test_text = [preprocess_text(sample) for sample in X_test]

#κάθε κείμενο μετατρέπεται σε έναν πυκνό πίνακα όπου κάθε λέξη αποτελεί μια διάσταση,
# και η τιμή σε κάθε διάσταση είναι ο αριθμός των φορών που εμφανίζεται αυτή η λέξη στο κείμενο.

# binary=True: Κάθε λέξη παίρνει την τιμή 1 αν υπάρχει στο κείμενο και 0 αν δεν υπάρχει (binary αναπαράσταση).
vectorizer = CountVectorizer(max_features=m, max_df=n, min_df=k, binary=True)

#οι δύο πίνακες X_train_binary και X_test_binary περιέχουν την αναπαράσταση των δεδομένων εκπαίδευσης και ελέγχου,
# αντίστοιχα, σε μορφή πίνακα συχνοτήτων λέξεων, όπου κάθε γραμμή αντιστοιχεί σε ένα έγγραφο και κάθε στήλη αντιστοιχεί σε μια λέξη
X_train_binary = vectorizer.fit_transform(X_train_text).toarray()
X_test_binary = vectorizer.transform(X_test_text).toarray()


# Εκπαίδευση του Ταξινομητή RF
random_forest_classifier = myRandomForestClassifier(n_estimators, max_depth)
random_forest_classifier.fit(X_train_binary, y_train)

# Προβλέψεις
y_train_pred_rf = random_forest_classifier.predict(X_train_binary)
y_test_pred_rf = random_forest_classifier.predict(X_test_binary)

# Εκτύπωση μετρικών
# συγκριση labels των αρχικων δεδομενων με αυτων που επιστρεφει η predict
print(f"Random Forest - Train Accuracy: {accuracy_score(y_train, y_train_pred_rf):.5f}")
print(f"Random Forest - Test Accuracy: {accuracy_score(y_test, y_test_pred_rf):.5f}")

# Εκτύπωση Αναφοράς Κατηγοριοποίησης
classification_report_str = classification_report(y_train, y_train_pred_rf)
print("Random Forest - Classification Report(train):")
print(classification_report_str)

classification_report_str = classification_report(y_test, y_test_pred_rf)
print("Random Forest - Classification Report(test):")
print(classification_report_str)

# Αξιολόγηση της απόδοσης του Ταξινομητή Τυχαίων Δασών σε διάφορα μεγέθη εκπαίδευσης
train_accuracies = []  # Λίστα για την ακρίβεια στο σύνολο εκπαίδευσης
test_accuracies = []  # Λίστα για την ακρίβεια στο σύνολο ελέγχου
precisions = []  # Λίστα για την precision
recalls = []  # Λίστα για το recall
f1_scores = []  # Λίστα για το F1 Score

training_sizes = [100, 500, 1000, 2000, 5000, 10000, 25000]  # Διάφορα μεγέθη εκπαίδευσης

for size in training_sizes:
    start_time = time.time()
    random_forest_classifier = myRandomForestClassifier(n_estimators, max_depth)
    random_forest_classifier.fit(X_train_binary[:size], y_train[:size])

    y_train_pred_rf = random_forest_classifier.predict(X_train_binary[:size])
    y_test_pred_rf = random_forest_classifier.predict(X_test_binary[:size])

    train_accuracy = accuracy_score(y_train[:size], y_train_pred_rf)
    test_accuracy = accuracy_score(y_test[:size], y_test_pred_rf)
    precision = precision_score(y_test[:size], y_test_pred_rf)  #μετρά το ποσοστό των προβλεπόμενων θετικών που είναι πράγματι θετικά, σε σχέση με το σύνολο των προβλεπόμενων θετικών
    recall = recall_score(y_test[:size], y_test_pred_rf) #μετρά το ποσοστό των πραγματικών θετικών που είναι επίσης προβλεπόμενα θετικά, σε σχέση με το σύνολο των πραγματικών θετικών
    f1 = f1_score(y_test[:size], y_test_pred_rf) # συνδυασμός της ακρίβειας και της ανάκλησης και υπολογίζεται ως το αναλογικό βάρος των δύο

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Training size: {size}, Runtime: {runtime} seconds")

# Σχεδίαση των καμπυλών μάθησης
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, train_accuracies, label='Training Accuracy')
plt.plot(training_sizes, test_accuracies, label='Test Accuracy')
plt.xlabel('Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves:RF')
plt.legend()
plt.show()

# Σχεδίαση των μετρικών απόδοσης
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, precisions, label='Precision')
plt.plot(training_sizes, recalls, label='Recall')
plt.plot(training_sizes, f1_scores, label='F1 Score')
plt.xlabel('Size')
plt.ylabel('Score')
plt.title('Performance Metrics:RF')
plt.legend()
plt.show()

# FilmReview-Classifier

The project aimed to classify movie reviews from the IMDB dataset using three different machine learning algorithms: Naïve Bayes, Random Forest, and Logistic Regression.

Part A: Custom Implementations
Naïve Bayes:
-Utilized Keras's imdb.load_data to load and preprocess the data, with parameters set to m=500, n=0.9, and k=50.
-Developed a CustomNaiveBayes class to compute class probabilities and feature probabilities.
-Trained the model on the training set and evaluated its performance on both training and test sets, reporting accuracy, precision, recall, and F1-score.

Random Forest:
-Loaded the IMDB dataset similarly and initialized hyperparameters (m=1000, n=0.9, k=50, n_estimators=10, max_depth=3).
-Implemented myRandomForestClassifier and ID3DecisionTree classes for training and prediction.
-Reported performance metrics and visualized learning curves.

Logistic Regression:
-Loaded the IMDB dataset and set hyperparameters (m=1000, n=0.9, k=50, lambda_param=0.01).
-Implemented a LogisticRegression class to train the model using stochastic gradient descent.
-Evaluated performance metrics, including accuracy and F1-score.

Part B: Scikit-Learn Comparisons

Naïve Bayes with Scikit-Learn
Used BernoulliNB() from Scikit-learn and compared results with the custom implementation.
The custom Naïve Bayes implementation showed slightly higher accuracy than the Scikit-learn version.
Random Forest with Scikit-Learn

Employed RandomForestClassifier() from Scikit-learn and noted better performance compared to the custom version, with higher accuracy and faster execution.
Logistic Regression with Scikit-Learn

Implemented Scikit-learn’s logistic regression and observed similar performance to the custom implementation but with significantly faster execution.

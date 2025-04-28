from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Generate a random classification dataset
X, y = make_classification(
    n_samples=100, n_features=10, n_informative=5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# K-Means (Clustering)
# Assume 2 clusters for binary classification
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
kmeans_predictions = kmeans.predict(X_test)
# K-Means is unsupervised, so we calculate accuracy as a comparison with the
# true labels
kmeans_accuracy = accuracy_score(y_test, kmeans_predictions)
print(f"K-Means Accuracy (compared to labels): {100 * kmeans_accuracy:.2f}")

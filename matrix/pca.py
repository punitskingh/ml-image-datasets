from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Generate a random classification dataset
X, y = make_classification(
    n_samples=100, n_features=10, n_informative=5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Principal Component Analysis (PCA)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
# Using PCA as a feature reduction technique
svm_pca = SVC(kernel='linear')
svm_pca.fit(X_train_pca, y_train)
svm_pca_predictions = svm_pca.predict(X_test_pca)
svm_pca_accuracy = accuracy_score(y_test, svm_pca_predictions)
print(f"SVM with PCA Accuracy: {100*svm_pca_accuracy:.2f}")

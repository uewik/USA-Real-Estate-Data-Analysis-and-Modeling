import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("realtor-data-cleaned3.csv")
X = data[['bed', 'acre_lot', 'zip_code', 'house_size', 'city']]
y = data['price']
y_binary = (y > 4000000).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=5805)

# Finding the best k value using the elbow method
error_rates = []
for i in range(1, 11):  # Test values of k from 1 to 11
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rates.append(np.mean(pred_i != y_test))

# Plot the elbow graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), error_rates, color='blue', linestyle='dashed', marker='o',  # 11 according to the range
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# Choose the best k as the one with the lowest error rate
best_k = error_rates.index(min(error_rates)) + 1
print(f"The best K value is: {best_k}")

# Define parameter grid for KNN excluding 'n_neighbors'
param_grid = {
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}



best_knn = KNeighborsClassifier(n_neighbors=best_k)

# Create a GridSearchCV object
grid_search = GridSearchCV(best_knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_knn = grid_search.best_estimator_


# Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(10, 8))

total_precision = 0
total_recall = 0
total_fscore = 0
total_specificity = 0
total_cm = np.zeros((2, 2))

for i, (train, test) in enumerate(cv.split(X, y_binary)):
    y_pred = best_knn.predict(X.iloc[test])
    cm = confusion_matrix(y_binary.iloc[test], y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    # Accumulate the results
    cm_temp = confusion_matrix(y_binary[test], y_pred)
    print(f"Fold {i + 1} - Confusion Matrix:\n{cm_temp}")
    total_cm += cm_temp
    precision = round(precision_score(y_binary[test], y_pred, average='weighted'), 2)
    print(f"Precision: {precision}")
    total_precision += precision
    recall = round(recall_score(y_binary[test], y_pred, average='weighted'), 2)
    print(f"Recall: {recall}")
    total_recall += recall
    spec = round(specificity, 2)
    print(f"Specificity: {spec}\n")
    total_specificity += spec
    fscore = round(f1_score(y_binary[test], y_pred, average='weighted'), 2)
    print(f"F-score: {fscore}\n")
    total_fscore += fscore

    # Calculate ROC curve and AUC for each fold
    fpr, tpr, thresholds = roc_curve(y_binary[test], best_knn.predict_proba(X.iloc[test])[:, 1], pos_label=1)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i + 1} (AUC = {roc_auc:.2f})')

# Plot Mean ROC and AUC
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random classifier', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2, alpha=.8)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC Curve')
plt.legend(loc="lower right")
plt.show()

# Print average precision, recall, specificity, and F-score
print(f"Average Precision: {total_precision / 5}")
print(f"Average Recall: {total_recall / 5}")
print(f"Average Specificity: {total_specificity / 5}")
print(f"Average F-score: {total_fscore / 5}")

# Plot the aggregated confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(total_cm, annot=True, fmt='g')
plt.title('Aggregated Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

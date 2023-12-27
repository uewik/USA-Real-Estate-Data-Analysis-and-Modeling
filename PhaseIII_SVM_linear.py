import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("realtor-data-cleaned3.csv")
X = data[['bed', 'acre_lot', 'zip_code', 'house_size', 'city']]
y = data['price']
y_binary = (y > 4000000).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=5805)

param_grid = {
    'C': [0.1, 1]  # Example range for C
}

# Support Vector Machine with linear kernel
svm_linear = SVC(kernel='linear', probability=True, random_state=5805)

grid_search = GridSearchCV(svm_linear, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_svm_linear = grid_search.best_estimator_

# Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)



total_precision = 0
total_recall = 0
total_fscore = 0
total_specificity = 0
total_cm = np.zeros((2, 2))

for i, (train, test) in enumerate(cv.split(X, y_binary)):
    y_pred = best_svm_linear.predict(X.iloc[test])
    cm = confusion_matrix(y_binary.iloc[test], y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    else:
        # Handle the case where the confusion matrix is not 2x2
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_pred)) == 1:
            if np.unique(y_pred)[0] == 0:
                tn = cm[0][0]
            else:
                tp = cm[0][0]
        specificity = 0

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

    # Calculate ROC curve and AUC for each fold using decision_function or predict_proba
    # Use decision_function or predict_proba depending on whether SVC is set with probability=True
    y_scores = best_svm_linear.decision_function(X.iloc[test])
    fpr, tpr, thresholds = roc_curve(y_binary.iloc[test], y_scores)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

plt.figure(figsize=(10, 8))
for i in range(len(tprs)):
    plt.plot(mean_fpr, tprs[i], lw=1, alpha=0.3, label=f'ROC fold {i + 1} (AUC = {aucs[i]:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Random classifier', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2, alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC Curve')
plt.legend(loc="lower right")
plt.show()

# Print average precision, recall, specificity, and F-score
print(f"Average Precision: {round(total_precision / 5, 2)}")
print(f"Average Recall: {round(total_recall / 5, 2)}")
print(f"Average Specificity: {round(total_specificity / 5, 2)}")
print(f"Average F-score: {total_fscore / 5}")

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(total_cm, annot=True, fmt='g')
plt.title('Aggregated Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

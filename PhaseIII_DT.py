import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("realtor-data-cleaned3.csv")
X = data[['bed', 'acre_lot', 'zip_code', 'house_size', 'city']]

y = data['price']
y_binary = (y > 4000000).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=5805)

param_grid_adjusted = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'max_features': ['auto', 'sqrt'],
    'ccp_alpha': [0.0, 0.01]
}

# Adjusted Grid Search with cross-validation
dt_grid_adjusted = DecisionTreeClassifier(random_state=5805)
grid_search_adjusted = GridSearchCV(dt_grid_adjusted, param_grid_adjusted, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_adjusted.fit(X_train, y_train)

# Best parameters from the adjusted grid search
best_params_adjusted = grid_search_adjusted.best_params_
best_dt_adjusted = grid_search_adjusted.best_estimator_

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
    best_dt_adjusted.fit(X.iloc[train], y_binary[train])
    y_pred = best_dt_adjusted.predict(X.iloc[test])
    cm = confusion_matrix(y_binary.iloc[test], y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    # Confusion matrix, Precision, Recall, F-score
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
    fpr, tpr, thresholds = roc_curve(y_binary[test], best_dt_adjusted.predict_proba(X.iloc[test])[:, 1], pos_label=1)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i + 1} (AUC = {roc_auc:.2f})')

# Print average precision, recall, specificity, and F-score
print(f"Average Precision: {round(total_precision / 5, 2)}")
print(f"Average Recall: {round(total_recall / 5, 2)}")
print(f"Average Specificity: {round(total_specificity / 5, 2)}")
print(f"Average F-score: {round(total_fscore / 5, 2)}")

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(total_cm, annot=True, fmt='g')
plt.title('Aggregated Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot Mean ROC and AUC
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random classifier', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f)' % mean_auc, lw=2, alpha=.8)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC Curve')
plt.legend(loc="lower right")
plt.show()

from prettytable import PrettyTable



table = PrettyTable()
table.title = "performance of different classifiers"
table.field_names = ['Classifier', 'AUC', 'Average Precision', 'Average Recall', 'Average Specificity', 'Average F-score']
table.add_row(["Decision Tree", 0.81, 0.97, 0.97, 0.99, 0.97])
table.add_row(["Logistic Regression", 0.61, 0.96, 0.98, 1.0, 0.96])
table.add_row(["KNN", 0.98, 1.0, 1.0, 1.0, 1.0])
table.add_row(["SVM (linear)", 0.7, 0.96, 0.98, 1.0, 0.96])
table.add_row(["SVM (polynomial)", 0.45, 0.95, 0.98, 1.0, 0.96])
table.add_row(["SVM (radial)", 0.44, 0.95, 0.98, 1.0, 0.96])
table.add_row(["Naive Bayes", 0.63, 0.96, 0.74, 0.74, 0.80])
table.add_row(["Random Forest (bagging)", 0.99, 1.0, 1.0, 1.0, 1.0])
table.add_row(["Random Forest (stacking)", 0.99, 1.0, 1.0, 1.0, 1.0])
table.add_row(["Random Forest (boosting)", 0.99, 0.99, 0.99, 1.0, 0.99])
table.add_row(["Neural Network", 0.93, 0.96, 0.98, 1.0, 0.96])

print(table)

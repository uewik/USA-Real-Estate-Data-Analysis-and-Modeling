This project focuses on applying machine learning techniques to the USA Real
Estate dataset, a comprehensive collection of real estate listings across the United
States. The objective is to gain hands-on experience in feature engineering, classification,
clustering, and association rule mining, thereby applying theoretical knowledge
to practical data analysis.

Phase I of the project involved feature engineering and exploratory data analysis
(EDA) of the real estate dataset. Key tasks included handling missing data, removing
duplicates, and optionally aggregating and downsampling data. Dimensionality
reduction techniques, such as Random Forest Analysis, Principal Component
Analysis, Singular Value Decomposition, and Variance Inflation Factor (VIF), were
employed to minimize collinearity. This phase also included discretization and binarization
of variables, normalization, and standardization, culminating in a detailed
visual analysis through heatmaps of sample covariance and Pearson correlation coefficients.

In Phase II, a regression analysis was conducted to predict house prices in the
real estate market. A multiple linear regression model was developed, incorporating
T-test and F-test analyses, with performance indicators like R-squared, adjusted Rsquare,
AIC, BIC, and MSE. The final model provided a prediction of the dependent
variable ’price’, along with confidence interval and backward stepwise regression
analyses.

Phase III dealt with classification analysis, applying various classifiers—Decision
Tree, Logistic Regression, KNN, SVM, Na¨ıve Bayes, Random Forest, and Neural
Networks—to the dataset. Each classifier was fine-tuned using grid search and evaluated
based on confusion matrix, precision, recall, specificity, F-score, and ROC/AUC
curves, using Stratified K-fold cross-validation. The phase culminated in identifying
the classifier that best fits the real estate data characteristics.

Finally, Phase IV focused on clustering and association rule mining as independent
research. The K-means++ algorithm was applied for clustering, including
silhouette analysis for optimal cluster number determination and within-cluster
variation analysis. The Apriori algorithm was employed for association rule mining,
providing insights into underlying patterns and associations in the USA real estate
market.

About files:

You can tell the content of programs by the names of py. files:

Phase I: There are ten py. files for Phase I: Feature Engineering & EDA: PhaseI-1_clean.py ~ PhaseI-10_Corr.py. Please run py. files in order.

Phase II: There is only one file for Phase II: Regression Analysis: PhaseII_2.py. Please run this program after you have run all ten programs in Phase I.

Phase III: There are 12 files for Phase III: Classification Analysis:

PhaseIII_DT.py: Decision tree

PhaseIII_Logistic.py: Logistic regression

PhaseIII_Naive.py: Naïve Bayes

PhaseIII_neural.py: Neural Network

PhaseIII_RF_bag.py: Random Forest Bagging

PhaseIII_RF_boost.py: Random Forest Boosting

PhaseIII_RF_stack.py: Random Forest Stacking

PhaseIII_summary.py: This is summary for Phase III. Please run this file after you run all files for Phase III.

PhaseIII_SVM_linear.py: SVM with linear kernel    WARNING: this file will take hours to run.

PhaseIII_SVM_POLY.py: SVM with polynomial kernel    WARNING: this file will take hours to run.

PhaseIII_SVM_radial.py: SVM with radial base kernel    WARNING: this file will take hours to run.

Phase IV: Two files: PhaseIV_Kmean++.py and PhaseIV_apriori.py. Please first run PhaseIV_Kmean++.py..

The original dataset is in 'realtor-data.csv'.

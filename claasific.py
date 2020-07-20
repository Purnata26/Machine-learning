import pandas as pd
from sklearn.model_selection import train_test_split
path = 'pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(path, header=None, names=col_names)
columns=['pregnant','insulin','bmi','age']
X=pima[columns]
y=pima.label
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred=logreg.predict(X_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))
confusion=metrics.confusion_matrix(y_test,y_pred)
TP=confusion[1,1]
TN=confusion[0,0]
FP=confusion[0,1]
FN=confusion[1,0]
print((TP + TN) / float(TP + TN + FP + FN))

print(metrics.f1_score(y_test,y_pred))
print(logreg.predict_proba(X_test)[0:10,1])
from matplotlib import pyplot as plt
y_pred_prob = logreg.predict_proba(X_test)[:, 1]



from sklearn.preprocessing import binarize
y_pred_class=binarize([y_pred_prob],threshold=0.34)[0]
from sklearn import metrics
con=metrics.confusion_matrix(y_test,y_pred_class)
print(con)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()
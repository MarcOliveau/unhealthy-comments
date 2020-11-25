from pathlib import Path



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingCVClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

dataset = pd.read_csv('train.csv')
dataset_majority = dataset[dataset.healthy == 1]
dataset_minority = dataset[dataset.healthy == 0]

dataset_minority_upsampled = resample(dataset_minority,
                                      replace=True,
                                      n_samples=16000,
                                      random_state=123)
dataset_majority_downsampled = resample(dataset_majority,
                                        replace=False,
                                        n_samples=16000,
                                        random_state=123)

## Train_Dataset Preprocessing
train_dataset = pd.concat([dataset_majority_downsampled, dataset_minority_upsampled])
X_set_1 = train_dataset.iloc[:, 3:13].values
df_X_1 = pd.DataFrame(X_set_1,
                      columns=["antagonise", "antagonise:confidence", "condescending", "condescending:confidence",
                               "dismissive", "dismissive:confidence", "generalisation", "generalisation:confidence",
                               "generalisation_unfair", "generalisation_unfair:confidence"])
X_set_2 = train_dataset.iloc[:, 15:].values
df_X_2 = pd.DataFrame(X_set_2, columns=["hostile", "hostile:confidence", "sarcastic", "sarcastic:confidence"])

X_data_train = pd.concat([df_X_1.reset_index(drop=True),
                          df_X_2.reset_index(drop=True)],
                         axis=1,
                         ignore_index=True)
X_data_columns = [
    list(df_X_1.columns),
    list(df_X_2.columns)]

flatten = lambda nested_lists: [item for sublist in nested_lists for item in sublist]
X_data_train.columns = flatten(X_data_columns)

c1 = X_data_train["antagonise"] - X_data_train["antagonise:confidence"]
c2 = X_data_train["condescending"] - X_data_train["condescending:confidence"]
c3 = X_data_train["dismissive"] - X_data_train["dismissive:confidence"]
c4 = X_data_train["generalisation"] - X_data_train["generalisation:confidence"]
c5 = X_data_train["generalisation_unfair"] - X_data_train["generalisation_unfair:confidence"]
c6 = X_data_train["hostile"] - X_data_train["hostile:confidence"]
c7 = X_data_train["sarcastic"] - X_data_train["sarcastic:confidence"]

X_train = pd.DataFrame({"c1": c1,
                        "c2": c2,
                        "c3": c3,
                        "c4": c4,
                        "c5": c5,
                        "c6": c6,
                        "c7": c7},
                       )
X_train = X_train.abs()
print(X_train)
y_data = train_dataset.iloc[:, 13].values
y_train = pd.DataFrame(y_data, columns=["Healthy"])
y_train = y_train.values.ravel()
print(y_train)

## Test_Dataset Preprocessing

test_dataset = pd.read_csv('test.csv')
X_set_1 = test_dataset.iloc[:, 3:13].values
df_X_1 = pd.DataFrame(X_set_1,
                      columns=["antagonise", "antagonise:confidence", "condescending", "condescending:confidence",
                               "dismissive", "dismissive:confidence", "generalisation", "generalisation:confidence",
                               "generalisation_unfair", "generalisation_unfair:confidence"])
X_set_2 = test_dataset.iloc[:, 15:].values
df_X_2 = pd.DataFrame(X_set_2, columns=["hostile", "hostile:confidence", "sarcastic", "sarcastic:confidence"])

X_data_test = pd.concat([df_X_1.reset_index(drop=True),
                         df_X_2.reset_index(drop=True)],
                        axis=1,
                        ignore_index=True)
X_data_columns = [
    list(df_X_1.columns),
    list(df_X_2.columns)]

flatten = lambda nested_lists: [item for sublist in nested_lists for item in sublist]
X_data_test.columns = flatten(X_data_columns)

c1 = X_data_test["antagonise"] - X_data_test["antagonise:confidence"]
c2 = X_data_test["condescending"] - X_data_test["condescending:confidence"]
c3 = X_data_test["dismissive"] - X_data_test["dismissive:confidence"]
c4 = X_data_test["generalisation"] - X_data_test["generalisation:confidence"]
c5 = X_data_test["generalisation_unfair"] - X_data_test["generalisation_unfair:confidence"]
c6 = X_data_test["hostile"] - X_data_test["hostile:confidence"]
c7 = X_data_test["sarcastic"] - X_data_test["sarcastic:confidence"]

X_test = pd.DataFrame({"c1": c1,
                       "c2": c2,
                       "c3": c3,
                       "c4": c4,
                       "c5": c5,
                       "c6": c6,
                       "c7": c7},
                      )
X_test = X_test.abs()
print(X_test)
y_data = test_dataset.iloc[:, 13].values
y_test = pd.DataFrame(y_data, columns=["Healthy"])
y_test.values.ravel()
print(y_test)
y_test = y_test.values.ravel()





## Bagging Technique
rnd_clf = RandomForestClassifier(
    n_estimators=400,
    max_leaf_nodes=15,
    random_state=42)
rnd_clf.fit(X_train, y_train)
## Confusion Matrix
y_pred_rf = rnd_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)
print("accuracy score rf: ",accuracy_score(y_test, y_pred_rf), "\n"
      "  roc auc score rf: ", roc_auc_score(y_test, y_pred_rf))
##Cross Validation
accuracies = cross_val_score(estimator = rnd_clf,
                             X = X_train, y = y_train,
                             scoring= "roc_auc",
                             cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))






## XGBoost Classifier
xg_clf = XGBClassifier()
xg_clf.fit(X_train, y_train)
## Confsuion Matrix
y_pred_xg = xg_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_xg)
print(cm)
print("accuracy score xg: ",accuracy_score(y_test, y_pred_xg), "\n"
      "  roc auc score xg: ", roc_auc_score(y_test, y_pred_xg))
## Cross Validation
accuracies_xg = cross_val_score(estimator = xg_clf, X=X_train, y=y_train, scoring= "roc_auc", cv=10)
print("Accuracy xg: {:.2f} %".format(accuracies_xg.mean()*100))
print("Standard Deviation xg: {:.2f} %".format(accuracies_xg.std()*100))




## Adaboost Classifier
boost_clf = AdaBoostClassifier(n_estimators=100, learning_rate=1)
boost_clf.fit(X_train, y_train)
## Confusion Matrix
y_pred_ab = boost_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_ab)
print(cm)
print("accuracy score ab: ",accuracy_score(y_test, y_pred_ab), "\n"
      "  roc auc score ab: ", roc_auc_score(y_test, y_pred_ab))
accuracy_score(y_test, y_pred_ab)
## Cross Validation
accuracies = cross_val_score(estimator=boost_clf, X=X_train, y=y_train, scoring="roc_auc", cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))




##Soft Voting Classifier
log_clf = LogisticRegression()
knn_clf = KNeighborsClassifier(n_neighbors=4)
svc_clf = SVC(probability=True)
voting_clf = VotingClassifier(
    estimators=[("lr", log_clf), ("knn", knn_clf), ("svc", svc_clf)],
    voting="soft",
)
for clf in (log_clf, knn_clf, svc_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, roc_auc_score(y_test, y_pred))



## Stack Method
stack_clf = StackingCVClassifier(classifiers=[voting_clf, rnd_clf, xg_clf],
                                 shuffle=False,
                                 use_probas=True,
                                 cv=5,
                                 meta_classifier=SVC(probability=True))
classifiers = {"voting": voting_clf,
               "RF": rnd_clf,
               "XG": xg_clf,
               "Stack": stack_clf}
for key in classifiers:
    classifier = classifiers[key]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(key, roc_auc_score(y_pred, y_test))
    classifiers[key] = classifier






import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import Imputer

feature_set = pd.read_csv('data/feature_set.csv')
feature_set.drop_duplicates(inplace=True)

Y = feature_set.label
# X = feature_set.drop(["label", "user_id", "seller_id"], 1)
X = feature_set.drop(["label"], 1)

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# def RfModelDecision(train_x, label, Ptest_x, Ntest_x):
#     print
#     'training random forest model...'
#     rf_model = RandomForestClassifier(n_estimators=60)
#     rf_model.fit(train_x, label)
#
#     Ppred_rf = rf_model.predict_proba(Ptest_x)
#     Npred_rf = rf_model.predict_proba(Ntest_x)
#
#     test_label = [1] * len(Ppred_rf) + [0] * len(Npred_rf)
#     score = Ppred_rf.tolist() + Npred_rf.tolist()
#     score_rf = [x[1] for x in score]
#     auc = roc_auc_score(test_label, score_rf)
#     print
#     'rf_AUC:', auc
#
#     return score_rf, rf_model
#
#
# def SVMModelDecision(train_x, label, Ptest_x, Ntest_x):
#     print
#     'training svm model...'
#     svm_model = svm.SVC(C=35, gamma=0.005, kernel='rbf', probability=True)
#     svm_model.fit(train_x, label)
#
#     Ppred_svm = svm_model.predict_proba(Ptest_x)
#     Npred_svm = svm_model.predict_proba(Ntest_x)
#
#     test_label = [1] * len(Ppred_svm) + [0] * len(Npred_svm)
#     score = Ppred_svm.tolist() + Npred_svm.tolist()
#     score_svm = [x[1] for x in score]
#     auc = roc_auc_score(test_label, score_svm)
#     print
#     'svm_AUC:', auc
#
#     return score_svm, svm_model

# tmp_score, rf_model = RfModelDecision(X_train, y_train, X_test, y_test)
#
# y_pred = rf_model.predict(X_test)
# predictions = [round(value) for value in y_pred]
#
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

model = XGBClassifier()
model.fit(X_train, y_train)

# save model to file
pickle.dump(model, open("pima.pickle.dat", "wb"))

# some time later...

# load model from file
loaded_model = pickle.load(open("pima.pickle.dat", "rb"))

y_pred = loaded_model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

result = loaded_model.predict(X)

prediction = pd.DataFrame(result, columns=['prob']).to_csv('data/prediction.csv')

# plot_importance(model)
# pyplot.show()

# save feature score
# feature_score = model.get_fscore()
# feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
# fs = []
# for (key, value) in feature_score:
#     fs.append("{0},{1}\n".format(key, value))
#
# with open('xgb_feature_score.csv', 'w') as f:
#     f.writelines("feature,score\n")
#     f.writelines(fs)

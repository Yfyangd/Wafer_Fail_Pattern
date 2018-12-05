
# coding: utf-8

# In[ ]:


#import lib
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Read Data
df = pd.read_csv("Input\data.csv")

#Data Preprocess
y = df['Label']
features = ['Vendor','Usage', 'Recycle', 'MAINPD_ID','EQP1','Recipe1','MAINPD_ID_1','MAINPD_ID_2']
x = df[features]

#OneHot Enocding
from sklearn.preprocessing import LabelEncoder
LE1 = LabelEncoder()
LE2 = LabelEncoder()
LE3 = LabelEncoder()
LE4 = LabelEncoder()
#LE5 = LabelEncoder()
LE6 = LabelEncoder()
LE7 = LabelEncoder()

x['Vendor'] = LE1.fit_transform(x['Vendor'])
x['MAINPD_ID'] = LE2.fit_transform(x['MAINPD_ID'])
x['EQP1'] = LE3.fit_transform(x['EQP1'])
x['Recipe1'] = LE4.fit_transform(x['Recipe1'])
#x['CAST_ID'] = LE5.fit_transform(x['CAST_ID'])
x['MAINPD_ID_1'] = LE6.fit_transform(x['MAINPD_ID_1'])
x['MAINPD_ID_2'] = LE7.fit_transform(x['MAINPD_ID_2'])


#Feature Extraction
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x, y)

#sns.set(style="darkgrid")
fig, ax = plt.subplots(figsize=(6,6))
y_pos = np.arange(len(features))
plt.barh(y_pos, model.feature_importances_, align='center', alpha=0.4)
plt.yticks(y_pos, x)
plt.xlabel('features')
plt.title('feature_importances')
plt.show()

#Model Training
y2 = df['Label'].values
from sklearn.ensemble import RandomForestClassifier
 
rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=100,
                             min_samples_split=12, #20
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1) 

rf.fit(x, y2) #filter SP data
print("Out Of Bag score is %.4f" % rf.oob_score_)

# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the data into training and testing sets with 20% test rate
X_train, X_test, y_train, y_test = train_test_split(x, y2, test_size = 0.2, random_state = 0)

# Training model
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(random_state=101)
RFC.fit(X_train,y_train)

# Import 4 metrics from sklearn for testing
from sklearn.metrics import accuracy_score,precision_score,recall_score,fbeta_score
print ("Accuracy on testing data of RandomForestClassifier: {:.4f}".format(accuracy_score(y_test, RFC.predict(X_test))))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, RFC.predict(X_test))

#KPI
tn, fp, fn, tp = confusion_matrix(y_test, RFC.predict(X_test)).ravel()

Accuracy = (tn+tp)/(tn+tp+fp+fn)
Precision = (tp)/(tp+fp)
Recall = tp/(tp+fn)
Fscore = (2*Precision*Recall)/(Precision+Recall)
print ("Accuracy on testing data of RandomForestClassifier: {:.4f}".format(Accuracy))
print ("Precision on testing data of RandomForestClassifier: {:.4f}".format(Precision))
print ("Recall on testing data of RandomForestClassifier: {:.4f}".format(Recall))
print ("F-score on testing data of RandomForestClassifier: {:.4f}".format(Fscore))


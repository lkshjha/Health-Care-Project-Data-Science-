#!/usr/bin/env python
# coding: utf-8

# In[61]:


get_ipython().run_line_magic('matplotlib', 'inline')
##import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import pandas as pd


# In[62]:


data = pd.read_csv('health care diabetes.csv')


# In[63]:


data.head()


# In[64]:


data.isnull().any()


# In[65]:


data.info()


# In[66]:


Positive = data[data['Outcome']==1]
Positive.head(5)


# In[67]:


data['Glucose'].value_counts().head(7)


# In[68]:


plt.hist(data['Glucose'])


# In[69]:


data['BloodPressure'].value_counts().head(7)


# In[70]:


plt.hist(data['BloodPressure'])


# In[71]:


data['SkinThickness'].value_counts().head(7)


# In[72]:


plt.hist(data['SkinThickness'])


# In[73]:


data['Insulin'].value_counts().head(7)


# In[74]:


plt.hist(data['Insulin'])


# In[75]:


data['BMI'].value_counts().head(7)


# In[76]:


plt.hist(data['BMI'])


# In[77]:


data.describe().transpose()


# In[ ]:





# # Week 2

# In[78]:


plt.hist(Positive['BMI'],histtype='stepfilled',bins=20)


# In[79]:


Positive['BMI'].value_counts().head(7)


# In[80]:


plt.hist(Positive['Glucose'],histtype='stepfilled',bins=20)


# In[81]:


Positive['Glucose'].value_counts().head(7)


# In[82]:


plt.hist(Positive['BloodPressure'],histtype='stepfilled',bins=20)


# In[83]:


Positive['BloodPressure'].value_counts().head(7)


# In[84]:


plt.hist(Positive['SkinThickness'],histtype='stepfilled',bins=20)


# In[85]:


Positive['SkinThickness'].value_counts().head(7)


# In[86]:


plt.hist(Positive['Insulin'],histtype='stepfilled',bins=20)


# In[87]:


Positive['Insulin'].value_counts().head(7)


# In[88]:


#Scatter plot


# In[89]:


BloodPressure = Positive['BloodPressure']
Glucose = Positive['Glucose']
SkinThickness = Positive['SkinThickness']
Insulin = Positive['Insulin']
BMI = Positive['BMI']


# In[90]:


plt.scatter(BloodPressure, Glucose, color=['b'])
plt.xlabel('BloodPressure')
plt.ylabel('Glucose')
plt.title('BloodPressure & Glucose')
plt.show()


# In[91]:


g =sns.scatterplot(x= "Glucose" ,y= "BloodPressure",
              hue="Outcome",
              data=data);


# In[92]:


B =sns.scatterplot(x= "BMI" ,y= "Insulin",
              hue="Outcome",
              data=data);


# In[93]:


S =sns.scatterplot(x= "SkinThickness" ,y= "Insulin",
              hue="Outcome",
              data=data);


# In[94]:


### correlation matrix
data.corr()


# In[95]:


### create correlation heat map
sns.heatmap(data.corr())


# In[96]:


plt.subplots(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,cmap='viridis')  ### gives correlation value


# In[97]:


plt.subplots(figsize=(8,8))
sns.heatmap(data.corr(),annot=True)  ### gives correlation value


# In[98]:


# Logistic Regreation and model building


# In[99]:


data.head(5)


# In[100]:


features = data.iloc[:,[0,1,2,3,4,5,6,7]].values
label = data.iloc[:,8].values


# In[101]:


#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,
                                                label,
                                                test_size=0.2,
                                                random_state =10)


# In[102]:


#Create model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train) 


# In[103]:


print(model.score(X_train,y_train))
print(model.score(X_test,y_test))


# In[104]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(label,model.predict(features))
cm


# In[105]:


from sklearn.metrics import classification_report
print(classification_report(label,model.predict(features)))


# In[106]:


#Preparing ROC Curve (Receiver Operating Characteristics Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# predict probabilities
probs = model.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(label, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(label, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')


# In[107]:


#Applying Decission Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier(max_depth=5)
model3.fit(X_train,y_train)


# In[108]:


model3.score(X_train,y_train)


# In[109]:


model3.score(X_test,y_test)


# In[132]:


#Applying Random Forest
from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(n_estimators=11)
model4.fit(X_train,y_train)


# In[111]:


model4.score(X_train,y_train)


# In[112]:


model4.score(X_test,y_test)


# In[113]:


#Support Vector Classifier

from sklearn.svm import SVC 
model5 = SVC(kernel='rbf',
           gamma='auto')
model5.fit(X_train,y_train)


# In[118]:


model5.score(X_test,y_test).score(X_train,y_train)


# In[124]:


model5.score(X_test,y_test)


# In[125]:


#Applying K-NN
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=7,
                             metric='minkowski',
                             p = 2)
model2.fit(X_train,y_train)


# In[126]:


#Preparing ROC Curve (Receiver Operating Characteristics Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# predict probabilities
probs = model2.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(label, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(label, probs)
print("True Positive Rate - {}, False Positive Rate - {} Thresholds - {}".format(tpr,fpr,thresholds))
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")


# In[127]:


#Precision Recall Curve for Logistic Regression

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.')


# In[128]:


#Precision Recall Curve for KNN

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model2.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model2.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.')


# In[129]:


#Precision Recall Curve for Decission Tree Classifier

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model3.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model3.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.')


# In[130]:


#Precision Recall Curve for Random Forest

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model4.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model4.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.')


# In[ ]:





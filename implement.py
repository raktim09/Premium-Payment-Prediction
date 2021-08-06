import pandas as pd
import numpy as np                    
import seaborn as sns                  
import matplotlib.pyplot as plt 
import seaborn as sn                   
%matplotlib inline

train=pd.read_csv(r"A:\Data Science Challenge in 7 days\train_qnU1GcL.csv")
test=pd.read_csv(r"A:\Data Science Challenge in 7 days\test_LxCaReE_DvdCKVT_gimWwKr.csv")

train=train.drop('id',axis=1)
idcol=test['id']
test=test.drop('id',axis=1)
train.shape, test.shape

train.columns

train.head()

test.columns

train.dtypes

#UNIVARIATE ANALYSIS
train['target'].value_counts()

train['target'].value_counts(normalize=True)

train['target'].value_counts().plot.bar()


sn.distplot(train["age_in_days"])

sn.distplot(train["Income"])

sn.distplot(train["no_of_premiums_paid"])



train['sourcing_channel'].value_counts().plot.bar()

#target vs sourcing channel
print(pd.crosstab(train['sourcing_channel'],train['target']))

job=pd.crosstab(train['sourcing_channel'],train['target'])
job.div(job.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('sourcing_channel')
plt.ylabel('Percentage')

#target vs residence_area_type
print(pd.crosstab(train['residence_area_type'],train['target']))

job=pd.crosstab(train['residence_area_type'],train['target'])
job.div(job.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('residence_area_type')
plt.ylabel('Percentage')







corr = train.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")

corr

train.isnull().sum()

test.isnull().sum()

train.fillna(train.mean(),inplace=True)

test.fillna(test.mean(),inplace=True)

#model building
target = train['target']
train = train.drop('target',1)

train = pd.get_dummies(train)

#FOR TEST
test = pd.get_dummies(test)

#x_train1=train.drop('target',axis=1)
#y_train1=train['target']

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state=5)

#LOGISTIC REG
from sklearn.linear_model import LogisticRegression

lreg = LogisticRegression()

# fitting the model on  X_train and y_train
lreg.fit(X_train,y_train)

prediction = lreg.predict(X_val)

from sklearn.metrics import accuracy_score

# calculating the accuracy score
accuracy_score(y_val, prediction)

X_val.shape, test.shape

X_val.columns

test.columns

finalpred=lreg.predict(test)

print(finalpred[500:])

submission0 = pd.DataFrame()
submission0['id'] = idcol
submission0['target'] = finalpred

submission0.to_csv('LRans.csv', header=True, index=False)

res1 = pd.DataFrame(finalpred)

#res1.columns = ["target"]
#res1.to_csv("ANSWERs.csv")

#soln=pd.read_csv("ANSWERs.csv")
#soln1=soln['targetfinalpred']
#soln1.value_counts()

#Decision
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=45, random_state=40)

clf.fit(X_train,y_train)

predict = clf.predict(X_val)

accuracy_score(y_val, predict)

test_prediction = clf.predict(test)

submission = pd.DataFrame()
submission['id'] = idcol

submission['target'] = test_prediction

submission.to_csv('submission.csv', header=True, index=False)

#THIS IS THE FINAL STORED VALUES FOR SUBMISSION 
soln2=pd.read_csv("submission.csv")
soln3=soln2['target']
soln3.value_counts()

#soln3.dtypes()


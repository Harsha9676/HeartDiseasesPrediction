# import librares

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#import file

hd = pd.read_csv('Heart Disease Prediction.csv')
hd



hd.info()

hd.shape

hd.isnull().sum()

hd.describe()

plt.figure(figsize=(7, 5))
target_count = [len(hd[hd['target'] == 0]),len(hd[hd['target'] == 1])]
labels = ['No Disease', 'Disease']
colors = ['blue', 'red']
explode = (0.05, 0.1)
plt.pie(target_count, explode=explode, labels=labels, colors=colors,autopct='%4.2f%%',shadow=True, startangle=45)
plt.title('Target Percent')
plt.axis('equal')
plt.show()  


plt.figure(figsize=(7, 5))
target_count = [len(hd[hd['sex'] == 1]),len(hd[hd['sex'] == 0])]
labels = ['male', 'female']
colors = ['blue', 'red']
explode = (0.05, 0.1)
plt.pie(target_count, explode=explode, labels=labels, colors=colors,autopct='%4.2f%%',shadow=True, startangle=45)
plt.title('Gender Percent')
plt.axis('equal')
plt.show()


categorical_val = []
continous_val = []
for column in hd.columns:
    print('==============================')
    print(f"{column} : {hd[column].unique()}")
    if len(hd[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)
        
        
        
        plt.figure(figsize=(15, 15))

for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    hd[hd["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    hd[hd["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)
    
    
    
    plt.figure(figsize=(15, 15))

for i, column in enumerate(continous_val, 1):
    plt.subplot(3, 2, i)
    hd[hd["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    hd[hd["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)
    
    
    
    
    cor = hd.corr()
features = cor.index
plt.figure(figsize=(20,20))

# heat map

heat = sns.heatmap(hd[features].corr(),annot= True,cmap="RdYlGn")



dummy_hd = pd.get_dummies(hd, columns =['sex','cp','fbs','restecg','exang','slope','ca','thal'])

dummy_hd.head()
sc = StandardScaler()
col = ['age','chol','trestbps','thalach','oldpeak']
dummy_hd[col]=sc.fit_transform(dummy_hd[col])


X = dummy_hd.drop(['target'], axis=1)
y = dummy_hd['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 25)

log_reg = LogisticRegression(solver='lbfgs')
log_reg.fit(X_train, y_train)


test_score1 = accuracy_score(y_test, log_reg.predict(X_test)) * 100
train_score1 = accuracy_score(y_train, log_reg.predict(X_train)) * 100

result_log = pd.DataFrame(data=[["Logistic Regression", train_score1, test_score1]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
result_log



# Predicting the Test set results
Y_pred = log_reg.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, Y_pred)
print(cm)


sns.heatmap(cm,annot = True)


x = pd.DataFrame(cm)

x.index = ['Actual No Disease','Actual Disease']
x.columns = ['Predicted No Disease','Predicted Disease']
x


print(classification_report(y_test, Y_pred))


knn_classifier = KNeighborsClassifier(metric='minkowski')
knn_classifier.fit(X_train, y_train)




test_score2 = accuracy_score(y_test, knn_classifier.predict(X_test)) * 100
train_score2 = accuracy_score(y_train, knn_classifier.predict(X_train)) * 100

result_knn = pd.DataFrame(data=[["K-nearest neighbors", train_score2, test_score2]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
result_knn = result_log.append(result_knn, ignore_index=True)
result_knn




# Predicting the Test set results
Y_pred = knn_classifier.predict(X_test)

# Making the Confusion Matrix
cm2 = confusion_matrix(y_test, Y_pred)
print(cm)





y = pd.DataFrame()
y = pd.DataFrame(cm2)

y.index = ['Actual No Disease','Actual Disease']
y.columns = ['Predicted No Disease','Predicted Disease']
y





print(classification_report(y_test, Y_pred))



train_score = []
test_score = []
neighbors = range(1, 21)

for k in neighbors:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    train_score.append(accuracy_score(y_train, model.predict(X_train)))
    test_score.append(accuracy_score(y_test, model.predict(X_test)))
plt.figure(figsize=(12, 8))

plt.plot(neighbors, train_score, label="Train score")
plt.plot(neighbors, test_score, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_score)*100:.2f}%")




#random forest
rand_forest = RandomForestClassifier(n_estimators=1900, random_state=15)
rand_forest.fit(X_train, y_train)


test_score5 = accuracy_score(y_test, rand_forest.predict(X_test)) * 100
train_score5 = accuracy_score(y_train, rand_forest.predict(X_train)) * 100

result_rf = pd.DataFrame(data=[["Random Forest Classifier", train_score5, test_score5]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
result_rf = result_rf.append(result_knn, ignore_index=True)
result_rf




# Predicting the Test set results
Y_pred = rand_forest.predict(X_test)

# Making the Confusion Matrix
cm5 = confusion_matrix(y_test, Y_pred)
print(cm5)




r = pd.DataFrame(cm5)

r.index = ['Actual No Disease','Actual Disease']
r.columns = ['Predicted No Disease','Predicted Disease']
r

print(classification_report(y_test, Y_pred))



# Applying Machine Learning Algorithms Using Hyperparameter Tuning
## Logistic Regression Hyperparameter Tuning


from sklearn.model_selection import GridSearchCV

params = {"C": np.logspace(-4, 4, 20),
          "solver": ["liblinear"]}

log_reg = LogisticRegression()

grid_search_cv = GridSearchCV(log_reg, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=5, iid=True)
grid_search_cv.fit(X_train, y_train)

log_reg = LogisticRegression(C=0.615848211066026, 
                             solver='liblinear')

log_reg.fit(X_train, y_train)

test_score = accuracy_score(y_test, log_reg.predict(X_test)) * 100
train_score = accuracy_score(y_train, log_reg.predict(X_train)) * 100

tuning_results_df = pd.DataFrame(data=[["Tuned Logistic Regression", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
tuning_results_df




# kn neighbour

train_score = []
test_score = []
neighbors = range(1, 21)

for k in neighbors:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    train_score.append(accuracy_score(y_train, model.predict(X_train)))
    test_score.append(accuracy_score(y_test, model.predict(X_test)))


knn_classifier = KNeighborsClassifier(n_neighbors=19)
knn_classifier.fit(X_train, y_train)

test_score = accuracy_score(y_test, knn_classifier.predict(X_test)) * 100
train_score = accuracy_score(y_train, knn_classifier.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["Tuned K-nearest neighbors", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
tuning_results_df = tuning_results_df.append(results_df_2, ignore_index=True)
tuning_results_df


# Random forest

rand_forest = RandomForestClassifier(n_estimators=1900, random_state=15)
rand_forest.fit(X_train, y_train)

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

rand_forest = RandomForestClassifier(random_state=42)

rf_random = RandomizedSearchCV(estimator=rand_forest, param_distributions=random_grid, n_iter=100, cv=3, 
                               verbose=2, random_state=42, n_jobs=-1)


rf_random.fit(X_train, y_train)



rand_forest = RandomForestClassifier(bootstrap=True,
                                     max_depth=70, 
                                     max_features='auto', 
                                     min_samples_leaf=4, 
                                     min_samples_split=10,
                                     n_estimators=400)
rand_forest.fit(X_train, y_train)

test_score = accuracy_score(y_test, rand_forest.predict(X_test)) * 100
train_score = accuracy_score(y_train, rand_forest.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["Tuned Random Forest Classifier", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
tuning_results_df = tuning_results_df.append(results_df_2, ignore_index=True)
tuning_results_df

# Optimization

from statsmodels.tools import add_constant as add_constant
heart_df_constant = add_constant(hd)
heart_df_constant.head()

import statsmodels.api as sm
import scipy.stats as st

st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
cols=heart_df_constant.columns[:-1]
model=sm.Logit(hd.target,heart_df_constant[cols])
result=model.fit()
result.summary()




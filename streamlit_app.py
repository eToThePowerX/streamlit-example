import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data_set = pd.read_csv('Dataset.txt',sep= '\t', header=None)
data_copy = data_set.copy()

target = data_set[61]
data_set = data_set.drop(columns = 61)

X_train, X_test, y_train, y_test = train_test_split(data_set,target, test_size=0.33, random_state=42)

##Catboost
clf = CatBoostClassifier()
clf.fit(X_train, y_train,eval_set=(X_test, y_test),use_best_model=True,verbose = False)

st.write(classification_report(y_train, clf.predict(X_train)))



#chart_data = pd.DataFrame(
 #    np.random.randn(20, 3),
#     columns=['a', 'b', 'c'])

#st.line_chart(chart_data)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.tree import export_graphviz as export
import numpy as np
from sklearn.metrics import roc_curve as roc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz as visual

df = pd.read_csv("C:\\Users\\a\\PycharmProjects\\pythonProject\\titanic.csv")
# print(df)

#Creating a new column named Male and it will give male=True and female=False
df["Male"]=df["Sex"]=="male"
# print(df)
#Features defining

x = df.drop(columns=["Survived","Sex"],axis=1).values
#Target defining
y = df["Survived"].values

#split the datasets
x_train,x_test,y_train,y_test = train_test_split(x,y)

# print(y_test.shape)

modelDt = DecisionTreeClassifier(max_depth=10,min_samples_leaf=15,max_leaf_nodes=5)
modelLgr = LogisticRegression()

modelDt.fit(x_train,y_train)
modelLgr.fit(x_train,y_train)

y_predDt = modelDt.predict_proba(x_test)
y_predLgr = modelLgr.predict_proba(x_test)

# fprDt,tprDt,thresholds = roc(y_test,y_predDt[:,1])
# fprLgr,tprLgr,thresholds = roc(y_test,y_predLgr[:,1])
# plt.ylabel("Sensitivity")
# plt.xlabel("1-Specificity")
# plt.plot(fprDt,tprDt)
# plt.plot(fprLgr,tprLgr,color='red')
# plt.plot([0,1],[0,1],linestyle = '--')
# plt.show()

features = ['Pclass','Male','Age','Siblings/Spouses','Parents/Children','Fare'];
def draw(model,features,name):
    dot_file = export(model,feature_names=features)
    graph = visual.Source(dot_file)
    graph.render(name,format='png',cleanup=True)
    return 0
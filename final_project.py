#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATS 131: Final Project
"""
# Package imports
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import statsmodels.api as sm
from sklearn import linear_model, metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

#%%
#Settings
np.set_printoptions(suppress=True)


#%%
#Functions

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def process_cm(confusion_mat, i=0, to_print=True):
    # i means which class to choose to do one-vs-the-rest calculation
    # rows are actual obs whereas columns are predictions
    TP = confusion_mat[i,i]  # correctly labeled as i
    FP = confusion_mat[:,i].sum() - TP  # incorrectly labeled as i
    FN = confusion_mat[i,:].sum() - TP  # incorrectly labeled as non-i
    TN = confusion_mat.sum().sum() - TP - FP - FN
    TPR = TP/(TP+FN) #Sensitivity (Recall)
    FPR = FP/(FP+TN) #False positive rate
    TNR = TN/(FP+TN) #Specificity
    Precision = TP/(TP+FP) #Precision 





    if to_print:
        print('TP: {}'.format(TP))
        print('FP: {}'.format(FP))
        print('FN: {}'.format(FN))
        print('TN: {}'.format(TN))
        print("Sensitivity: {}".format(TPR))
        print("FPR: {}".format(FPR))
        print("Speicificity: {}".format(TNR))
        print("Precision: {}:".format(Precision))
    return TP, FP, FN, TN, TPR, FPR, TNR,Precision


#%%

df = pd.read_csv('A://Spring 2018/Stats 131/Final Project/Most-Recent-Cohorts-All-Data-Elements.csv')
location = pd.read_csv("A://Spring 2018/Stats 131/Final Project/State.csv")
#%%


cols = [ "STABBR","MD_EARN_WNE_P10", "MD_INC_DEBT_MDN", "LO_INC_DEBT_MDN", "HIGHDEG","DEBT_MDN","PREDDEG", "AVGFACSAL","COSTT4_A","CONTROL","HCM2","D_PCTPELL_PCTFLOAN"]


income = df[cols]
income = income[income != 'PrivacySuppressed']
income = income.dropna()

#Drop these?
income.shape
income = income.drop(income.loc[income["HCM2"] == 1,:].index)
income.shape

#class change strings into numeric
income[cols[1:]] = income[cols[1:]].astype(float)

income["HIGHDEG"] = income["HIGHDEG"].replace({0:"Non-degree-granting",
      1:"certificate Degree",
      2:"Associate degree",
      3:"Bachelor's degree",
      4:"Graduate degree"})

income["CONTROL"] = income["CONTROL"].replace({1:"Public",
      2:"Private nonprofit ",
      3:"Private for-profit"})

income["PREDDEG"] = income["PREDDEG"].replace({0:"Not classified",
      1:"Predominantly certificate-degree granting",
      2:"Predominantly associate's-degree granting",
      3:"Predominantly bachelor's-degree granting",
      4:"Entirely graduate-degree granting"})


kwargs = dict( data = income, scatter_kws={"alpha":0.3}, ci = False, fit_reg = False)

#p = sns.lmplot(x = "MD_EARN_WNE_P10", y = "LO_INC_DEBT_MDN", **kwargs, hue = "HIGHDEG")
#plt.xlabel("Median Earning of studetns after 10 years")
#legend = p._legend
#legend.set_title("Highest degree awarded")
#for t, l in zip(legend.texts, ("Non-degree-granting","certificate Degree","Associate degree","Bachelor's degree","Graduate degree")):
#   t.set_text(l)

g = sns.lmplot(x = "MD_EARN_WNE_P10", y = "LO_INC_DEBT_MDN", **kwargs, hue = "CONTROL", col = "HIGHDEG", col_wrap = 2)
g = (g.set_axis_labels("Median Earning of studetns after 10 years", "Low Income Median Debt"),
     g.set_titles("{col_name}"))


g = sns.lmplot(x = "MD_EARN_WNE_P10", y = "MD_INC_DEBT_MDN", **kwargs, hue = "CONTROL", col = "HIGHDEG", col_wrap = 2)
g = (g.set_axis_labels("Median Earning of studetns after 10 years", "Median Income Median Debt"),
     g.set_titles("{col_name}"))
plt.show()


#PREDDEG
g = sns.lmplot(x = "MD_EARN_WNE_P10", y = "LO_INC_DEBT_MDN", **kwargs, hue = "CONTROL", col = "PREDDEG", col_wrap = 2)
g = (g.set_axis_labels("Median Earning of studetns after 10 years", "Low Income Median Debt"),
     g.set_titles("{col_name}"))

g = sns.lmplot(x = "MD_EARN_WNE_P10", y = "MD_INC_DEBT_MDN", **kwargs, hue = "CONTROL", col = "PREDDEG", col_wrap = 2)
g = (g.set_axis_labels("Median Earning of studetns after 10 years", "Median Income Median Debt"),
     g.set_titles("{col_name}"))
plt.show()



#Low income and Median income was pretty similar .....
#%%

#%%
#DEBT RATION
income["LO_INC_DEBT_RATIO"] = income["LO_INC_DEBT_MDN"]/income["DEBT_MDN"]
income["MD_INC_DEBT_RATIO"] = income["MD_INC_DEBT_MDN"]/income["DEBT_MDN"]

income["LO_INC_DEBT_RATIO"].describe()
income["MD_INC_DEBT_RATIO"].describe()

#Low income bracket

g = sns.lmplot(x = "LO_INC_DEBT_RATIO", y = "MD_EARN_WNE_P10", **kwargs, hue = "CONTROL", col = "PREDDEG", col_wrap = 2)
g = (g.set_axis_labels("Low Income Median Debt vs Overall Median Debt Ratio","Median Earning of studetns after 10 years"),
     g.set_titles("{col_name}"))
plt.show()
#Need a zooming at bechelor degree level 

g = sns.lmplot(data = income[income["PREDDEG"] == "Predominantly bachelor's-degree granting"], 
               x = "LO_INC_DEBT_RATIO", y = "MD_EARN_WNE_P10",
               hue = "CONTROL", col = "CONTROL", col_wrap = 2, ci = False, fit_reg = False)
g = (g.set_axis_labels("Low Income Median Debt vs Overall Median Debt Ratio","Median Earning of studetns after 10 years"),
     g.set_titles("{col_name}"))
plt.show()
print(income.loc[:,["LO_INC_DEBT_RATIO","CONTROL","PREDDEG","MD_EARN_WNE_P10"]].groupby(["CONTROL", "PREDDEG"]).mean())

#Privat-profit: High variance of debt ratio. On average, low income student carry less debt compare to overall student
#but the mean of median income is lower than the other two bracket

#Public school: low incoe student who wants a bachelor degree usudally carry around the same debt as the overall student 
#but when it comes to lower degree, low income student carry more debt than overall student. 

#private non profit: similar to public school, but student graduate from private non-profit typically earns more than the other school type.

#Why privat-profit is better??? is it because people who are in the lower income bracket only go to private-profit school
#when they offer good scholarship?   


#Median income bracket

g = sns.lmplot(x = "MD_INC_DEBT_RATIO", y = "MD_EARN_WNE_P10", **kwargs, hue = "CONTROL", col = "PREDDEG", col_wrap = 2)
g = (g.set_axis_labels("Low Income Median Debt vs Over Median Debt Ratio","Median Earning of studetns after 10 years"),
     g.set_titles("{col_name}"))
plt.show()
#Need a zooming at bechelor degree level 

g = sns.lmplot(data = income[income["PREDDEG"] == "Predominantly bachelor's-degree granting"], 
               x = "MD_INC_DEBT_RATIO", y = "MD_EARN_WNE_P10",
               hue = "CONTROL", col = "CONTROL", col_wrap = 2, ci = False, fit_reg = False)
g = (g.set_axis_labels("Median Income Median Debt vs Overall Median Debt Ratio","Median Earning of studetns after 10 years"),
     g.set_titles("{col_name}"))
plt.show()
print(income.loc[:,["MD_INC_DEBT_RATIO","CONTROL","PREDDEG", "MD_EARN_WNE_P10"]].groupby(["CONTROL", "PREDDEG"]).mean())

#Private-profit: the median income group carry a lot more debt than the lower income group,
# and student graduate from this group do not earn that much. 
#It's not a good idea for median income student to attend this type of school


#Public school: The median income bracket generally carry slightly less debt in predominatly degree lower than bachelor school group.
#The rest of the degree groups are slightly above one. This might due to the low cost of community college.

#private non-profit: Except for predominately degree lower than bachlor school group, the ration is similar to public school. 
#student graduate from private non-profit typically earns more than the other school type.  


 
 
#%%


plotly.tools.set_credentials_file(username='f128482000', api_key='jXtBjmGtpEoXrzjrUiq1')

#Low income ratio and region


print(income.loc[:,["LO_INC_DEBT_RATIO","STABBR", "MD_EARN_WNE_P10"]].groupby("STABBR").mean())

LO_MEAN = income.loc[:,["LO_INC_DEBT_RATIO","STABBR", "MD_EARN_WNE_P10"]].groupby("STABBR").mean().reset_index()
LO_MEAN

size = (LO_MEAN['MD_EARN_WNE_P10']- np.min(LO_MEAN['MD_EARN_WNE_P10']))/(np.max(LO_MEAN['MD_EARN_WNE_P10']) - np.min(LO_MEAN["MD_EARN_WNE_P10"]))*100
size = size.astype(int)



scale = [[0, 'rgb(255,245,240)'], [0.2, 'rgb(254,224,210)'], [0.4, 'rgb(252,187,161)'], [0.5, 'rgb(252,146,114)'], [0.6, 'rgb(251,106,74)'], [0.7, 'rgb(239,59,44)'], [0.8, 'rgb(203,24,29)'], [0.9, 'rgb(165,15,21)'], [1, 'rgb(103,0,13)']]

map = [ go.Choropleth(
        type='choropleth',
        autocolorscale = False,
        locations = LO_MEAN['STABBR'],
        z = LO_MEAN["LO_INC_DEBT_RATIO"].astype(float),
        locationmode = 'USA-states',
        text = np.array(["Median Earning of studetns after 10 years: " + str(x) for x in LO_MEAN['MD_EARN_WNE_P10']]),
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Ratio",
            x = 0.9,
            thickness = 10,
            len = 0.2)
        ) ,
        go.Scattergeo(
                lon = location["Longitude"],
                lat = location["Latitude"],
                mode = "markers+text",
                text = np.array(LO_MEAN['MD_EARN_WNE_P10']).round(),
                marker = dict(
                        size = size,
                        line = dict(width = 0),
                        autocolorscale = False,
                        colorscale = scale,
                        color = LO_MEAN["MD_EARN_WNE_P10"],
                        cmin = LO_MEAN["MD_EARN_WNE_P10"].min(),
                        cmax = LO_MEAN["MD_EARN_WNE_P10"].max(),
                        colorbar = dict(
                            title = "Median Earning",
                            x = 0.95,
                            thickness = 10,
                            len = 0.2))
                )
        ]

layout = go.Layout(
        title = 'Low Income Median Debt vs Overall Debt Ration by US Territory',
        autosize=False,
        width=2000,
        height=2000,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = go.Figure(data = map, layout=layout)

#py.iplot(fig, filename='d3-cloropleth-MD-map')
py.image.ishow(fig)
#plotly.offline.plot(fig)

#%%

#%%
#median income ratio and region

print(income.loc[:,["MD_INC_DEBT_RATIO","STABBR", "MD_EARN_WNE_P10"]].groupby("STABBR").mean())

MD_MEAN = income.loc[:,["MD_INC_DEBT_RATIO","STABBR", "MD_EARN_WNE_P10"]].groupby("STABBR").mean().reset_index()
MD_MEAN

size = (MD_MEAN['MD_EARN_WNE_P10']- np.min(MD_MEAN['MD_EARN_WNE_P10']))/(np.max(MD_MEAN['MD_EARN_WNE_P10']) - np.min(MD_MEAN["MD_EARN_WNE_P10"]))*100
size = size.astype(int)



scale = [[0, 'rgb(255,245,240)'], [0.2, 'rgb(254,224,210)'], [0.4, 'rgb(252,187,161)'], [0.5, 'rgb(252,146,114)'], [0.6, 'rgb(251,106,74)'], [0.7, 'rgb(239,59,44)'], [0.8, 'rgb(203,24,29)'], [0.9, 'rgb(165,15,21)'], [1, 'rgb(103,0,13)']]

map = [ go.Choropleth(
        type='choropleth',
        autocolorscale = False,
        locations = MD_MEAN['STABBR'],
        z = MD_MEAN["MD_INC_DEBT_RATIO"].astype(float),
        locationmode = 'USA-states',
        text = np.array(["Median Earning of studetns after 10 years: " + str(x) for x in MD_MEAN['MD_EARN_WNE_P10']]),
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Ratio",
            x = 0.9,
            thickness = 10,
            len = 0.2)
        ) ,
        go.Scattergeo(
                lon = location["Longitude"],
                lat = location["Latitude"],
                mode = "markers+text",
                text = np.array(MD_MEAN['MD_EARN_WNE_P10']).round(),
                marker = dict(
                        size = size,
                        line = dict(width = 0),
                        autocolorscale = False,
                        colorscale = scale,
                        color = MD_MEAN["MD_EARN_WNE_P10"],
                        cmin = MD_MEAN["MD_EARN_WNE_P10"].min(),
                        cmax = MD_MEAN["MD_EARN_WNE_P10"].max(),
                        colorbar = dict(
                            title = "Median Earning",
                            x = 0.95,
                            thickness = 10,
                            len = 0.2))
                )
        ]

layout = go.Layout(
        title = 'Median Income Median Debt vs Overall Debt Ration by US Territory',
        autosize=False,
        width=2000,
        height=2000,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = go.Figure(data = map, layout=layout)

#py.iplot(fig, filename='d3-cloropleth-MD-map')
py.image.ishow(fig)
#plotly.offline.plot(fig)

#%%


#%%

# logistic regression

#We want to do a logistic regression model,but school type has three cases

#We merged private profit and private non-profit into the same group.
subset = cols[1:12]
#subset.append("MD_INC_DEBT_RATIO")
X = income[subset]
X = X.drop(columns = ["HCM2","HIGHDEG", "MD_INC_DEBT_MDN", "LO_INC_DEBT_MDN", "MD_INC_DEBT_MDN","AVGFACSAL","CONTROL"])

y = income.iloc[:,-5]

print(X.corr())


X["PREDDEG"] = X["PREDDEG"].replace({"Not classified":0,
 "Predominantly certificate-degree granting":1,
 "Predominantly associate's-degree granting":2,
 "Predominantly bachelor's-degree granting":3,
 "Entirely graduate-degree granting":4})

#X["PREDDEG"] = X["PREDDEG"].astype("category")

X.info()
    
sns.pairplot(X)
plt.show()
plt.clf()
y = y.replace({"Public":0,
               "Private nonprofit ":1,
               "Private for-profit":1})
y = y.astype("category")


train_x, test_x, train_y, test_y = train_test_split(X,y, train_size=0.7, random_state = 20)

log = linear_model.LogisticRegression(C=1e8)
log.fit(train_x, train_y)
print(np.exp(log.coef_))

    
    
class_name = ["Public","Privite"]

print(metrics.accuracy_score(test_y, log.predict(test_x)))
cm = confusion_matrix(test_y, log.predict(test_x))
plot_confusion_matrix(cm,class_name)
plt.show()
plt.clf()


logit_model = sm.Logit(train_y,train_x)
result = logit_model.fit()
print(result.summary2())

np.exp(result.params)

#%%
#ROC analysis


logit_roc_auc = roc_auc_score(test_y, log.predict(test_x))
fpr, tpr, thresholds = roc_curve(test_y, log.predict_proba(test_x)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


#%%
#CM stats

process_cm(cm)

#public
print("Number of public schools:",y[y==1].count())
#private
print("Number of private schools:",y[y==2].count())
#%%

#K-fold
param_grid = {}
log_cv = GridSearchCV(log, param_grid, cv = 10)
log_cv.fit(train_x, train_y)
print(log_cv.cv_results_)
print(log_cv.grid_scores_)
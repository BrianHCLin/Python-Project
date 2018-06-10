#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATS 131: Final Project
"""

# Package imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.plotly as py
import plotly.graph_objs as go


#%%

df = pd.read_csv('A://Spring 2018/Stats 131/Final Project/Most-Recent-Cohorts-All-Data-Elements.csv')
location = pd.read_csv("A://Spring 2018/Stats 131/Final Project/State.csv")
#%%


cols = [ "STABBR","MD_EARN_WNE_P10", "MD_INC_DEBT_MDN", "LO_INC_DEBT_MDN", "HIGHDEG","DEBT_MDN", "CONTROL","PREDDEG", "HCM2"]

cols[1:]

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


plotly.tools.set_credentials_file(username='f128481212', api_key='039hw6eTG72Z8BfA1A6Z')

#Low income ratio and region


print(income.loc[:,["LO_INC_DEBT_RATIO","STABBR", "MD_EARN_WNE_P10"]].groupby("STABBR").mean())

LO_MEAN = income.loc[:,["LO_INC_DEBT_RATIO","STABBR", "MD_EARN_WNE_P10"]].groupby("STABBR").mean().reset_index()
LO_MEAN
map = [ go.Choropleth(
        type='choropleth',
        colorscale = "coolwarm",
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
            title = "Ratio")
        ) ]

layout = go.Layout(
        title = 'Low Income Median Debt vs Overall Debt Ration by Territory',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = go.Figure(data = map, layout=layout )
py.iplot(fig, filename='d3-cloropleth-LO-map')
py.image.ishow(fig)
plotly.offline.plot(fig)

#Blue = Better, Red = BAD


#median income ratio and region

print(income.loc[:,["MD_INC_DEBT_RATIO","STABBR", "MD_EARN_WNE_P10"]].groupby("STABBR").mean())

MD_MEAN = income.loc[:,["MD_INC_DEBT_RATIO","STABBR", "MD_EARN_WNE_P10"]].groupby("STABBR").mean().reset_index()
MD_MEAN

map = [ go.Choropleth(
        type='choropleth',
        colorscale = "coolwarm",
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
            title = "Ratio")
        ) ]

layout = go.Layout(
        title = 'Median Income Median Debt vs Overall Debt Ration by US Territory',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = go.Figure(data = map, layout=layout )
py.iplot(fig, filename='d3-cloropleth-MD-map')
py.image.ishow(fig)
plotly.offline.plot(fig)


#%%

#CIP01ASSOC  is it worth to attend online program

#%%


map = [ go.Choropleth(
        type='choropleth',
        colorscale = "coolwarm",
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
            title = "Ratio")
        ) ,
        go.Scattergeo(
                lon = location["Longitude"],
                lat = location["Latitude"],
                mode = "markers+text",
                text = np.array(MD_MEAN['MD_EARN_WNE_P10']).round(),
                marker = dict(
                        size = ((MD_MEAN['MD_EARN_WNE_P10']- np.mean(MD_MEAN['MD_EARN_WNE_P10']))/np.std(MD_MEAN['MD_EARN_WNE_P10']))*20,
                        color = "Blue",
                        line = dict(width = 0)))
        ]

layout = go.Layout(
        title = 'Median Income Median Debt vs Overall Debt Ration by US Territory',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = go.Figure(data = map, layout=layout )

py.image.ishow(fig)

#%%
#%%
#%%

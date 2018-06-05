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
#%%
# Import the dataset
#df = pd.read_csv('Stats 131/Final Project/CollegeScorecard_Raw_Data/MERGED2014_15_PP.csv', low_memory=False)

#%%

df = pd.read_csv('A://Spring 2018/Stats 131/Final Project/Most-Recent-Cohorts-All-Data-Elements.csv')
#%%

df.columns.tolist()[0:25]

#%%

cols = ['INSTNM', 'CITY', 'STABBR', 'ZIP', 'NUMBRANCH', 'HIGHDEG']
df_basic_info = df[cols]

#%%

cols = ['INSTNM', 'STABBR', 'UGDS', 'AVGFACSAL', 'LOCALE2',
        'PAR_ED_PCT_PS', 'PCT_WHITE', 'PCT_BLACK']
df_big = df[cols]


ca_schools = df_big.loc[(df['UGDS'].notnull()) & (df['STABBR'] == 'CA')]
ca_schools = ca_schools.sort_values(by='UGDS', axis=0, ascending=False)
ca_schools.loc[:, ['INSTNM', 'LOCALE2'] ]

#%%
# Average cost of attendance vs median earnings of students after 10 years

cols = ['COSTT4_A', 'MD_EARN_WNE_P10', 'CONTROL']

df_return = df[cols]

# Filter out PrivacySuppressed
df_return = df_return[(df_return['MD_EARN_WNE_P10'] != 'PrivacySuppressed')]
df_return['COSTT4_A'] = pd.to_numeric(df_return['COSTT4_A'])
df_return['MD_EARN_WNE_P10'] = pd.to_numeric(df_return['MD_EARN_WNE_P10'])

df_return.plot(x='MD_EARN_WNE_P10', y='COSTT4_A', c='CONTROL', kind='scatter', colormap='winter', alpha = 0.5)

#%%
# 1	Public
# 2	Private nonprofit
# 3	Private for-profit

groups = df_return.groupby('CONTROL')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.MD_EARN_WNE_P10, group.COSTT4_A, linestyle='', marker='o', label=name, alpha = 0.5)
ax.legend()
plt.show()


#%%
sns.lmplot(x='MD_EARN_WNE_P10', y='COSTT4_A', data=df_return, fit_reg=False,
           hue='CONTROL')


#%%
sns.lmplot(x='MD_EARN_WNE_P10', y='COSTT4_A', data=df_return, fit_reg=False,
           hue='CONTROL', col='CONTROL', aspect=0.7)

#%%

#Low income debt vs. median earnings of students after 10 years

#HIGHDEG is the highest category of award conferred by the institution, 
#in descending order of graduate degree/certificate, bachelor’s degree, associate’s degree, 
#and certificate, calculated from the IPEDS Completions component.

cols = ["MD_EARN_WNE_P10", "LO_INC_DEBT_MDN", "HIGHDEG", "CITY","DEBT_MDN"]

low_income = df[cols]
low_income = low_income[low_income != 'PrivacySuppressed']
low_income = low_income.dropna()


low_income["MD_EARN_WNE_P10"] = pd.to_numeric(low_income["MD_EARN_WNE_P10"])
low_income["LO_INC_DEBT_MDN"] = pd.to_numeric(low_income["LO_INC_DEBT_MDN"])
low_income["HIGHDEG"] = pd.to_numeric(low_income["HIGHDEG"])
low_income["DEBT_MDN"] = pd.to_numeric(low_income["DEBT_MDN"])
low_income.plot(x = "MD_EARN_WNE_P10", y = "LO_INC_DEBT_MDN", kind = "scatter", c = "HIGHDEG", colormap = "winter", alpha = 0.5)
plt.show()

#Compare the low income student median debt and overall student median debt 
#greater than 1 = low income student has more debt than the overall median debt.
low_income["DEBT_RATIO"] = low_income["LO_INC_DEBT_MDN"]/low_income["DEBT_MDN"]

RATIO = low_income.loc[:,["DEBT_RATIO","CITY"]].groupby("CITY").mean().sort_values("DEBT_RATIO", ascending=True)

print(RATIO.head(50))
#%%

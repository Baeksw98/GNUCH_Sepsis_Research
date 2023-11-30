"""
Created on November 16th, 2021
@author: Sangwon Baek
KyeongsangUniversity Bacteremia Analysis Code  
"""

import numpy as np
import pandas as pd 
import os
from scipy.stats import kstest
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import contingency_tables
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import warnings
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

#setting working directory to the folder, use if putting the py file into the working folder does not work 
#This code will set the working directory to the file that you wish to work with 
path=r"C:\Users\user-pc\Desktop\KSU\Sepsis Research\Data\CSV"
os.chdir(path)
os.getcwd()
warnings.filterwarnings('ignore')

#Reading the files
BloodCultureData = pd.read_csv('BacteremiaData(2019).csv', encoding='cp949')
ClinicalData = pd.read_csv('ClinicalDataWithScores.csv', encoding='cp949')
ClinicalData = ClinicalData.drop(['Unnamed: 0','Registration Number', 'Age', 'Bacteremia Result'], axis=1)

#Merge the Clinical Data and Original Data
BacteremiaData = pd.merge(ClinicalData, BloodCultureData, on=['Serial Number'], how = 'left')

#Remove non-septic patients with SOFA score < 2
BacteremiaData = BacteremiaData[BacteremiaData['SOFA score']>=2]

#Select only the Bacteremia Positivse Patients from Total Patients from BacteremiaData data frame
PositivePatients = BacteremiaData.loc[BacteremiaData["Bacteremia Result"]=='Positive']

#Select only the Bacteremia Negative Patients from Total Patients from BacteremiaData data frame
NegativePatients = BacteremiaData.loc[BacteremiaData["Bacteremia Result"]=='Negative']
BacteremiaData = pd.concat([PositivePatients,NegativePatients],axis=0)
BacteremiaData = BacteremiaData.reset_index()
BacteremiaData = BacteremiaData.drop(['index'], axis=1)

#Creating a copy of BacteremiaData and name it Tester
Tester = BacteremiaData.copy()

#Select only the necessary columns in an organized manner
ColumnNames = ['Serial Number', 'Age', 'Gender', 'Bacteremia Result dummy', 'Bacteremia Result', 'Death Status', 'Time to Death',
                 'BUN', 'MAP', 'Na', 'GCS Score', 'Neutrophil Lymphocyte Ratio', 'Creatinine', 'Lactate Level', 'AST',
                 'Bilirubin', 'ESR', 'Platelets','CRP(mg/L)', 'PCT(ng/ml)', 'APACHE II score', 'SOFA score']
Tester = Tester[ColumnNames]
PositivePatients = PositivePatients[ColumnNames]
NegativePatients = NegativePatients[ColumnNames]


#Running a Kolmogorov-Smirnov test (KS test) to check the normality of the distribution
#When the p-value is <0.01, we can reject the null hypothesis of normality: meaning the distribution is not normal. Then, use median & IQR 
k_statAge, k_pvalAge = kstest(Tester['Age'], 'norm')
k_statBUN, k_pvalBUN = kstest(Tester['BUN'], 'norm')
k_statMAP, k_pvalMAP = kstest(Tester['MAP'], 'norm')
k_statNa, k_pvalNa = kstest(Tester['Na'], 'norm')
k_statGCS, k_pvalGCS = kstest(Tester['GCS Score'], 'norm') 
k_statNLR, k_pvalNLR = kstest(Tester['Neutrophil Lymphocyte Ratio'], 'norm')
k_statCreatinine, k_pvalCreatinine = kstest(Tester['Creatinine'], 'norm')
k_statLactate, k_pvalLactate = kstest(Tester['Lactate Level'], 'norm')
k_statBilirubin, k_pvalBilirubin = kstest(Tester['Bilirubin'], 'norm')
k_statESR, k_pvalESR = kstest(Tester['ESR'], 'norm')
k_statPLT, k_pvalPLT = kstest(Tester['Platelets'], 'norm')
k_statCRP, k_pvalCRP = kstest(Tester['CRP(mg/L)'], 'norm')
k_statPCT, k_pvalPCT = kstest(Tester['PCT(ng/ml)'], 'norm')
k_statAST, k_pvalAST = kstest(Tester['AST'], 'norm')
k_statAPACHE, k_pvalAPACHE = kstest(Tester['APACHE II score'], 'norm')
k_statSOFA, k_pvalSOFA = kstest(Tester['SOFA score'], 'norm')

#________________Median Scaling and Test Set Variables_______________________________________________________
#Serial Number
SN = Tester['Serial Number']
BR = pd.DataFrame(Tester['Bacteremia Result dummy'], columns=['Bacteremia Result dummy'])

#Dependent Variable
Y = Tester['Bacteremia Result dummy']

BUN = pd.DataFrame(Tester['BUN'], columns=['BUN'])
MAP = pd.DataFrame(Tester['MAP'], columns=['MAP'])
Na = pd.DataFrame(Tester['Na'], columns=['Na'])
GCS = pd.DataFrame(Tester['GCS Score'], columns=['GCS Score'])
NLR = pd.DataFrame(Tester['Neutrophil Lymphocyte Ratio'], columns=['Neutrophil Lymphocyte Ratio'])
Creatinine = pd.DataFrame(Tester['Creatinine'], columns=['Creatinine'])
Lactate = pd.DataFrame(Tester['Lactate Level'], columns=['Lactate Level'])
Bilirubin = pd.DataFrame(Tester['Bilirubin'], columns=['Bilirubin'])
ESR = pd.DataFrame(Tester['ESR'], columns=['ESR'])
PLT = pd.DataFrame(Tester['Platelets'], columns=['Platelets'])
CRP = pd.DataFrame(Tester['CRP(mg/L)'], columns=['CRP(mg/L)'])
PCT = pd.DataFrame(Tester['PCT(ng/ml)'], columns=['PCT(ng/ml)'])
AST = pd.DataFrame(Tester['AST'], columns =['AST'])
APACHE = pd.DataFrame(Tester['APACHE II score'], columns=['APACHE II score'])
SOFA = pd.DataFrame(Tester['SOFA score'], columns=['SOFA score'])

#________________Univariate Logistic Regression Testing_______________________________________________________________________

# Creating a logistic regression model
LRmodel0 = smf.logit('Y ~ BUN', data=Tester).fit()
LRmodel1 = smf.logit('Y ~ MAP', data=Tester).fit()
LRmodel2 = smf.logit('Y ~ Na', data=Tester).fit()
LRmodel3 = smf.logit('Y ~ GCS', data=Tester).fit()
LRmodel4 = smf.logit('Y ~ NLR', data=Tester).fit()
LRmodel5 = smf.logit('Y ~ Creatinine', data=Tester).fit()
LRmodel6 = smf.logit('Y ~ Lactate', data=Tester).fit()
LRmodel7 = smf.logit('Y ~ Bilirubin', data=Tester).fit()
LRmodel8 = smf.logit('Y ~ ESR', data=Tester).fit()
LRmodel9 = smf.logit('Y ~ PLT', data=Tester).fit()
LRmodel10 = smf.logit('Y ~ CRP', data=Tester).fit()
LRmodel11 = smf.logit('Y ~ PCT', data=Tester).fit()
LRmodel12 = smf.logit('Y ~ AST', data=Tester).fit()
LRmodel13 = smf.logit('Y ~ APACHE', data=Tester).fit()
LRmodel14 = smf.logit('Y ~ SOFA', data=Tester).fit()

# Finding the predictions using the independent variables throughout the model
y_predLR0 = LRmodel0.predict(BUN)
y_predLR1 = LRmodel1.predict(MAP)
y_predLR2 = LRmodel2.predict(Na)
y_predLR3 = LRmodel3.predict(GCS)
y_predLR4 = LRmodel4.predict(NLR)
y_predLR5 = LRmodel5.predict(Creatinine)
y_predLR6 = LRmodel6.predict(Lactate)
y_predLR7 = LRmodel7.predict(Bilirubin)
y_predLR8 = LRmodel8.predict(ESR)
y_predLR9 = LRmodel9.predict(PLT)
y_predLR10 = LRmodel10.predict(CRP)
y_predLR11 = LRmodel11.predict(PCT)
y_predLR12 = LRmodel12.predict(AST)
y_predLR13 = LRmodel13.predict(APACHE)
y_predLR14 = LRmodel14.predict(SOFA)

# Finding false positive rate, true positive rate, thresholds, area under curve of ROC score
fpr0, tpr0, thresholds0 = roc_curve(y_true=Y, y_score=y_predLR0)
auc0 = roc_auc_score(Y, y_predLR0)
fpr1, tpr1, thresholds1 = roc_curve(y_true=Y, y_score=y_predLR1)
auc1 = roc_auc_score(Y, y_predLR1)
fpr2, tpr2, thresholds2 = roc_curve(y_true=Y, y_score=y_predLR2)
auc2 = roc_auc_score(Y, y_predLR2)
fpr3, tpr3, thresholds3 = roc_curve(y_true=Y, y_score=y_predLR3)
auc3 = roc_auc_score(Y, y_predLR3)
fpr4, tpr4, thresholds4 = roc_curve(y_true=Y, y_score=y_predLR4)
auc4 = roc_auc_score(Y, y_predLR4)
fpr5, tpr5, thresholds5 = roc_curve(y_true=Y, y_score=y_predLR5)
auc5 = roc_auc_score(Y, y_predLR5)
fpr6, tpr6, thresholds6 = roc_curve(y_true=Y, y_score=y_predLR6)
auc6 = roc_auc_score(Y, y_predLR6)
fpr7, tpr7, thresholds7 = roc_curve(y_true=Y, y_score=y_predLR7)
auc7 = roc_auc_score(Y, y_predLR7)
fpr8, tpr8, thresholds8 = roc_curve(y_true=Y, y_score=y_predLR8)
auc8 = roc_auc_score(Y, y_predLR8)
fpr9, tpr9, thresholds9 = roc_curve(y_true=Y, y_score=y_predLR9)
auc9 = roc_auc_score(Y, y_predLR9)
fpr10, tpr10, thresholds10 = roc_curve(y_true=Y, y_score=y_predLR10)
auc10 = roc_auc_score(Y, y_predLR10)
fpr11, tpr11, thresholds11 = roc_curve(y_true=Y, y_score=y_predLR11)
auc11 = roc_auc_score(Y, y_predLR11)
fpr12, tpr12, thresholds12 = roc_curve(y_true=Y, y_score=y_predLR12)
auc12 = roc_auc_score(Y, y_predLR12)
fpr13, tpr13, thresholds13 = roc_curve(y_true=Y, y_score=y_predLR13)
auc13 = roc_auc_score(Y, y_predLR13)
fpr14, tpr14, thresholds14 = roc_curve(y_true=Y, y_score=y_predLR14)
auc14 = roc_auc_score(Y, y_predLR14)

# Find optimal probability threshold
optimal_threshold0 = thresholds0[np.argmax(tpr0-fpr0)] 
optimal_threshold1 = thresholds1[np.argmax(tpr1-fpr1)]
optimal_threshold2 = thresholds2[np.argmax(tpr2-fpr2)]
optimal_threshold3 = thresholds3[np.argmax(tpr3-fpr3)] 
optimal_threshold4 = thresholds4[np.argmax(tpr4-fpr4)]
optimal_threshold5 = thresholds5[np.argmax(tpr5-fpr5)]
optimal_threshold6 = thresholds6[np.argmax(tpr6-fpr6)] 
optimal_threshold7 = thresholds7[np.argmax(tpr7-fpr7)]
optimal_threshold8 = thresholds8[np.argmax(tpr8-fpr8)]
optimal_threshold9 = thresholds9[np.argmax(tpr9-fpr9)] 
optimal_threshold10 = thresholds10[np.argmax(tpr10-fpr10)]
optimal_threshold11 = thresholds11[np.argmax(tpr11-fpr11)]
optimal_threshold12 = thresholds12[np.argmax(tpr12-fpr12)] 
optimal_threshold13 = thresholds13[np.argmax(tpr13-fpr13)]
optimal_threshold14 = thresholds14[np.argmax(tpr14-fpr14)]

#Logistic Regression's P-value
P_BUN = str(round(LRmodel0.pvalues[1],6))
P_MAP = str(round(LRmodel1.pvalues[1],3))
P_Na = str(round(LRmodel2.pvalues[1],3))
P_GCS = str(round(LRmodel3.pvalues[1],3))
P_NLR = str(round(LRmodel4.pvalues[1],7))
P_Creatinine = str(round(LRmodel5.pvalues[1],3))
P_Lactate = str(round(LRmodel6.pvalues[1],3))
P_Bilirubin = str(round(LRmodel7.pvalues[1],4))
P_ESR = str(round(LRmodel8.pvalues[1],3))
P_PLT = str(round(LRmodel9.pvalues[1],5))
P_CRP = str(round(LRmodel10.pvalues[1],9))
P_PCT = str(round(LRmodel11.pvalues[1],8))
P_AST = str(round(LRmodel12.pvalues[1],3))
P_APACHE = str(round(LRmodel13.pvalues[1],6))
P_SOFA = str(round(LRmodel14.pvalues[1],8))

# Logistic Regression's Odd Ratio and Confidence Interval
params0 = LRmodel0.params
conf0 = round(np.exp(LRmodel0.conf_int()),3)
conf0['Odds Ratio'] = round(np.exp(params0),3)
conf0.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf0=conf0.drop(index='Intercept')

params1 = LRmodel1.params
conf1 = round(np.exp(LRmodel1.conf_int()),3)
conf1['Odds Ratio'] = round(np.exp(params1),3)
conf1.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf1=conf1.drop(index='Intercept')

params2 = LRmodel2.params
conf2 = round(np.exp(LRmodel2.conf_int()),3)
conf2['Odds Ratio'] = round(np.exp(params2),3)
conf2.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf2=conf2.drop(index='Intercept')

params3 = LRmodel3.params
conf3 = round(np.exp(LRmodel3.conf_int()),3)
conf3['Odds Ratio'] = round(np.exp(params3),3)
conf3.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf3=conf3.drop(index='Intercept')

params4 = LRmodel4.params
conf4 = round(np.exp(LRmodel4.conf_int()),3)
conf4['Odds Ratio'] = round(np.exp(params4),3)
conf4.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf4=conf4.drop(index='Intercept')

params5 = LRmodel5.params
conf5 = round(np.exp(LRmodel5.conf_int()),3)
conf5['Odds Ratio'] = round(np.exp(params5),3)
conf5.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf5=conf5.drop(index='Intercept')

params6 = LRmodel6.params
conf6 = round(np.exp(LRmodel6.conf_int()),3)
conf6['Odds Ratio'] = round(np.exp(params6),3)
conf6.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf6=conf6.drop(index='Intercept')

params7 = LRmodel7.params
conf7 = round(np.exp(LRmodel7.conf_int()),3)
conf7['Odds Ratio'] = round(np.exp(params7),3)
conf7.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf7=conf7.drop(index='Intercept')

params8 = LRmodel8.params
conf8 = round(np.exp(LRmodel8.conf_int()),3)
conf8['Odds Ratio'] = round(np.exp(params8),3)
conf8.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf8=conf8.drop(index='Intercept')

params9 = LRmodel9.params
conf9 = round(np.exp(LRmodel9.conf_int()),3)
conf9['Odds Ratio'] = round(np.exp(params9),3)
conf9.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf9=conf9.drop(index='Intercept')

params10 = LRmodel10.params
conf10 = round(np.exp(LRmodel10.conf_int()),3)
conf10['Odds Ratio'] = round(np.exp(params10),3)
conf10.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf10=conf10.drop(index='Intercept')

params11 = LRmodel11.params
conf11 = round(np.exp(LRmodel11.conf_int()),3)
conf11['Odds Ratio'] = round(np.exp(params11),3)
conf11.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf11=conf11.drop(index='Intercept')

params12 = LRmodel12.params
conf12 = round(np.exp(LRmodel12.conf_int()),3)
conf12['Odds Ratio'] = round(np.exp(params12),3)
conf12.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf12=conf12.drop(index='Intercept')

params13 = LRmodel13.params
conf13 = round(np.exp(LRmodel13.conf_int()),3)
conf13['Odds Ratio'] = round(np.exp(params13),3)
conf13.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf13=conf13.drop(index='Intercept')

params14 = LRmodel14.params
conf14 = round(np.exp(LRmodel14.conf_int()),3)
conf14['Odds Ratio'] = round(np.exp(params14),3)
conf14.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf14=conf14.drop(index='Intercept')

conf0=conf0.append(conf1)
conf0=conf0.append(conf2)
conf0=conf0.append(conf3)
conf0=conf0.append(conf4)
conf0=conf0.append(conf5)
conf0=conf0.append(conf6)
conf0=conf0.append(conf7)
conf0=conf0.append(conf8)
conf0=conf0.append(conf9)
conf0=conf0.append(conf10)
conf0=conf0.append(conf11)
conf0=conf0.append(conf12)
conf0=conf0.append(conf13)
conf0=conf0.append(conf14)

UniOR=[]
for i in range(len(conf0)):
    input=str(conf0["Odds Ratio"].iloc[i]) + " [" + str(conf0["2.5%"].iloc[i]) + "-" + str(conf0["97.5%"].iloc[i]) + "]"
    UniOR.append(input)

#_______________Printing the results of Log Regression__________________________________________________

#AUC, Sensitivity, Specificity with confidence interval: Median + [UB - LB]: CRP, PCT, CRP&PCT 
rng_seed = 1
n_bootstraps = 100
rng = np.random.RandomState(rng_seed)

#BUN
bootstrapped_auc0 = []  
bootstrapped_fpr0 = []
bootstrapped_tpr0 = []   
for i in range(n_bootstraps):
    indices0 = rng.randint(0, len(y_predLR0), len(y_predLR0))  
    if len(np.unique(Y[indices0])) < 2:
        continue
    auc_score0 = roc_auc_score(Y[indices0], y_predLR0[indices0])
    fpr_score0, tpr_score0, th0 = roc_curve(Y[indices0], y_predLR0[indices0])
    bootstrapped_auc0.append(auc_score0)
    bootstrapped_fpr0.append(1-fpr_score0[np.argmax(tpr_score0-fpr_score0)])
    bootstrapped_tpr0.append(tpr_score0[np.argmax(tpr_score0-fpr_score0)])
auc_sorted0 = np.array(bootstrapped_auc0)
auc_sorted0.sort()
fpr_sorted0 = np.array(bootstrapped_fpr0)
fpr_sorted0.sort()
tpr_sorted0 = np.array(bootstrapped_tpr0)
tpr_sorted0.sort()
AUC_BUN = str(round(np.median(auc_sorted0),3)) + ' [' + str(round(auc_sorted0[int(0.025 * len(auc_sorted0))],3)) + '-' + str(round(auc_sorted0[int(0.975 * len(auc_sorted0))],3)) + ']'
SEN_BUN = str(round(np.median(tpr_sorted0)*100,1)) + '% [' + str(round(tpr_sorted0[int(0.025 * len(tpr_sorted0))]*100,1)) + '-' + str(round(tpr_sorted0[int(0.975 * len(tpr_sorted0))]*100,1)) + ']'
SPE_BUN = str(round(np.median(fpr_sorted0)*100,1)) + '% [' + str(round(fpr_sorted0[int(0.025 * len(fpr_sorted0))]*100,1)) + '-' + str(round(fpr_sorted0[int(0.975 * len(fpr_sorted0))]*100,1)) + ']'

#MAP
bootstrapped_auc1 = []  
bootstrapped_fpr1 = []
bootstrapped_tpr1 = []   
for i in range(n_bootstraps):
    indices1 = rng.randint(0, len(y_predLR1), len(y_predLR1))  
    if len(np.unique(Y[indices1])) < 2:
        continue
    auc_score1 = roc_auc_score(Y[indices1], y_predLR1[indices1])
    fpr_score1, tpr_score1, th1 = roc_curve(Y[indices1], y_predLR1[indices1])
    bootstrapped_auc1.append(auc_score1)
    bootstrapped_fpr1.append(1-fpr_score1[np.argmax(tpr_score1-fpr_score1)])
    bootstrapped_tpr1.append(tpr_score1[np.argmax(tpr_score1-fpr_score1)])
auc_sorted1 = np.array(bootstrapped_auc1)
auc_sorted1.sort()
fpr_sorted1 = np.array(bootstrapped_fpr1)
fpr_sorted1.sort()
tpr_sorted1 = np.array(bootstrapped_tpr1)
tpr_sorted1.sort()
AUC_MAP = str(round(np.median(auc_sorted1),3)) + ' [' + str(round(auc_sorted1[int(0.025 * len(auc_sorted1))],3)) + '-' + str(round(auc_sorted1[int(0.975 * len(auc_sorted1))],3)) + ']'
SEN_MAP = str(round(np.median(tpr_sorted1)*100,1)) + '% [' + str(round(tpr_sorted1[int(0.025 * len(tpr_sorted1))]*100,1)) + '-' + str(round(tpr_sorted1[int(0.975 * len(tpr_sorted1))]*100,1)) + ']'
SPE_MAP = str(round(np.median(fpr_sorted1)*100,1)) + '% [' + str(round(fpr_sorted1[int(0.025 * len(fpr_sorted1))]*100,1)) + '-' + str(round(fpr_sorted1[int(0.975 * len(fpr_sorted1))]*100,1)) + ']'

#Na
bootstrapped_auc2 = []  
bootstrapped_fpr2 = []
bootstrapped_tpr2 = []   
for i in range(n_bootstraps):
    indices2 = rng.randint(0, len(y_predLR2), len(y_predLR2))  
    if len(np.unique(Y[indices2])) < 2:
        continue
    auc_score2 = roc_auc_score(Y[indices2], y_predLR2[indices2])
    fpr_score2, tpr_score2, th2 = roc_curve(Y[indices2], y_predLR2[indices2])
    bootstrapped_auc2.append(auc_score2)
    bootstrapped_fpr2.append(1-fpr_score2[np.argmax(tpr_score2-fpr_score2)])
    bootstrapped_tpr2.append(tpr_score2[np.argmax(tpr_score2-fpr_score2)])
auc_sorted2 = np.array(bootstrapped_auc2)
auc_sorted2.sort()
fpr_sorted2 = np.array(bootstrapped_fpr2)
fpr_sorted2.sort()
tpr_sorted2 = np.array(bootstrapped_tpr2)
tpr_sorted2.sort()
AUC_Na = str(round(np.median(auc_sorted2),3)) + ' [' + str(round(auc_sorted2[int(0.025 * len(auc_sorted2))],3)) + '-' + str(round(auc_sorted2[int(0.975 * len(auc_sorted2))],3)) + ']'
SEN_Na = str(round(np.median(tpr_sorted2)*100,1)) + '% [' + str(round(tpr_sorted2[int(0.025 * len(tpr_sorted2))]*100,1)) + '-' + str(round(tpr_sorted2[int(0.975 * len(tpr_sorted2))]*100,1)) + ']'
SPE_Na = str(round(np.median(fpr_sorted2)*100,1)) + '% [' + str(round(fpr_sorted2[int(0.025 * len(fpr_sorted2))]*100,1)) + '-' + str(round(fpr_sorted2[int(0.975 * len(fpr_sorted2))]*100,1)) + ']'

#GCS score
bootstrapped_auc3 = []  
bootstrapped_fpr3 = []
bootstrapped_tpr3 = []   
for i in range(n_bootstraps):
    indices3 = rng.randint(0, len(y_predLR3), len(y_predLR3))  
    if len(np.unique(Y[indices3])) < 2:
        continue
    auc_score3 = roc_auc_score(Y[indices3], y_predLR3[indices3])
    fpr_score3, tpr_score3, th3 = roc_curve(Y[indices3], y_predLR3[indices3])
    bootstrapped_auc3.append(auc_score3)
    bootstrapped_fpr3.append(1-fpr_score3[np.argmax(tpr_score3-fpr_score3)])
    bootstrapped_tpr3.append(tpr_score3[np.argmax(tpr_score3-fpr_score3)])
auc_sorted3 = np.array(bootstrapped_auc3)
auc_sorted3.sort()
fpr_sorted3 = np.array(bootstrapped_fpr3)
fpr_sorted3.sort()
tpr_sorted3 = np.array(bootstrapped_tpr3)
tpr_sorted3.sort()
AUC_GCS = str(round(np.median(auc_sorted3),3)) + ' [' + str(round(auc_sorted3[int(0.025 * len(auc_sorted3))],3)) + '-' + str(round(auc_sorted3[int(0.975 * len(auc_sorted3))],3)) + ']'
SEN_GCS = str(round(np.median(tpr_sorted3)*100,1)) + '% [' + str(round(tpr_sorted3[int(0.025 * len(tpr_sorted3))]*100,1)) + '-' + str(round(tpr_sorted3[int(0.975 * len(tpr_sorted3))]*100,1)) + ']'
SPE_GCS = str(round(np.median(fpr_sorted3)*100,1)) + '% [' + str(round(fpr_sorted3[int(0.025 * len(fpr_sorted3))]*100,1)) + '-' + str(round(fpr_sorted3[int(0.975 * len(fpr_sorted3))]*100,1)) + ']'

#Neutrophil Lymphocyte Ratio
bootstrapped_auc4 = []  
bootstrapped_fpr4 = []
bootstrapped_tpr4 = []   
for i in range(n_bootstraps):
    indices4 = rng.randint(0, len(y_predLR4), len(y_predLR4))  
    if len(np.unique(Y[indices4])) < 2:
        continue
    auc_score4 = roc_auc_score(Y[indices4], y_predLR4[indices4])
    fpr_score4, tpr_score4, th4 = roc_curve(Y[indices4], y_predLR4[indices4])
    bootstrapped_auc4.append(auc_score4)
    bootstrapped_fpr4.append(1-fpr_score4[np.argmax(tpr_score4-fpr_score4)])
    bootstrapped_tpr4.append(tpr_score4[np.argmax(tpr_score4-fpr_score4)])
auc_sorted4 = np.array(bootstrapped_auc4)
auc_sorted4.sort()
fpr_sorted4 = np.array(bootstrapped_fpr4)
fpr_sorted4.sort()
tpr_sorted4 = np.array(bootstrapped_tpr4)
tpr_sorted4.sort()
AUC_NLR = str(round(np.median(auc_sorted4),3)) + ' [' + str(round(auc_sorted4[int(0.025 * len(auc_sorted4))],3)) + '-' + str(round(auc_sorted4[int(0.975 * len(auc_sorted4))],3)) + ']'
SEN_NLR = str(round(np.median(tpr_sorted4)*100,1)) + '% [' + str(round(tpr_sorted4[int(0.025 * len(tpr_sorted4))]*100,1)) + '-' + str(round(tpr_sorted4[int(0.975 * len(tpr_sorted4))]*100,1)) + ']'
SPE_NLR = str(round(np.median(fpr_sorted4)*100,1)) + '% [' + str(round(fpr_sorted4[int(0.025 * len(fpr_sorted4))]*100,1)) + '-' + str(round(fpr_sorted4[int(0.975 * len(fpr_sorted4))]*100,1)) + ']'

#Creatinine
bootstrapped_auc5 = []  
bootstrapped_fpr5 = []
bootstrapped_tpr5 = []   
for i in range(n_bootstraps):
    indices5 = rng.randint(0, len(y_predLR5), len(y_predLR5))  
    if len(np.unique(Y[indices5])) < 2:
        continue
    auc_score5 = roc_auc_score(Y[indices5], y_predLR5[indices5])
    fpr_score5, tpr_score5, th5 = roc_curve(Y[indices5], y_predLR5[indices5])
    bootstrapped_auc5.append(auc_score5)
    bootstrapped_fpr5.append(1-fpr_score5[np.argmax(tpr_score5-fpr_score5)])
    bootstrapped_tpr5.append(tpr_score5[np.argmax(tpr_score5-fpr_score5)])
auc_sorted5 = np.array(bootstrapped_auc5)
auc_sorted5.sort()
fpr_sorted5 = np.array(bootstrapped_fpr5)
fpr_sorted5.sort()
tpr_sorted5 = np.array(bootstrapped_tpr5)
tpr_sorted5.sort()
AUC_Creatinine = str(round(np.median(auc_sorted5),3)) + ' [' + str(round(auc_sorted5[int(0.025 * len(auc_sorted5))],3)) + '-' + str(round(auc_sorted5[int(0.975 * len(auc_sorted5))],3)) + ']'
SEN_Creatinine = str(round(np.median(tpr_sorted5)*100,1)) + '% [' + str(round(tpr_sorted5[int(0.025 * len(tpr_sorted5))]*100,1)) + '-' + str(round(tpr_sorted5[int(0.975 * len(tpr_sorted5))]*100,1)) + ']'
SPE_Creatinine = str(round(np.median(fpr_sorted5)*100,1)) + '% [' + str(round(fpr_sorted5[int(0.025 * len(fpr_sorted5))]*100,1)) + '-' + str(round(fpr_sorted5[int(0.975 * len(fpr_sorted5))]*100,1)) + ']'

#Lactate level
bootstrapped_auc6 = []  
bootstrapped_fpr6 = []
bootstrapped_tpr6 = []   
for i in range(n_bootstraps):
    indices6 = rng.randint(0, len(y_predLR6), len(y_predLR6))  
    if len(np.unique(Y[indices6])) < 2:
        continue
    auc_score6 = roc_auc_score(Y[indices6], y_predLR6[indices6])
    fpr_score6, tpr_score6, th6 = roc_curve(Y[indices6], y_predLR6[indices6])
    bootstrapped_auc6.append(auc_score6)
    bootstrapped_fpr6.append(1-fpr_score6[np.argmax(tpr_score6-fpr_score6)])
    bootstrapped_tpr6.append(tpr_score6[np.argmax(tpr_score6-fpr_score6)])
auc_sorted6 = np.array(bootstrapped_auc6)
auc_sorted6.sort()
fpr_sorted6 = np.array(bootstrapped_fpr6)
fpr_sorted6.sort()
tpr_sorted6 = np.array(bootstrapped_tpr6)
tpr_sorted6.sort()
AUC_Lactate = str(round(np.median(auc_sorted6),3)) + ' [' + str(round(auc_sorted6[int(0.025 * len(auc_sorted6))],3)) + '-' + str(round(auc_sorted6[int(0.975 * len(auc_sorted6))],3)) + ']'
SEN_Lactate = str(round(np.median(tpr_sorted6)*100,1)) + '% [' + str(round(tpr_sorted6[int(0.025 * len(tpr_sorted6))]*100,1)) + '-' + str(round(tpr_sorted6[int(0.975 * len(tpr_sorted6))]*100,1)) + ']'
SPE_Lactate = str(round(np.median(fpr_sorted6)*100,1)) + '% [' + str(round(fpr_sorted6[int(0.025 * len(fpr_sorted6))]*100,1)) + '-' + str(round(fpr_sorted6[int(0.975 * len(fpr_sorted6))]*100,1)) + ']'

#Bilirubin
bootstrapped_auc7 = []  
bootstrapped_fpr7 = []
bootstrapped_tpr7 = []   
for i in range(n_bootstraps):
    indices7 = rng.randint(0, len(y_predLR7), len(y_predLR7))  
    if len(np.unique(Y[indices7])) < 2:
        continue
    auc_score7 = roc_auc_score(Y[indices7], y_predLR7[indices7])
    fpr_score7, tpr_score7, th7 = roc_curve(Y[indices7], y_predLR7[indices7])
    bootstrapped_auc7.append(auc_score7)
    bootstrapped_fpr7.append(1-fpr_score7[np.argmax(tpr_score7-fpr_score7)])
    bootstrapped_tpr7.append(tpr_score7[np.argmax(tpr_score7-fpr_score7)])
auc_sorted7 = np.array(bootstrapped_auc7)
auc_sorted7.sort()
fpr_sorted7 = np.array(bootstrapped_fpr7)
fpr_sorted7.sort()
tpr_sorted7 = np.array(bootstrapped_tpr7)
tpr_sorted7.sort()
AUC_Bilirubin = str(round(np.median(auc_sorted7),3)) + ' [' + str(round(auc_sorted7[int(0.025 * len(auc_sorted7))],3)) + '-' + str(round(auc_sorted7[int(0.975 * len(auc_sorted7))],3)) + ']'
SEN_Bilirubin = str(round(np.median(tpr_sorted7)*100,1)) + '% [' + str(round(tpr_sorted7[int(0.025 * len(tpr_sorted7))]*100,1)) + '-' + str(round(tpr_sorted7[int(0.975 * len(tpr_sorted7))]*100,1)) + ']'
SPE_Bilirubin = str(round(np.median(fpr_sorted7)*100,1)) + '% [' + str(round(fpr_sorted7[int(0.025 * len(fpr_sorted7))]*100,1)) + '-' + str(round(fpr_sorted7[int(0.975 * len(fpr_sorted7))]*100,1)) + ']'

#ESR
bootstrapped_auc8 = []  
bootstrapped_fpr8 = []
bootstrapped_tpr8 = []   
for i in range(n_bootstraps):
    indices8 = rng.randint(0, len(y_predLR8), len(y_predLR8))  
    if len(np.unique(Y[indices8])) < 2:
        continue
    auc_score8 = roc_auc_score(Y[indices8], y_predLR8[indices8])
    fpr_score8, tpr_score8, th8 = roc_curve(Y[indices8], y_predLR8[indices8])
    bootstrapped_auc8.append(auc_score8)
    bootstrapped_fpr8.append(1-fpr_score8[np.argmax(tpr_score8-fpr_score8)])
    bootstrapped_tpr8.append(tpr_score8[np.argmax(tpr_score8-fpr_score8)])
auc_sorted8 = np.array(bootstrapped_auc8)
auc_sorted8.sort()
fpr_sorted8 = np.array(bootstrapped_fpr8)
fpr_sorted8.sort()
tpr_sorted8 = np.array(bootstrapped_tpr8)
tpr_sorted8.sort()
AUC_ESR = str(round(np.median(auc_sorted8),3)) + ' [' + str(round(auc_sorted8[int(0.025 * len(auc_sorted8))],3)) + '-' + str(round(auc_sorted8[int(0.975 * len(auc_sorted8))],3)) + ']'
SEN_ESR = str(round(np.median(tpr_sorted8)*100,1)) + '% [' + str(round(tpr_sorted8[int(0.025 * len(tpr_sorted8))]*100,1)) + '-' + str(round(tpr_sorted8[int(0.975 * len(tpr_sorted8))]*100,1)) + ']'
SPE_ESR = str(round(np.median(fpr_sorted8)*100,1)) + '% [' + str(round(fpr_sorted8[int(0.025 * len(fpr_sorted8))]*100,1)) + '-' + str(round(fpr_sorted8[int(0.975 * len(fpr_sorted8))]*100,1)) + ']'

#Platelets
bootstrapped_auc9 = []  
bootstrapped_fpr9 = []
bootstrapped_tpr9 = []   
for i in range(n_bootstraps):
    indices9 = rng.randint(0, len(y_predLR9), len(y_predLR9))  
    if len(np.unique(Y[indices9])) < 2:
        continue
    auc_score9 = roc_auc_score(Y[indices9], y_predLR9[indices9])
    fpr_score9, tpr_score9, th9 = roc_curve(Y[indices9], y_predLR9[indices9])
    bootstrapped_auc9.append(auc_score9)
    bootstrapped_fpr9.append(1-fpr_score9[np.argmax(tpr_score9-fpr_score9)])
    bootstrapped_tpr9.append(tpr_score9[np.argmax(tpr_score9-fpr_score9)])
auc_sorted9 = np.array(bootstrapped_auc9)
auc_sorted9.sort()
fpr_sorted9 = np.array(bootstrapped_fpr9)
fpr_sorted9.sort()
tpr_sorted9 = np.array(bootstrapped_tpr9)
tpr_sorted9.sort()
AUC_PLT = str(round(np.median(auc_sorted9),3)) + ' [' + str(round(auc_sorted9[int(0.025 * len(auc_sorted9))],3)) + '-' + str(round(auc_sorted9[int(0.975 * len(auc_sorted9))],3)) + ']'
SEN_PLT = str(round(np.median(tpr_sorted9)*100,1)) + '% [' + str(round(tpr_sorted9[int(0.025 * len(tpr_sorted9))]*100,1)) + '-' + str(round(tpr_sorted9[int(0.975 * len(tpr_sorted9))]*100,1)) + ']'
SPE_PLT = str(round(np.median(fpr_sorted9)*100,1)) + '% [' + str(round(fpr_sorted9[int(0.025 * len(fpr_sorted9))]*100,1)) + '-' + str(round(fpr_sorted9[int(0.975 * len(fpr_sorted9))]*100,1)) + ']'

#CRP
bootstrapped_auc10 = []  
bootstrapped_fpr10 = []
bootstrapped_tpr10 = []   
for i in range(n_bootstraps):
    indices10 = rng.randint(0, len(y_predLR10), len(y_predLR10))  
    if len(np.unique(Y[indices10])) < 2:
        continue
    auc_score10 = roc_auc_score(Y[indices10], y_predLR10[indices10])
    fpr_score10, tpr_score10, th10 = roc_curve(Y[indices10], y_predLR10[indices10])
    bootstrapped_auc10.append(auc_score10)
    bootstrapped_fpr10.append(1-fpr_score10[np.argmax(tpr_score10-fpr_score10)])
    bootstrapped_tpr10.append(tpr_score10[np.argmax(tpr_score10-fpr_score10)])
auc_sorted10 = np.array(bootstrapped_auc10)
auc_sorted10.sort()
fpr_sorted10 = np.array(bootstrapped_fpr10)
fpr_sorted10.sort()
tpr_sorted10 = np.array(bootstrapped_tpr10)
tpr_sorted10.sort()
AUC_CRP = str(round(np.median(auc_sorted10),3)) + ' [' + str(round(auc_sorted10[int(0.025 * len(auc_sorted10))],3)) + '-' + str(round(auc_sorted10[int(0.975 * len(auc_sorted10))],3)) + ']'
SEN_CRP = str(round(np.median(tpr_sorted10)*100,1)) + '% [' + str(round(tpr_sorted10[int(0.025 * len(tpr_sorted10))]*100,1)) + '-' + str(round(tpr_sorted10[int(0.975 * len(tpr_sorted10))]*100,1)) + ']'
SPE_CRP = str(round(np.median(fpr_sorted10)*100,1)) + '% [' + str(round(fpr_sorted10[int(0.025 * len(fpr_sorted10))]*100,1)) + '-' + str(round(fpr_sorted10[int(0.975 * len(fpr_sorted10))]*100,1)) + ']'

#PCT
bootstrapped_auc11 = []  
bootstrapped_fpr11 = []
bootstrapped_tpr11 = []   
for i in range(n_bootstraps):
    indices11 = rng.randint(0, len(y_predLR11), len(y_predLR11))  
    if len(np.unique(Y[indices11])) < 2:
        continue
    auc_score11 = roc_auc_score(Y[indices11], y_predLR11[indices11])
    fpr_score11, tpr_score11, th11 = roc_curve(Y[indices11], y_predLR11[indices11])
    bootstrapped_auc11.append(auc_score11)
    bootstrapped_fpr11.append(1-fpr_score11[np.argmax(tpr_score11-fpr_score11)])
    bootstrapped_tpr11.append(tpr_score11[np.argmax(tpr_score11-fpr_score11)])
auc_sorted11 = np.array(bootstrapped_auc11)
auc_sorted11.sort()
fpr_sorted11 = np.array(bootstrapped_fpr11)
fpr_sorted11.sort()
tpr_sorted11 = np.array(bootstrapped_tpr11)
tpr_sorted11.sort()
AUC_PCT = str(round(np.median(auc_sorted11),3)) + ' [' + str(round(auc_sorted11[int(0.025 * len(auc_sorted11))],3)) + '-' + str(round(auc_sorted11[int(0.975 * len(auc_sorted11))],3)) + ']'
SEN_PCT = str(round(np.median(tpr_sorted11)*100,1)) + '% [' + str(round(tpr_sorted11[int(0.025 * len(tpr_sorted11))]*100,1)) + '-' + str(round(tpr_sorted11[int(0.975 * len(tpr_sorted11))]*100,1)) + ']'
SPE_PCT = str(round(np.median(fpr_sorted11)*100,1)) + '% [' + str(round(fpr_sorted11[int(0.025 * len(fpr_sorted11))]*100,1)) + '-' + str(round(fpr_sorted11[int(0.975 * len(fpr_sorted11))]*100,1)) + ']'

#AST
bootstrapped_auc12 = []  
bootstrapped_fpr12 = []
bootstrapped_tpr12 = []   
for i in range(n_bootstraps):
    indices12 = rng.randint(0, len(y_predLR12), len(y_predLR12))  
    if len(np.unique(Y[indices12])) < 2:
        continue
    auc_score12 = roc_auc_score(Y[indices12], y_predLR12[indices12])
    fpr_score12, tpr_score12, th12 = roc_curve(Y[indices12], y_predLR12[indices12])
    bootstrapped_auc12.append(auc_score12)
    bootstrapped_fpr12.append(1-fpr_score12[np.argmax(tpr_score12-fpr_score12)])
    bootstrapped_tpr12.append(tpr_score12[np.argmax(tpr_score12-fpr_score12)])
auc_sorted12 = np.array(bootstrapped_auc12)
auc_sorted12.sort()
fpr_sorted12 = np.array(bootstrapped_fpr12)
fpr_sorted12.sort()
tpr_sorted12 = np.array(bootstrapped_tpr12)
tpr_sorted12.sort()
AUC_AST = str(round(np.median(auc_sorted12),3)) + ' [' + str(round(auc_sorted12[int(0.025 * len(auc_sorted12))],3)) + '-' + str(round(auc_sorted12[int(0.975 * len(auc_sorted12))],3)) + ']'
SEN_AST = str(round(np.median(tpr_sorted12)*100,1)) + '% [' + str(round(tpr_sorted12[int(0.025 * len(tpr_sorted12))]*100,1)) + '-' + str(round(tpr_sorted12[int(0.975 * len(tpr_sorted12))]*100,1)) + ']'
SPE_AST = str(round(np.median(fpr_sorted12)*100,1)) + '% [' + str(round(fpr_sorted12[int(0.025 * len(fpr_sorted12))]*100,1)) + '-' + str(round(fpr_sorted12[int(0.975 * len(fpr_sorted12))]*100,1)) + ']'

#APACHE score
bootstrapped_auc13 = []  
bootstrapped_fpr13 = []
bootstrapped_tpr13 = []   
for i in range(n_bootstraps):
    indices13 = rng.randint(0, len(y_predLR13), len(y_predLR13))  
    if len(np.unique(Y[indices13])) < 2:
        continue
    auc_score13 = roc_auc_score(Y[indices13], y_predLR13[indices13])
    fpr_score13, tpr_score13, th13 = roc_curve(Y[indices13], y_predLR13[indices13])
    bootstrapped_auc13.append(auc_score13)
    bootstrapped_fpr13.append(1-fpr_score13[np.argmax(tpr_score13-fpr_score13)])
    bootstrapped_tpr13.append(tpr_score13[np.argmax(tpr_score13-fpr_score13)])
auc_sorted13 = np.array(bootstrapped_auc13)
auc_sorted13.sort()
fpr_sorted13 = np.array(bootstrapped_fpr13)
fpr_sorted13.sort()
tpr_sorted13 = np.array(bootstrapped_tpr13)
tpr_sorted13.sort()
AUC_APACHE = str(round(np.median(auc_sorted13),3)) + ' [' + str(round(auc_sorted13[int(0.025 * len(auc_sorted13))],3)) + '-' + str(round(auc_sorted13[int(0.975 * len(auc_sorted13))],3)) + ']'
SEN_APACHE = str(round(np.median(tpr_sorted13)*100,1)) + '% [' + str(round(tpr_sorted13[int(0.025 * len(tpr_sorted13))]*100,1)) + '-' + str(round(tpr_sorted13[int(0.975 * len(tpr_sorted13))]*100,1)) + ']'
SPE_APACHE = str(round(np.median(fpr_sorted13)*100,1)) + '% [' + str(round(fpr_sorted13[int(0.025 * len(fpr_sorted13))]*100,1)) + '-' + str(round(fpr_sorted13[int(0.975 * len(fpr_sorted13))]*100,1)) + ']'

#SOFA score
bootstrapped_auc14 = []  
bootstrapped_fpr14 = []
bootstrapped_tpr14 = []   
for i in range(n_bootstraps):
    indices14 = rng.randint(0, len(y_predLR14), len(y_predLR14))  
    if len(np.unique(Y[indices14])) < 2:
        continue
    auc_score14 = roc_auc_score(Y[indices14], y_predLR14[indices14])
    fpr_score14, tpr_score14, th14 = roc_curve(Y[indices14], y_predLR14[indices14])
    bootstrapped_auc14.append(auc_score14)
    bootstrapped_fpr14.append(1-fpr_score14[np.argmax(tpr_score14-fpr_score14)])
    bootstrapped_tpr14.append(tpr_score14[np.argmax(tpr_score14-fpr_score14)])
auc_sorted14 = np.array(bootstrapped_auc14)
auc_sorted14.sort()
fpr_sorted14 = np.array(bootstrapped_fpr14)
fpr_sorted14.sort()
tpr_sorted14 = np.array(bootstrapped_tpr14)
tpr_sorted14.sort()
AUC_SOFA = str(round(np.median(auc_sorted14),3)) + ' [' + str(round(auc_sorted14[int(0.025 * len(auc_sorted14))],3)) + '-' + str(round(auc_sorted14[int(0.975 * len(auc_sorted14))],3)) + ']'
SEN_SOFA = str(round(np.median(tpr_sorted14)*100,1)) + '% [' + str(round(tpr_sorted14[int(0.025 * len(tpr_sorted14))]*100,1)) + '-' + str(round(tpr_sorted14[int(0.975 * len(tpr_sorted14))]*100,1)) + ']'
SPE_SOFA = str(round(np.median(fpr_sorted14)*100,1)) + '% [' + str(round(fpr_sorted14[int(0.025 * len(fpr_sorted14))]*100,1)) + '-' + str(round(fpr_sorted14[int(0.975 * len(fpr_sorted14))]*100,1)) + ']'

# NPV, PPV, DOR 
#0: BUN, 1: MAP, 2:Na, 3: GCS, 4: NLR, 5: Creatinine, 6: Lactate, 7: Bilirubin, 8:ESR, 9: PLT, 10: CRP, 11: PCT, 12: AST, 13: APACHE, 14: SOFA
pred0 = y_predLR0.map(lambda x: 0 if x < optimal_threshold0 else 1)
pred1 = y_predLR1.map(lambda x: 0 if x < optimal_threshold1 else 1)
pred2 = y_predLR2.map(lambda x: 0 if x < optimal_threshold2 else 1)
pred3 = y_predLR3.map(lambda x: 0 if x < optimal_threshold3 else 1)
pred4 = y_predLR4.map(lambda x: 0 if x < optimal_threshold4 else 1)
pred5 = y_predLR5.map(lambda x: 0 if x < optimal_threshold5 else 1)
pred6 = y_predLR6.map(lambda x: 0 if x < optimal_threshold6 else 1)
pred7 = y_predLR7.map(lambda x: 0 if x < optimal_threshold7 else 1)
pred8 = y_predLR8.map(lambda x: 0 if x < optimal_threshold8 else 1)
pred9 = y_predLR9.map(lambda x: 0 if x < optimal_threshold9 else 1)
pred10 = y_predLR10.map(lambda x: 0 if x < optimal_threshold10 else 1)
pred11 = y_predLR11.map(lambda x: 0 if x < optimal_threshold11 else 1)
pred12 = y_predLR12.map(lambda x: 0 if x < optimal_threshold12 else 1)
pred13 = y_predLR13.map(lambda x: 0 if x < optimal_threshold13 else 1)
pred14 = y_predLR14.map(lambda x: 0 if x < optimal_threshold14 else 1)

cmLR0 = confusion_matrix(Y, pred1)
cmLR1 = confusion_matrix(Y, pred2)
cmLR2 = confusion_matrix(Y, pred3)
cmLR3 = confusion_matrix(Y, pred4)
cmLR4 = confusion_matrix(Y, pred5)
cmLR5 = confusion_matrix(Y, pred6)
cmLR6 = confusion_matrix(Y, pred7)
cmLR7 = confusion_matrix(Y, pred8)
cmLR8 = confusion_matrix(Y, pred9)
cmLR9 = confusion_matrix(Y, pred10)
cmLR10 = confusion_matrix(Y, pred11)
cmLR11 = confusion_matrix(Y, pred12)
cmLR12 = confusion_matrix(Y, pred13)
cmLR13 = confusion_matrix(Y, pred14)
cmLR14 = confusion_matrix(Y, pred0)

# Odds ratio /confidence interval after applying the threshold of Optimal Cutoff - statsmodel contingency tables
ctLR0 = contingency_tables.Table2x2(cmLR0)
ctLR1 = contingency_tables.Table2x2(cmLR1)
ctLR2 = contingency_tables.Table2x2(cmLR2)
ctLR3 = contingency_tables.Table2x2(cmLR3)
ctLR4 = contingency_tables.Table2x2(cmLR4)
ctLR5 = contingency_tables.Table2x2(cmLR5)
ctLR6 = contingency_tables.Table2x2(cmLR6)
ctLR7 = contingency_tables.Table2x2(cmLR7)
ctLR8 = contingency_tables.Table2x2(cmLR8)
ctLR9 = contingency_tables.Table2x2(cmLR9)
ctLR10 = contingency_tables.Table2x2(cmLR10)
ctLR11 = contingency_tables.Table2x2(cmLR11)
ctLR12 = contingency_tables.Table2x2(cmLR12)
ctLR13 = contingency_tables.Table2x2(cmLR13)
ctLR14 = contingency_tables.Table2x2(cmLR14)

#Calculating the Likelihood Ratio
LRPositive0 = tpr0[np.argmax(tpr0-fpr0)]/(fpr0[np.argmax(tpr0-fpr0)])
LRPositive1 = tpr1[np.argmax(tpr1-fpr1)]/(fpr1[np.argmax(tpr1-fpr1)])
LRPositive2 = tpr2[np.argmax(tpr2-fpr2)]/(fpr2[np.argmax(tpr2-fpr2)])
LRPositive3 = tpr3[np.argmax(tpr3-fpr3)]/(fpr3[np.argmax(tpr3-fpr3)])
LRPositive4 = tpr4[np.argmax(tpr4-fpr4)]/(fpr4[np.argmax(tpr4-fpr4)])
LRPositive5 = tpr5[np.argmax(tpr5-fpr5)]/(fpr5[np.argmax(tpr5-fpr5)])
LRPositive6 = tpr6[np.argmax(tpr6-fpr6)]/(fpr6[np.argmax(tpr6-fpr6)])
LRPositive7 = tpr7[np.argmax(tpr7-fpr7)]/(fpr7[np.argmax(tpr7-fpr7)])
LRPositive8 = tpr8[np.argmax(tpr8-fpr8)]/(fpr8[np.argmax(tpr8-fpr8)])
LRPositive9 = tpr9[np.argmax(tpr9-fpr9)]/(fpr9[np.argmax(tpr9-fpr9)])
LRPositive10 = tpr10[np.argmax(tpr10-fpr10)]/(fpr10[np.argmax(tpr10-fpr10)])
LRPositive11 = tpr11[np.argmax(tpr11-fpr11)]/(fpr11[np.argmax(tpr11-fpr11)])
LRPositive12 = tpr12[np.argmax(tpr12-fpr12)]/(fpr12[np.argmax(tpr12-fpr12)])
LRPositive13 = tpr13[np.argmax(tpr13-fpr13)]/(fpr13[np.argmax(tpr13-fpr13)])
LRPositive14 = tpr14[np.argmax(tpr14-fpr14)]/(fpr14[np.argmax(tpr14-fpr14)])
LRP_BUN = str(round(LRPositive0,2))
LRP_MAP = str(round(LRPositive1,2))
LRP_Na = str(round(LRPositive2,2))
LRP_GCS = str(round(LRPositive3,2))
LRP_NLR = str(round(LRPositive4,2))
LRP_Creatinine = str(round(LRPositive5,2))
LRP_Lactate = str(round(LRPositive6,2))
LRP_Bilirubin = str(round(LRPositive7,2))
LRP_ESR = str(round(LRPositive8,2))
LRP_PLT = str(round(LRPositive9,2))
LRP_CRP = str(round(LRPositive10,2))
LRP_PCT = str(round(LRPositive11,2))
LRP_AST = str(round(LRPositive12,2))
LRP_APACHE = str(round(LRPositive13,2))
LRP_SOFA = str(round(LRPositive14,2))

LRNegative0 = (1-tpr0[np.argmax(tpr0-fpr0)])/(1-fpr0[np.argmax(tpr0-fpr0)])
LRNegative1 = (1-tpr1[np.argmax(tpr1-fpr1)])/(1-fpr1[np.argmax(tpr1-fpr1)])
LRNegative2 = (1-tpr2[np.argmax(tpr2-fpr2)])/(1-fpr2[np.argmax(tpr2-fpr2)])
LRNegative3 = (1-tpr3[np.argmax(tpr3-fpr3)])/(1-fpr3[np.argmax(tpr3-fpr3)])
LRNegative4 = (1-tpr4[np.argmax(tpr4-fpr4)])/(1-fpr4[np.argmax(tpr4-fpr4)])
LRNegative5 = (1-tpr5[np.argmax(tpr5-fpr5)])/(1-fpr5[np.argmax(tpr5-fpr5)])
LRNegative6 = (1-tpr6[np.argmax(tpr6-fpr6)])/(1-fpr6[np.argmax(tpr6-fpr6)])
LRNegative7 = (1-tpr7[np.argmax(tpr7-fpr7)])/(1-fpr7[np.argmax(tpr7-fpr7)])
LRNegative8 = (1-tpr8[np.argmax(tpr8-fpr8)])/(1-fpr8[np.argmax(tpr8-fpr8)])
LRNegative9 = (1-tpr9[np.argmax(tpr9-fpr9)])/(1-fpr9[np.argmax(tpr9-fpr9)])
LRNegative10 = (1-tpr10[np.argmax(tpr10-fpr10)])/(1-fpr10[np.argmax(tpr10-fpr10)])
LRNegative11 = (1-tpr11[np.argmax(tpr11-fpr11)])/(1-fpr11[np.argmax(tpr11-fpr11)])
LRNegative12 = (1-tpr12[np.argmax(tpr12-fpr12)])/(1-fpr12[np.argmax(tpr12-fpr12)])
LRNegative13 = (1-tpr13[np.argmax(tpr13-fpr13)])/(1-fpr13[np.argmax(tpr13-fpr13)])
LRNegative14 = (1-tpr14[np.argmax(tpr14-fpr14)])/(1-fpr14[np.argmax(tpr14-fpr14)])
LRN_BUN = str(round(LRNegative0,2))
LRN_MAP = str(round(LRNegative1,2))
LRN_Na = str(round(LRNegative2,2))
LRN_GCS = str(round(LRNegative3,2))
LRN_NLR = str(round(LRNegative4,2))
LRN_Creatinine = str(round(LRNegative5,2))
LRN_Lactate = str(round(LRNegative6,2))
LRN_Bilirubin = str(round(LRNegative7,2))
LRN_ESR = str(round(LRNegative8,2))
LRN_PLT = str(round(LRNegative9,2))
LRN_CRP = str(round(LRNegative10,2))
LRN_PCT = str(round(LRNegative11,2))
LRN_AST = str(round(LRNegative12,2))
LRN_APACHE = str(round(LRNegative13,2))
LRN_SOFA = str(round(LRNegative14,2))

#Calculating Diagnostic Odd Ratio
DOR0 = LRPositive0/LRNegative0
DOR1 = LRPositive1/LRNegative1
DOR2 = LRPositive2/LRNegative2
DOR3 = LRPositive3/LRNegative3
DOR4 = LRPositive4/LRNegative4
DOR5 = LRPositive5/LRNegative5
DOR6 = LRPositive6/LRNegative6
DOR7 = LRPositive7/LRNegative7
DOR8 = LRPositive8/LRNegative8
DOR9 = LRPositive9/LRNegative9
DOR10 = LRPositive10/LRNegative10
DOR11 = LRPositive11/LRNegative11
DOR12 = LRPositive12/LRNegative12
DOR13 = LRPositive13/LRNegative13
DOR14 = LRPositive14/LRNegative14
DOR_BUN = str(round(DOR0,2))
DOR_MAP = str(round(DOR1,2))
DOR_Na = str(round(DOR2,2))
DOR_GCS = str(round(DOR3,2))
DOR_NLR = str(round(DOR4,2))
DOR_Creatinine = str(round(DOR5,2))
DOR_Lactate = str(round(DOR6,2))
DOR_Bilirubin = str(round(DOR7,2))
DOR_ESR = str(round(DOR8,2))
DOR_PLT = str(round(DOR9,2))
DOR_CRP = str(round(DOR10,2))
DOR_PCT = str(round(DOR11,2))
DOR_AST = str(round(DOR12,2))
DOR_APACHE = str(round(DOR13,2))
DOR_SOFA = str(round(DOR14,2))

#Calculating PPV and NPV PPV&NPV: CRP PPV1&NPV1: PCT PPV2&NPV2: CRP&PCT
prevalence = len(PositivePatients)/len(BacteremiaData)
PPV0= (tpr0[np.argmax(tpr0-fpr0)]* prevalence) / (tpr0[np.argmax(tpr0-fpr0)]* prevalence + (fpr0[np.argmax(tpr0-fpr0)])*(1-prevalence))
PPV1= (tpr1[np.argmax(tpr1-fpr1)]* prevalence) / (tpr1[np.argmax(tpr1-fpr1)]* prevalence + (fpr1[np.argmax(tpr1-fpr1)])*(1-prevalence))
PPV2= (tpr2[np.argmax(tpr2-fpr2)]* prevalence) / (tpr2[np.argmax(tpr2-fpr2)]* prevalence + (fpr2[np.argmax(tpr2-fpr2)])*(1-prevalence))
PPV3= (tpr3[np.argmax(tpr3-fpr3)]* prevalence) / (tpr3[np.argmax(tpr3-fpr3)]* prevalence + (fpr3[np.argmax(tpr3-fpr3)])*(1-prevalence))
PPV4= (tpr4[np.argmax(tpr4-fpr4)]* prevalence) / (tpr4[np.argmax(tpr4-fpr4)]* prevalence + (fpr4[np.argmax(tpr4-fpr4)])*(1-prevalence))
PPV5= (tpr5[np.argmax(tpr5-fpr5)]* prevalence) / (tpr5[np.argmax(tpr5-fpr5)]* prevalence + (fpr5[np.argmax(tpr5-fpr5)])*(1-prevalence))
PPV6= (tpr6[np.argmax(tpr6-fpr6)]* prevalence) / (tpr6[np.argmax(tpr6-fpr6)]* prevalence + (fpr6[np.argmax(tpr6-fpr6)])*(1-prevalence))
PPV7= (tpr7[np.argmax(tpr7-fpr7)]* prevalence) / (tpr7[np.argmax(tpr7-fpr7)]* prevalence + (fpr7[np.argmax(tpr7-fpr7)])*(1-prevalence))
PPV8= (tpr8[np.argmax(tpr8-fpr8)]* prevalence) / (tpr8[np.argmax(tpr8-fpr8)]* prevalence + (fpr8[np.argmax(tpr8-fpr8)])*(1-prevalence))
PPV9= (tpr9[np.argmax(tpr9-fpr9)]* prevalence) / (tpr9[np.argmax(tpr9-fpr9)]* prevalence + (fpr9[np.argmax(tpr9-fpr9)])*(1-prevalence))
PPV10= (tpr10[np.argmax(tpr10-fpr10)]* prevalence) / (tpr10[np.argmax(tpr10-fpr10)]* prevalence + (fpr10[np.argmax(tpr10-fpr10)])*(1-prevalence))
PPV11= (tpr11[np.argmax(tpr11-fpr11)]* prevalence) / (tpr11[np.argmax(tpr11-fpr11)]* prevalence + (fpr11[np.argmax(tpr11-fpr11)])*(1-prevalence))
PPV12= (tpr12[np.argmax(tpr12-fpr12)]* prevalence) / (tpr12[np.argmax(tpr12-fpr12)]* prevalence + (fpr12[np.argmax(tpr12-fpr12)])*(1-prevalence))
PPV13= (tpr13[np.argmax(tpr13-fpr13)]* prevalence) / (tpr13[np.argmax(tpr13-fpr13)]* prevalence + (fpr13[np.argmax(tpr13-fpr13)])*(1-prevalence))
PPV14= (tpr14[np.argmax(tpr14-fpr14)]* prevalence) / (tpr14[np.argmax(tpr14-fpr14)]* prevalence + (fpr14[np.argmax(tpr14-fpr14)])*(1-prevalence))
PPV_BUN = str(round(PPV0,3)*100)+'%'
PPV_MAP = str(round(PPV1,3)*100)+'%'
PPV_Na = str(round(PPV2,3)*100)+'%'
PPV_GCS = str(round(PPV3,3)*100)+'%'
PPV_NLR = str(round(PPV4,3)*100)+'%'
PPV_Creatinine = str(round(PPV5,3)*100)+'%'
PPV_Lactate = str(round(PPV6,3)*100)+'%'
PPV_Bilirubin = str(round(PPV7,3)*100)+'%'
PPV_ESR = str(round(PPV8,3)*100)+'%'
PPV_PLT = str(round(PPV9,3)*100)+'%'
PPV_CRP = str(round(PPV10,3)*100)+'%'
PPV_PCT = str(round(PPV11,3)*100)+'%'
PPV_AST = str(round(PPV12,3)*100)+'%'
PPV_APACHE = str(round(PPV13,3)*100)+'%'
PPV_SOFA = str(round(PPV14,3)*100)+'%'

NPV0= ((1-fpr0[np.argmax(tpr0-fpr0)]) * (1-prevalence)) / ((1-tpr0[np.argmax(tpr0-fpr0)])* prevalence + ((1-fpr0[np.argmax(tpr0-fpr0)]))*(1-prevalence))
NPV1= ((1-fpr1[np.argmax(tpr1-fpr1)]) * (1-prevalence)) / ((1-tpr1[np.argmax(tpr1-fpr1)])* prevalence + ((1-fpr1[np.argmax(tpr1-fpr1)]))*(1-prevalence))
NPV2= ((1-fpr2[np.argmax(tpr2-fpr2)]) * (1-prevalence)) / ((1-tpr2[np.argmax(tpr2-fpr2)])* prevalence + ((1-fpr2[np.argmax(tpr2-fpr2)]))*(1-prevalence))
NPV3= ((1-fpr3[np.argmax(tpr3-fpr3)]) * (1-prevalence)) / ((1-tpr3[np.argmax(tpr3-fpr3)])* prevalence + ((1-fpr3[np.argmax(tpr3-fpr3)]))*(1-prevalence))
NPV4= ((1-fpr4[np.argmax(tpr4-fpr4)]) * (1-prevalence)) / ((1-tpr4[np.argmax(tpr4-fpr4)])* prevalence + ((1-fpr4[np.argmax(tpr4-fpr4)]))*(1-prevalence))
NPV5= ((1-fpr5[np.argmax(tpr5-fpr5)]) * (1-prevalence)) / ((1-tpr5[np.argmax(tpr5-fpr5)])* prevalence + ((1-fpr5[np.argmax(tpr5-fpr5)]))*(1-prevalence))
NPV6= ((1-fpr6[np.argmax(tpr6-fpr6)]) * (1-prevalence)) / ((1-tpr6[np.argmax(tpr6-fpr6)])* prevalence + ((1-fpr6[np.argmax(tpr6-fpr6)]))*(1-prevalence))
NPV7= ((1-fpr7[np.argmax(tpr7-fpr7)]) * (1-prevalence)) / ((1-tpr7[np.argmax(tpr7-fpr7)])* prevalence + ((1-fpr7[np.argmax(tpr7-fpr7)]))*(1-prevalence))
NPV8= ((1-fpr8[np.argmax(tpr8-fpr8)]) * (1-prevalence)) / ((1-tpr8[np.argmax(tpr8-fpr8)])* prevalence + ((1-fpr8[np.argmax(tpr8-fpr8)]))*(1-prevalence))
NPV9= ((1-fpr9[np.argmax(tpr9-fpr9)]) * (1-prevalence)) / ((1-tpr9[np.argmax(tpr9-fpr9)])* prevalence + ((1-fpr9[np.argmax(tpr9-fpr9)]))*(1-prevalence))
NPV10= ((1-fpr10[np.argmax(tpr10-fpr10)]) * (1-prevalence)) / ((1-tpr10[np.argmax(tpr10-fpr10)])* prevalence + ((1-fpr10[np.argmax(tpr10-fpr10)]))*(1-prevalence))
NPV11= ((1-fpr11[np.argmax(tpr11-fpr11)]) * (1-prevalence)) / ((1-tpr11[np.argmax(tpr11-fpr11)])* prevalence + ((1-fpr11[np.argmax(tpr11-fpr11)]))*(1-prevalence))
NPV12= ((1-fpr12[np.argmax(tpr12-fpr12)]) * (1-prevalence)) / ((1-tpr12[np.argmax(tpr12-fpr12)])* prevalence + ((1-fpr12[np.argmax(tpr12-fpr12)]))*(1-prevalence))
NPV13= ((1-fpr13[np.argmax(tpr13-fpr13)]) * (1-prevalence)) / ((1-tpr13[np.argmax(tpr13-fpr13)])* prevalence + ((1-fpr13[np.argmax(tpr13-fpr13)]))*(1-prevalence))
NPV14= ((1-fpr14[np.argmax(tpr14-fpr14)]) * (1-prevalence)) / ((1-tpr14[np.argmax(tpr14-fpr14)])* prevalence + ((1-fpr14[np.argmax(tpr14-fpr14)]))*(1-prevalence))
NPV_BUN = str(round(NPV0,3)*100)+'%'
NPV_MAP = str(round(NPV1,3)*100)+'%'
NPV_Na = str(round(NPV2,3)*100)+'%'
NPV_GCS = str(round(NPV3,3)*100)+'%'
NPV_NLR = str(round(NPV4,3)*100)+'%'
NPV_Creatinine = str(round(NPV5,3)*100)+'%'
NPV_Lactate = str(round(NPV6,3)*100)+'%'
NPV_Bilirubin = str(round(NPV7,3)*100)+'%'
NPV_ESR = str(round(NPV8,3)*100)+'%'
NPV_PLT = str(round(NPV9,3)*100)+'%'
NPV_CRP = str(round(NPV10,3)*100)+'%'
NPV_PCT = str(round(NPV11,3)*100)+'%'
NPV_AST = str(round(NPV12,3)*100)+'%'
NPV_APACHE = str(round(NPV13,3)*100)+'%'
NPV_SOFA = str(round(NPV14,3)*100)+'%'

# Unscaling the Robust Scaled data for the interpretation of the optimal cut-off value using optimal_threshold
#0: BUN, 1: MAP, 2:Na, 3: GCS, 4: NLR, 5: Creatinine, 6: Lactate, 7: Bilirubin, 8:ESR, 9: PLT, 10: CRP, 11: PCT, 12: AST, 13: APACHE, 14: SOFA
BUN_CutOffFinder = pd.concat([y_predLR0.copy(),Tester['BUN']], axis=1)
MAP_CutOffFinder = pd.concat([y_predLR1.copy(),Tester['MAP']], axis=1)
Na_CutOffFinder = pd.concat([y_predLR2.copy(),Tester['Na']], axis=1)
GCS_CutOffFinder = pd.concat([y_predLR3.copy(),Tester['GCS Score']], axis=1)
NLR_CutOffFinder = pd.concat([y_predLR4.copy(),Tester['Neutrophil Lymphocyte Ratio']], axis=1)
Creatinine_CutOffFinder = pd.concat([y_predLR5.copy(),Tester['Creatinine']], axis=1)
Lactate_CutOffFinder = pd.concat([y_predLR6.copy(),Tester['Lactate Level']], axis=1)
Bilirubin_CutOffFinder = pd.concat([y_predLR7.copy(),Tester['Bilirubin']], axis=1)
ESR_CutOffFinder = pd.concat([y_predLR8.copy(),Tester['ESR']], axis=1)
Platelets_CutOffFinder = pd.concat([y_predLR9.copy(),Tester['Platelets']], axis=1)
CRP_CutOffFinder = pd.concat([y_predLR10.copy(),Tester['CRP(mg/L)']], axis=1)
PCT_CutOffFinder = pd.concat([y_predLR11.copy(),Tester['PCT(ng/ml)']], axis=1)
AST_CutOffFinder = pd.concat([y_predLR12.copy(),Tester['AST']], axis=1)
APACHE_CutOffFinder = pd.concat([y_predLR13.copy(),Tester['APACHE II score']], axis=1)
SOFA_CutOffFinder = pd.concat([y_predLR14.copy(),Tester['SOFA score']], axis=1)

#Table 2 New Version
LogData = {'Cut-off': pd.Series(['19.2', '67.3', '133.4', '10', '14.26', '1.86', '6.1', '0.97', '41', '168', '164.3', '3.18', '62', '24', '13'], 
                      index =['BUN', 'MAP', 'Na', 'GCS Score','Neutrophil Lymphocyte Ratio','Creatinine', 'Lactate Level', 'Bilirubin', 
                              'ESR', 'Platelets', 'CRP', 'PCT', 'AST', 'APACHE II score','SOFA score']),
        'AUC' : pd.Series([AUC_BUN, AUC_MAP, AUC_Na, AUC_GCS, AUC_NLR, AUC_Creatinine, AUC_Lactate, AUC_Bilirubin, AUC_ESR, AUC_PLT, AUC_CRP, AUC_PCT, AUC_AST, AUC_APACHE, AUC_SOFA], 
                index =['BUN', 'MAP', 'Na', 'GCS Score','Neutrophil Lymphocyte Ratio','Creatinine', 'Lactate Level', 'Bilirubin', 
                         'ESR', 'Platelets', 'CRP', 'PCT', 'AST', 'APACHE II score','SOFA score']),
        'Sensitivity' : pd.Series([SEN_BUN, SEN_MAP, SEN_Na, SEN_GCS, SEN_NLR, SEN_Creatinine, SEN_Lactate, SEN_Bilirubin, SEN_ESR, SEN_PLT, SEN_CRP, SEN_PCT, SEN_AST, SEN_APACHE, SEN_SOFA], 
                        index =['BUN', 'MAP', 'Na', 'GCS Score', 'Neutrophil Lymphocyte Ratio', 'Creatinine', 'Lactate Level', 
                                'Bilirubin', 'ESR', 'Platelets', 'CRP', 'PCT', 'AST', 'APACHE II score','SOFA score']),
        'Specificity' : pd.Series([SPE_BUN, SPE_MAP, SPE_Na, SPE_GCS, SPE_NLR, SPE_Creatinine, SPE_Lactate, SPE_Bilirubin, SPE_ESR, SPE_PLT, SPE_CRP, SPE_PCT, SPE_AST, SPE_APACHE, SPE_SOFA], 
                        index =['BUN', 'MAP', 'Na', 'GCS Score', 'Neutrophil Lymphocyte Ratio', 'Creatinine', 'Lactate Level', 
                                'Bilirubin', 'ESR', 'Platelets', 'CRP', 'PCT', 'AST', 'APACHE II score','SOFA score']),
        'PPV': pd.Series([PPV_BUN, PPV_MAP, PPV_Na, PPV_GCS, PPV_NLR, PPV_Creatinine, PPV_Lactate, PPV_Bilirubin, PPV_ESR, PPV_PLT, PPV_CRP, PPV_PCT, PPV_AST, PPV_APACHE, PPV_SOFA], 
               index =['BUN', 'MAP', 'Na', 'GCS Score','Neutrophil Lymphocyte Ratio','Creatinine', 'Lactate Level', 'Bilirubin', 
                       'ESR', 'Platelets', 'CRP', 'PCT', 'AST', 'APACHE II score','SOFA score']),
        'NPV': pd.Series([NPV_BUN, NPV_MAP, NPV_Na, NPV_GCS, NPV_NLR, NPV_Creatinine, NPV_Lactate, NPV_Bilirubin, NPV_ESR, NPV_PLT, NPV_CRP, NPV_PCT, NPV_AST, NPV_APACHE, NPV_SOFA], 
               index =['BUN', 'MAP', 'Na', 'GCS Score','Neutrophil Lymphocyte Ratio','Creatinine', 'Lactate Level', 'Bilirubin', 
                       'ESR', 'Platelets', 'CRP', 'PCT', 'AST', 'APACHE II score','SOFA score']),
        'LR +': pd.Series([LRP_BUN, LRP_MAP, LRP_Na, LRP_GCS, LRP_NLR, LRP_Creatinine, LRP_Lactate, LRP_Bilirubin, LRP_ESR, LRP_PLT, LRP_CRP, LRP_PCT, LRP_AST, LRP_APACHE, LRP_SOFA], 
               index =['BUN', 'MAP', 'Na', 'GCS Score','Neutrophil Lymphocyte Ratio','Creatinine', 'Lactate Level', 'Bilirubin', 
                       'ESR', 'Platelets', 'CRP', 'PCT', 'AST', 'APACHE II score','SOFA score']),
        'LR -': pd.Series([LRN_BUN, LRN_MAP, LRN_Na, LRN_GCS, LRN_NLR, LRN_Creatinine, LRN_Lactate, LRN_Bilirubin, LRN_ESR, LRN_PLT, LRN_CRP, LRN_PCT, LRN_AST, LRN_APACHE, LRN_SOFA], 
               index =['BUN', 'MAP', 'Na', 'GCS Score','Neutrophil Lymphocyte Ratio','Creatinine', 'Lactate Level', 'Bilirubin', 
                       'ESR', 'Platelets', 'CRP', 'PCT', 'AST', 'APACHE II score','SOFA score']),
        'DOR': pd.Series([DOR_BUN, DOR_MAP, DOR_Na, DOR_GCS, DOR_NLR, DOR_Creatinine, DOR_Lactate, DOR_Bilirubin, DOR_ESR, DOR_PLT, DOR_CRP, DOR_PCT, DOR_AST, DOR_APACHE, DOR_SOFA], 
               index =['BUN', 'MAP', 'Na', 'GCS Score','Neutrophil Lymphocyte Ratio','Creatinine', 'Lactate Level', 'Bilirubin', 
                       'ESR', 'Platelets', 'CRP', 'PCT', 'AST', 'APACHE II score','SOFA score']),
        'P-value': pd.Series([P_BUN, P_MAP, P_Na, P_GCS, P_NLR, P_Creatinine, P_Lactate, P_Bilirubin, P_ESR, P_PLT, P_CRP, P_PCT, P_AST, P_APACHE, P_SOFA], 
                   index =['BUN', 'MAP', 'Na', 'GCS Score','Neutrophil Lymphocyte Ratio','Creatinine', 'Lactate Level', 'Bilirubin', 
                           'ESR', 'Platelets', 'CRP', 'PCT', 'AST', 'APACHE II score','SOFA score'])}
LogResultTableNew = pd.DataFrame(LogData)
LogResultTableNew.to_csv('LogResultTableNew.csv')

#________________Multivariable Logistic Regression Testing_______________________________________________________________________
#Stepwise Regression performed
# PCT = 0.846
# PCT + CRP = 0.812 
# PCT + SOFA = 0.835
# PCT + CRP + SOFA = 0.846
# PCT + CRP + SOFA + Bilirubin = 0.862
# PCT + CRP + SOFA + Bilirubin + BUN = 0.864
# PCT + CRP + SOFA + Bilirubin + PLT = 0.865
# PCT + CRP + SOFA + Bilirubin + NLR = 0.886
# PCT + CRP + SOFA + Bilirubin + BUN + NLR = 0.886
# PCT + CRP + SOFA + Bilirubin + BUN + PLT = 0.867

#Backward Elimination performed
#SOFA : 0.981
#CRP : 0.744
#Creatinine : 0.731
#Na : 0.472
#APACHE : 0.322
#BUN : 0.389
#MAP: 0.187
#AST: 0.094
LRmodel15 = smf.logit('Y ~ PCT + Bilirubin + NLR + PLT + Lactate + GCS + ESR', data=Tester).fit()
X15 = Tester[['PCT(ng/ml)', 'Bilirubin', 'Neutrophil Lymphocyte Ratio', 'Platelets', 'Lactate Level', 'GCS Score', 'ESR']]
X15_C = sm.add_constant(X15)
y_predLR15 = LRmodel15.predict(X15_C)
fpr15, tpr15, thresholds15 = roc_curve(y_true=Y, y_score=y_predLR15)
auc15 = roc_auc_score(Y, y_predLR15)
optimal_threshold15 = thresholds15[np.argmax(tpr15-fpr15)]

params15 = LRmodel15.params
conf15 = round(np.exp(LRmodel15.conf_int()),3)
conf15['Odds Ratio'] = round(np.exp(params15),3)
conf15.columns = ['2.5%', '97.5%', 'Odds Ratio']

#Multiple Logistic Regression
bootstrapped_auc15 = []  
bootstrapped_fpr15 = []
bootstrapped_tpr15 = []   
for i in range(n_bootstraps):
    indices15 = rng.randint(0, len(y_predLR15), len(y_predLR15))  
    if len(np.unique(Y[indices15])) < 2:
        continue
    auc_score15 = roc_auc_score(Y[indices15], y_predLR15[indices15])
    fpr_score15, tpr_score15, th15 = roc_curve(Y[indices15], y_predLR15[indices15])
    bootstrapped_auc15.append(auc_score15)
    bootstrapped_fpr15.append(1-fpr_score15[np.argmax(tpr_score15-fpr_score15)])
    bootstrapped_tpr15.append(tpr_score15[np.argmax(tpr_score15-fpr_score15)])
auc_sorted15 = np.array(bootstrapped_auc15)
auc_sorted15.sort()
fpr_sorted15 = np.array(bootstrapped_fpr15)
fpr_sorted15.sort()
tpr_sorted15 = np.array(bootstrapped_tpr15)
tpr_sorted15.sort()
AUC_ML = str(round(np.median(auc_sorted15),3)) + ' [' + str(round(auc_sorted15[int(0.025 * len(auc_sorted15))],3)) + '-' + str(round(auc_sorted15[int(0.975 * len(auc_sorted15))],3)) + ']'
SEN_ML = str(round(np.median(tpr_sorted15)*100,1)) + ' [' + str(round(tpr_sorted15[int(0.025 * len(tpr_sorted15))]*100,1)) + '-' + str(round(tpr_sorted15[int(0.975 * len(tpr_sorted15))]*100,1)) + ']'
SPE_ML = str(round(np.median(fpr_sorted15)*100,1)) + ' [' + str(round(fpr_sorted15[int(0.025 * len(fpr_sorted15))]*100,1)) + '-' + str(round(fpr_sorted15[int(0.975 * len(fpr_sorted15))]*100,1)) + ']'

dataML = {'Sensitivity' : pd.Series([SEN_ML], index =['Multiple Logistic']),
        'Specificity' : pd.Series([SPE_ML], index =['Multiple Logistic']),
        'AUC' : pd.Series([AUC_ML], index =['Multiple Logistic'])}
MultiLogResultTable = pd.DataFrame(dataML)

LRPositive15 = tpr15[np.argmax(tpr15-fpr15)]/(fpr15[np.argmax(tpr15-fpr15)])
LRP_MLR = str(round(LRPositive15,2))
LRNegative15 = (1-tpr15[np.argmax(tpr15-fpr15)])/(1-fpr15[np.argmax(tpr15-fpr15)])
LRN_MLR = str(round(LRNegative15,2))
DOR15 = LRPositive15/LRNegative15
DOR_MLR = str(round(DOR15,2))
PPV15 = (tpr15[np.argmax(tpr15-fpr15)]* prevalence) / (tpr15[np.argmax(tpr15-fpr15)]* prevalence + (fpr15[np.argmax(tpr15-fpr15)])*(1-prevalence))
PPV_MLR = str(round(PPV15,3)*100)+'%'
NPV15 = ((1-fpr15[np.argmax(tpr15-fpr15)]) * (1-prevalence)) / ((1-tpr15[np.argmax(tpr15-fpr15)])* prevalence + ((1-fpr15[np.argmax(tpr15-fpr15)]))*(1-prevalence))
NPV_MLR = str(round(NPV15,3)*100)+'%'

#________________ROC curves + Survival Analysis_______________________________________________________________________

#ROC curve with all included
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 14})
plt.grid()
plt.plot(fpr0, tpr0, color='darkorange', lw=2,
         label='BUN={0:.3f}'.format(auc0))
plt.plot(fpr1, tpr1, color='darkgreen', lw=2,
         label='MAP={0:.3f}'.format(auc1))
plt.plot(fpr2, tpr2, color='lightpink', lw=2,
         label='Na={0:.3f}'.format(auc2))
plt.plot(fpr3, tpr3, color='gray', lw=2,
         label='GCS={0:.3f}'.format(auc3))
plt.plot(fpr4, tpr4, color='darkred', lw=2,
         label='NLR={0:.3f}'.format(auc4))
plt.plot(fpr5, tpr5, color='tan', lw=2,
         label='Creatinine={0:.3f}'.format(auc5))
plt.plot(fpr6, tpr6, color='gold', lw=2,
         label='Lactate={0:.3f}'.format(auc6))
plt.plot(fpr7, tpr7, color='yellow', lw=2,
         label='Bilirubin={0:.3f}'.format(auc7))
plt.plot(fpr8, tpr8, color='lime', lw=2,
         label='ESR={0:.3f}'.format(auc8))
plt.plot(fpr9, tpr9, color='aqua', lw=2,
         label='PLT={0:.3f}'.format(auc9))
plt.plot(fpr10, tpr10, color='royalblue', lw=2,
         label='CRP={0:.3f}'.format(auc10))
plt.plot(fpr11, tpr11, color='darkblue', lw=2,
         label='PCT={0:.3f}'.format(auc11))
plt.plot(fpr12, tpr12, color='deeppink', lw=2,
         label='AST={0:.3f}'.format(auc12))
plt.plot(fpr13, tpr13, color='blueviolet', lw=2,
         label='APACHE II={0:.3f}'.format(auc13))
plt.plot(fpr14, tpr14, color='purple', lw=2,
         label='SOFA={0:.3f}'.format(auc14))
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend(loc="lower right", fontsize=10)
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()

#ROC curve for multivariable logistic regression
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 14})
plt.grid()
plt.plot(fpr15, tpr15, color='darkorange', lw=2,
          label='AUC={0:.3f}'.format(auc15))
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend(loc="lower right", fontsize=10)
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
fprval15,tprval15 = fpr15[np.argmax(tpr15-fpr15)], tpr15[np.argmax(tpr15-fpr15)]
arrowprops={'arrowstyle': '-', 'ls':'--'}
plt.annotate(str(fprval15), xy=(fprval15,tprval15), xytext=(fprval15, 0), 
              textcoords=plt.gca().get_xaxis_transform(),
              arrowprops=arrowprops,
              va='top', ha='center', fontsize=0)
plt.annotate(str(tprval15), xy=(fprval15,tprval15), xytext=(0, tprval15), 
              textcoords=plt.gca().get_yaxis_transform(),
              arrowprops=arrowprops,
              va='center', ha='right', fontsize=0)
plt.text(0.78,0.15, 'Sensitivity: {:0.3f}'.format(tprval15) , fontsize=10, bbox=dict(boxstyle='square', color='lightgray'))
plt.text(0.78,0.1, 'Specificity: {:0.3f}'.format(1-fprval15) , fontsize=10, bbox=dict(boxstyle='square', color='lightgray'))
plt.show()

#0: BUN, 1: MAP, 2:Na, 3: GCS, 4: NLR, 5: Creatinine, 6: Lactate, 7: Bilirubin, 8:ESR, 9: PLT, 10: CRP, 11: PCT, 12: AST, 13: APACHE, 14: SOFA
#ROC curve with significant variables + multiple logistic regression 
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 14})
plt.grid()
plt.plot(fpr15, tpr15, color='red', linestyle='solid', lw=2,
         label='MLR={0:.3f}'.format(auc15))
plt.plot(fpr11, tpr11, color='orange', linestyle='solid', lw=2,
         label='PCT={0:.3f}'.format(auc11))
plt.plot(fpr7, tpr7, color='yellow', linestyle='solid', lw=2,
         label='Bilirubin={0:.3f}'.format(auc7))
plt.plot(fpr4, tpr4, color='green', linestyle='solid', lw=2,
         label='NLR={0:.3f}'.format(auc4))
plt.plot(fpr9, tpr9, color='blue', linestyle='solid', lw=2,
         label='Platelets={0:.3f}'.format(auc9))
plt.plot(fpr6, tpr6, color='navy', linestyle='solid', lw=2,
         label='Lactic acid={0:.3f}'.format(auc6))
plt.plot(fpr3, tpr3, color='purple', linestyle='solid', lw=2,
         label='GCS score={0:.3f}'.format(auc3))
plt.plot(fpr8, tpr8, color='black', linestyle='solid', lw=2,
         label='ESR={0:.3f}'.format(auc8))
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='dotted')
plt.legend(loc="lower right", fontsize=10)
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()

#Log-rank Test
PositivePatients['Death Result dummy']=np.hsplit(pd.get_dummies(PositivePatients['Death Status']),2)[1]
NegativePatients['Death Result dummy']=np.hsplit(pd.get_dummies(NegativePatients['Death Status']),2)[1]
logResult = logrank_test(PositivePatients['Time to Death'], NegativePatients['Time to Death'], event_observed_A=PositivePatients['Death Result dummy'], event_observed_B=NegativePatients['Death Result dummy'])
logP = logResult.p_value

#Survival Curve Analysis: Kaplan-Meier Fitter
#Death Status: P=1, N=0
styles = ['--', '-']
colors = ['black', 'black']
lw = 3
kmf= KaplanMeierFitter()
ax = plt.subplot(111)
kmf.fit(NegativePatients['Time to Death'], event_observed=NegativePatients['Death Result dummy'], label = 'non-bacteremia')
ax = kmf.plot(ci_show=False, linewidth=lw, style=styles[1], c=colors[1], ylim=(0.5,1.02))
kmf.fit(PositivePatients['Time to Death'], event_observed=PositivePatients['Death Result dummy'], label = "bacteremia")
ax = kmf.plot(ci_show=False, linewidth=lw, style=styles[0], c=colors[0], ylim=(0.5,1.02))
ax.text(0.75, 0.7, '* P = 0.004', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, style = 'italic')
ax.set_xlabel('Time (in days)')
ax.set_ylabel('Survival Probability (%)')

# model0 = smf.logit('Y~CRP+PCT+SOFA+Bilirubin+PLT', data=Tester).fit()
# A = Tester[['CRP(mg/L)', 'PCT(ng/ml)', 'SOFA score', 'Bilirubin', 'Platelets']]
# A_C = sm.add_constant(A)
# AB = model0.predict(A_C)
# fpr, tp, th = roc_curve(y_true=Y, y_score=AB)
# auc = roc_auc_score(Y, AB)

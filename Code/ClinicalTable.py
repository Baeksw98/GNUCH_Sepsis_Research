"""
Created on November 16th, 2021
@author: Sangwon Baek
KyeongsangUniversity Bacteremia Clinical Table Code  
"""

import numpy as np
import pandas as pd 
import os
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
from scipy import stats

#setting working directory to the folder, use if putting the py file into the working folder does not work 
#This code will set the working directory to the file that you wish to work with 
path=r"C:\Users\user-pc\Desktop\KSU\Sepsis Research\Data\CSV"
os.chdir(path)
os.getcwd()

#Reading the files
BloodCultureData = pd.read_csv('BacteremiaData(2019).csv', encoding='cp949')
ClinicalData = pd.read_csv('ClinicalDataWithScores.csv', encoding='cp949')
ClinicalData = ClinicalData.drop(['Unnamed: 0','Registration Number', 'Age', 'Bacteremia Result'], axis=1)

#Merge the Clinical Data and Original Data
BacteremiaData = pd.merge(ClinicalData, BloodCultureData, on=['Serial Number'], how = 'left')

#Remove non-septic patients with SOFA score < 2
BacteremiaData = BacteremiaData[BacteremiaData['SOFA score']>=2]

#Convert the categorical values into a dummy value: 0:Negative | 1:Positive  
BacteremiaData['Bacteremia Result dummy']=np.hsplit(pd.get_dummies(BacteremiaData['Bacteremia Result']),2)[1]

#Select only the Bacteremia Positivse Patients from Total Patients from BacteremiaData data frame
PositivePatients = BacteremiaData.loc[BacteremiaData["Bacteremia Result"]=='Positive']

#Select only the Bacteremia Negative Patients from Total Patients from BacteremiaData data frame
NegativePatients = BacteremiaData.loc[BacteremiaData["Bacteremia Result"]=='Negative']

BacteremiaData = pd.concat([PositivePatients,NegativePatients],axis=0)
BacteremiaData = BacteremiaData.reset_index()
BacteremiaData = BacteremiaData.drop(['index'], axis=1)

#Creating a copy of BacteremiaData and name it TotalPatients
TotalPatients = BacteremiaData.copy()

#Select only the necessary columns in an organized manner
patientColumnNames = ['Serial Number', 'Age', 'Gender', 'Volume', 'Body Temperature', 'Body Weight', 'Bacteremia Result dummy', 'Bacteremia Result', 'Bacteremia Type','Death Status', 
                      'PaO2', 'FiO2', 'PaO2/FiO2', 'Respiratory Rate', 'Heart Rate', 'SBP', 'DBP', 'MAP', 'Arterial pH', 'K', 'Na', 'GCS Score', 'Seg Neutrophil', 
                      'Bands Neutrophil', 'Absolute Neutrophil Count', 'Lymphocyte', 'Absolute Lymphocyte Count', 'Neutrophil Lymphocyte Ratio', 'Creatinine', 
                      'Lactate Level', 'Bilirubin', 'AST', 'Platelets', 'BUN', 'WBC', 'Hematocrite', 'ESR', 'CRP(mg/L)', 'PCT(ng/ml)', 'Pitt Bacteremia score', 'APACHE II score', 'SOFA score',
                      'Diabetes Mellintus', 'Hypertension', 'Heart Failure', 'Cerebrovascular Disease', 'Renal Disease', 'Liver Disease', 'COPD', 'Known Neoplasm', 'Catheter Related Bloodstream Infection', 'Intra Abdominal Infection',
                      'Respiratory Tract Infection', 'Skin and Soft Tissue Infection', 'Urinary Tract Infection', 'Others', 'Fever of Unknown Origin','Prior Antibiotics']
TotalPatients = TotalPatients[patientColumnNames]
PositivePatients = PositivePatients[patientColumnNames]
NegativePatients = NegativePatients[patientColumnNames]

#________________Describing the distribution of each columns_______________________________________________________
#Continuous Variables : Med [IQR]
T_MedAge = int(TotalPatients['Age'].median())
T_Q1Age = int(np.percentile(TotalPatients['Age'], 25, interpolation = 'midpoint'))
T_Q3Age = int(np.percentile(TotalPatients['Age'], 75, interpolation = 'midpoint'))
T_Age = str(T_MedAge) + ' [' + str(T_Q1Age) + '-' + str(T_Q3Age) + ']'
P_MedAge = int(PositivePatients['Age'].median())
P_Q1Age = int(np.percentile(PositivePatients['Age'], 25, interpolation = 'midpoint'))
P_Q3Age = int(np.percentile(PositivePatients['Age'], 75, interpolation = 'midpoint'))
P_Age = str(P_MedAge) + ' [' + str(P_Q1Age) + '-' + str(P_Q3Age) + ']'
N_MedAge = int(NegativePatients['Age'].median())
N_Q1Age = int(np.percentile(NegativePatients['Age'], 25, interpolation = 'midpoint'))
N_Q3Age = int(np.percentile(NegativePatients['Age'], 75, interpolation = 'midpoint'))
N_Age = str(N_MedAge) + ' [' + str(N_Q1Age) + '-' + str(N_Q3Age) + ']'

T_MedVolume = round(TotalPatients['Volume'].median(),1)
T_Q1Volume = round(np.percentile(TotalPatients['Volume'], 25, interpolation = 'midpoint'),1)
T_Q3Volume = round(np.percentile(TotalPatients['Volume'], 75, interpolation = 'midpoint'),1)
T_Volume = str(T_MedVolume) + ' [' + str(T_Q1Volume) + '-' + str(T_Q3Volume) + ']'
P_MedVolume = round(PositivePatients['Volume'].median(),1)
P_Q1Volume = round(np.percentile(PositivePatients['Volume'], 25, interpolation = 'midpoint'),1)
P_Q3Volume = round(np.percentile(PositivePatients['Volume'], 75, interpolation = 'midpoint'),1)
P_Volume = str(P_MedVolume) + ' [' + str(P_Q1Volume) + '-' + str(P_Q3Volume) + ']'
N_MedVolume = round(NegativePatients['Volume'].median(),1)
N_Q1Volume = round(np.percentile(NegativePatients['Volume'], 25, interpolation = 'midpoint'),1)
N_Q3Volume = round(np.percentile(NegativePatients['Volume'], 75, interpolation = 'midpoint'),1)
N_Volume = str(N_MedVolume) + ' [' + str(N_Q1Volume) + '-' + str(N_Q3Volume) + ']'

T_MedBT = round(TotalPatients['Body Temperature'].median(),1)
T_Q1BT = round(np.percentile(TotalPatients['Body Temperature'], 25, interpolation = 'midpoint'),1)
T_Q3BT = round(np.percentile(TotalPatients['Body Temperature'], 75, interpolation = 'midpoint'),1)
T_BT = str(T_MedBT) + ' [' + str(T_Q1BT) + '-' + str(T_Q3BT) + ']'
P_MedBT = round(PositivePatients['Body Temperature'].median(),1)
P_Q1BT = round(np.percentile(PositivePatients['Body Temperature'], 25, interpolation = 'midpoint'),1)
P_Q3BT = round(np.percentile(PositivePatients['Body Temperature'], 75, interpolation = 'midpoint'),1)
P_BT = str(P_MedBT) + ' [' + str(P_Q1BT) + '-' + str(P_Q3BT) + ']'
N_MedBT = round(NegativePatients['Body Temperature'].median(),1)
N_Q1BT = round(np.percentile(NegativePatients['Body Temperature'], 25, interpolation = 'midpoint'),1)
N_Q3BT = round(np.percentile(NegativePatients['Body Temperature'], 75, interpolation = 'midpoint'),1)
N_BT = str(N_MedBT) + ' [' + str(N_Q1BT) + '-' + str(N_Q3BT) + ']'

T_MedBW = round(TotalPatients['Body Weight'].median(),1)
T_Q1BW = round(np.percentile(TotalPatients['Body Weight'], 25, interpolation = 'midpoint'),1)
T_Q3BW = round(np.percentile(TotalPatients['Body Weight'], 75, interpolation = 'midpoint'),1)
T_BW = str(T_MedBW) + ' [' + str(T_Q1BW) + '-' + str(T_Q3BW) + ']'
P_MedBW = round(PositivePatients['Body Weight'].median(),1)
P_Q1BW = round(np.percentile(PositivePatients['Body Weight'], 25, interpolation = 'midpoint'),1)
P_Q3BW = round(np.percentile(PositivePatients['Body Weight'], 75, interpolation = 'midpoint'),1)
P_BW = str(P_MedBW) + ' [' + str(P_Q1BW) + '-' + str(P_Q3BW) + ']'
N_MedBW = round(NegativePatients['Body Weight'].median(),1)
N_Q1BW = round(np.percentile(NegativePatients['Body Weight'], 25, interpolation = 'midpoint'),1)
N_Q3BW = round(np.percentile(NegativePatients['Body Weight'], 75, interpolation = 'midpoint'),1)
N_BW = str(N_MedBW) + ' [' + str(N_Q1BW) + '-' + str(N_Q3BW) + ']'

T_MedPaO2 = int(TotalPatients['PaO2'].median())
T_Q1PaO2 = int(np.percentile(TotalPatients['PaO2'], 25, interpolation = 'midpoint'))
T_Q3PaO2 = int(np.percentile(TotalPatients['PaO2'], 75, interpolation = 'midpoint'))
T_PaO2 = str(T_MedPaO2) + ' [' + str(T_Q1PaO2) + '-' + str(T_Q3PaO2) + ']'
P_MedPaO2 = int(PositivePatients['PaO2'].median())
P_Q1PaO2 = int(np.percentile(PositivePatients['PaO2'], 25, interpolation = 'midpoint'))
P_Q3PaO2 = int(np.percentile(PositivePatients['PaO2'], 75, interpolation = 'midpoint'))
P_PaO2 = str(P_MedPaO2) + ' [' + str(P_Q1PaO2) + '-' + str(P_Q3PaO2) + ']'
N_MedPaO2 = int(NegativePatients['PaO2'].median())
N_Q1PaO2 = int(np.percentile(NegativePatients['PaO2'], 25, interpolation = 'midpoint'))
N_Q3PaO2 = int(np.percentile(NegativePatients['PaO2'], 75, interpolation = 'midpoint'))
N_PaO2 = str(N_MedPaO2) + ' [' + str(N_Q1PaO2) + '-' + str(N_Q3PaO2) + ']'

T_MedFiO2 = round(TotalPatients['FiO2'].median(),1)
T_Q1FiO2 = round(np.percentile(TotalPatients['FiO2'], 25, interpolation = 'midpoint'),1)
T_Q3FiO2 = round(np.percentile(TotalPatients['FiO2'], 75, interpolation = 'midpoint'),1)
T_FiO2 = str(T_MedFiO2) + ' [' + str(T_Q1FiO2) + '-' + str(T_Q3FiO2) + ']'
P_MedFiO2 = round(PositivePatients['FiO2'].median(),1)
P_Q1FiO2 = round(np.percentile(PositivePatients['FiO2'], 25, interpolation = 'midpoint'),1)
P_Q3FiO2 = round(np.percentile(PositivePatients['FiO2'], 75, interpolation = 'midpoint'),1)
P_FiO2 = str(P_MedFiO2) + ' [' + str(P_Q1FiO2) + '-' + str(P_Q3FiO2) + ']'
N_MedFiO2 = round(NegativePatients['FiO2'].median(),1)
N_Q1FiO2 = round(np.percentile(NegativePatients['FiO2'], 25, interpolation = 'midpoint'),1)
N_Q3FiO2 = round(np.percentile(NegativePatients['FiO2'], 75, interpolation = 'midpoint'),1)
N_FiO2 = str(N_MedFiO2) + ' [' + str(N_Q1FiO2) + '-' + str(N_Q3FiO2) + ']'

T_MedPaO2FiO2 = int(TotalPatients['PaO2/FiO2'].median())
T_Q1PaO2FiO2 = int(np.percentile(TotalPatients['PaO2/FiO2'], 25, interpolation = 'midpoint'))
T_Q3PaO2FiO2 = int(np.percentile(TotalPatients['PaO2/FiO2'], 75, interpolation = 'midpoint'))
T_PaO2FiO2 = str(T_MedPaO2FiO2) + ' [' + str(T_Q1PaO2FiO2) + '-' + str(T_Q3PaO2FiO2) + ']'
P_MedPaO2FiO2 = int(PositivePatients['PaO2/FiO2'].median())
P_Q1PaO2FiO2 = int(np.percentile(PositivePatients['PaO2/FiO2'], 25, interpolation = 'midpoint'))
P_Q3PaO2FiO2 = int(np.percentile(PositivePatients['PaO2/FiO2'], 75, interpolation = 'midpoint'))
P_PaO2FiO2 = str(P_MedPaO2FiO2) + ' [' + str(P_Q1PaO2FiO2) + '-' + str(P_Q3PaO2FiO2) + ']'
N_MedPaO2FiO2 = int(NegativePatients['PaO2/FiO2'].median())
N_Q1PaO2FiO2 = int(np.percentile(NegativePatients['PaO2/FiO2'], 25, interpolation = 'midpoint'))
N_Q3PaO2FiO2 = int(np.percentile(NegativePatients['PaO2/FiO2'], 75, interpolation = 'midpoint'))
N_PaO2FiO2 = str(N_MedPaO2FiO2) + ' [' + str(N_Q1PaO2FiO2) + '-' + str(N_Q3PaO2FiO2) + ']'

T_MedRR = int(TotalPatients['Respiratory Rate'].median())
T_Q1RR = int(np.percentile(TotalPatients['Respiratory Rate'], 25, interpolation = 'midpoint'))
T_Q3RR = int(np.percentile(TotalPatients['Respiratory Rate'], 75, interpolation = 'midpoint'))
T_RR = str(T_MedRR) + ' [' + str(T_Q1RR) + '-' + str(T_Q3RR) + ']'
P_MedRR = int(PositivePatients['Respiratory Rate'].median())
P_Q1RR = int(np.percentile(PositivePatients['Respiratory Rate'], 25, interpolation = 'midpoint'))
P_Q3RR = int(np.percentile(PositivePatients['Respiratory Rate'], 75, interpolation = 'midpoint'))
P_RR = str(P_MedRR) + ' [' + str(P_Q1RR) + '-' + str(P_Q3RR) + ']'
N_MedRR = int(NegativePatients['Respiratory Rate'].median())
N_Q1RR = int(np.percentile(NegativePatients['Respiratory Rate'], 25, interpolation = 'midpoint'))
N_Q3RR = int(np.percentile(NegativePatients['Respiratory Rate'], 75, interpolation = 'midpoint'))
N_RR = str(N_MedRR) + ' [' + str(N_Q1RR) + '-' + str(N_Q3RR) + ']'

T_MedHR = int(TotalPatients['Heart Rate'].median())
T_Q1HR = int(np.percentile(TotalPatients['Heart Rate'], 25, interpolation = 'midpoint'))
T_Q3HR = int(np.percentile(TotalPatients['Heart Rate'], 75, interpolation = 'midpoint'))
T_HR = str(T_MedHR) + ' [' + str(T_Q1HR) + '-' + str(T_Q3HR) + ']'
P_MedHR = int(PositivePatients['Heart Rate'].median())
P_Q1HR = int(np.percentile(PositivePatients['Heart Rate'], 25, interpolation = 'midpoint'))
P_Q3HR = int(np.percentile(PositivePatients['Heart Rate'], 75, interpolation = 'midpoint'))
P_HR = str(P_MedHR) + ' [' + str(P_Q1HR) + '-' + str(P_Q3HR) + ']'
N_MedHR = int(NegativePatients['Heart Rate'].median())
N_Q1HR = int(np.percentile(NegativePatients['Heart Rate'], 25, interpolation = 'midpoint'))
N_Q3HR = int(np.percentile(NegativePatients['Heart Rate'], 75, interpolation = 'midpoint'))
N_HR = str(N_MedHR) + ' [' + str(N_Q1HR) + '-' + str(N_Q3HR) + ']'

T_MedSBP = int(TotalPatients['SBP'].median())
T_Q1SBP = int(np.percentile(TotalPatients['SBP'], 25, interpolation = 'midpoint'))
T_Q3SBP = int(np.percentile(TotalPatients['SBP'], 75, interpolation = 'midpoint'))
T_SBP = str(T_MedSBP) + ' [' + str(T_Q1SBP) + '-' + str(T_Q3SBP) + ']'
P_MedSBP = int(PositivePatients['SBP'].median())
P_Q1SBP = int(np.percentile(PositivePatients['SBP'], 25, interpolation = 'midpoint'))
P_Q3SBP = int(np.percentile(PositivePatients['SBP'], 75, interpolation = 'midpoint'))
P_SBP = str(P_MedSBP) + ' [' + str(P_Q1SBP) + '-' + str(P_Q3SBP) + ']'
N_MedSBP = int(NegativePatients['SBP'].median())
N_Q1SBP = int(np.percentile(NegativePatients['SBP'], 25, interpolation = 'midpoint'))
N_Q3SBP = int(np.percentile(NegativePatients['SBP'], 75, interpolation = 'midpoint'))
N_SBP = str(N_MedSBP) + ' [' + str(N_Q1SBP) + '-' + str(N_Q3SBP) + ']'

T_MedDBP = int(TotalPatients['DBP'].median())
T_Q1DBP = int(np.percentile(TotalPatients['DBP'], 25, interpolation = 'midpoint'))
T_Q3DBP = int(np.percentile(TotalPatients['DBP'], 75, interpolation = 'midpoint'))
T_DBP = str(T_MedDBP) + ' [' + str(T_Q1DBP) + '-' + str(T_Q3DBP) + ']'
P_MedDBP = int(PositivePatients['DBP'].median())
P_Q1DBP = int(np.percentile(PositivePatients['DBP'], 25, interpolation = 'midpoint'))
P_Q3DBP = int(np.percentile(PositivePatients['DBP'], 75, interpolation = 'midpoint'))
P_DBP = str(P_MedDBP) + ' [' + str(P_Q1DBP) + '-' + str(P_Q3DBP) + ']'
N_MedDBP = int(NegativePatients['DBP'].median())
N_Q1DBP = int(np.percentile(NegativePatients['DBP'], 25, interpolation = 'midpoint'))
N_Q3DBP = int(np.percentile(NegativePatients['DBP'], 75, interpolation = 'midpoint'))
N_DBP = str(N_MedDBP) + ' [' + str(N_Q1DBP) + '-' + str(N_Q3DBP) + ']'

T_MedMAP = int(TotalPatients['MAP'].median())
T_Q1MAP = int(np.percentile(TotalPatients['MAP'], 25, interpolation = 'midpoint'))
T_Q3MAP = int(np.percentile(TotalPatients['MAP'], 75, interpolation = 'midpoint'))
T_MAP = str(T_MedMAP) + ' [' + str(T_Q1MAP) + '-' + str(T_Q3MAP) + ']'
P_MedMAP = int(PositivePatients['MAP'].median())
P_Q1MAP = int(np.percentile(PositivePatients['MAP'], 25, interpolation = 'midpoint'))
P_Q3MAP = int(np.percentile(PositivePatients['MAP'], 75, interpolation = 'midpoint'))
P_MAP = str(P_MedMAP) + ' [' + str(P_Q1MAP) + '-' + str(P_Q3MAP) + ']'
N_MedMAP = int(NegativePatients['MAP'].median())
N_Q1MAP = int(np.percentile(NegativePatients['MAP'], 25, interpolation = 'midpoint'))
N_Q3MAP = int(np.percentile(NegativePatients['MAP'], 75, interpolation = 'midpoint'))
N_MAP = str(N_MedMAP) + ' [' + str(N_Q1MAP) + '-' + str(N_Q3MAP) + ']'

T_MedPH = round(TotalPatients['Arterial pH'].median(),1)
T_Q1PH = round(np.percentile(TotalPatients['Arterial pH'], 25, interpolation = 'midpoint'),1)
T_Q3PH = round(np.percentile(TotalPatients['Arterial pH'], 75, interpolation = 'midpoint'),1)
T_PH = str(T_MedPH) + ' [' + str(T_Q1PH) + '-' + str(T_Q3PH) + ']'
P_MedPH = round(PositivePatients['Arterial pH'].median(),1)
P_Q1PH = round(np.percentile(PositivePatients['Arterial pH'], 25, interpolation = 'midpoint'),1)
P_Q3PH = round(np.percentile(PositivePatients['Arterial pH'], 75, interpolation = 'midpoint'),1)
P_PH = str(P_MedPH) + ' [' + str(P_Q1PH) + '-' + str(P_Q3PH) + ']'
N_MedPH = round(NegativePatients['Arterial pH'].median(),1)
N_Q1PH = round(np.percentile(NegativePatients['Arterial pH'], 25, interpolation = 'midpoint'),1)
N_Q3PH = round(np.percentile(NegativePatients['Arterial pH'], 75, interpolation = 'midpoint'),1)
N_PH = str(N_MedPH) + ' [' + str(N_Q1PH) + '-' + str(N_Q3PH) + ']'

T_MedK = round(TotalPatients['K'].median(),1)
T_Q1K = round(np.percentile(TotalPatients['K'], 25, interpolation = 'midpoint'),1)
T_Q3K = round(np.percentile(TotalPatients['K'], 75, interpolation = 'midpoint'),1)
T_K = str(T_MedK) + ' [' + str(T_Q1K) + '-' + str(T_Q3K) + ']'
P_MedK = round(PositivePatients['K'].median(),1)
P_Q1K = round(np.percentile(PositivePatients['K'], 25, interpolation = 'midpoint'),1)
P_Q3K = round(np.percentile(PositivePatients['K'], 75, interpolation = 'midpoint'),1)
P_K = str(P_MedK) + ' [' + str(P_Q1K) + '-' + str(P_Q3K) + ']'
N_MedK = round(NegativePatients['K'].median(),1)
N_Q1K = round(np.percentile(NegativePatients['K'], 25, interpolation = 'midpoint'),1)
N_Q3K = round(np.percentile(NegativePatients['K'], 75, interpolation = 'midpoint'),1)
N_K = str(N_MedK) + ' [' + str(N_Q1K) + '-' + str(N_Q3K) + ']'

T_MedNA = round(TotalPatients['Na'].median(),1)
T_Q1NA = round(np.percentile(TotalPatients['Na'], 25, interpolation = 'midpoint'),1)
T_Q3NA = round(np.percentile(TotalPatients['Na'], 75, interpolation = 'midpoint'),1)
T_NA = str(T_MedNA) + ' [' + str(T_Q1NA) + '-' + str(T_Q3NA) + ']'
P_MedNA = round(PositivePatients['Na'].median(),1)
P_Q1NA = round(np.percentile(PositivePatients['Na'], 25, interpolation = 'midpoint'),1)
P_Q3NA = round(np.percentile(PositivePatients['Na'], 75, interpolation = 'midpoint'),1)
P_NA = str(P_MedNA) + ' [' + str(P_Q1NA) + '-' + str(P_Q3NA) + ']'
N_MedNA = round(NegativePatients['Na'].median(),1)
N_Q1NA = round(np.percentile(NegativePatients['Na'], 25, interpolation = 'midpoint'),1)
N_Q3NA = round(np.percentile(NegativePatients['Na'], 75, interpolation = 'midpoint'),1)
N_NA = str(N_MedNA) + ' [' + str(N_Q1NA) + '-' + str(N_Q3NA) + ']'

T_MedGCS = int(TotalPatients['GCS Score'].median())
T_Q1GCS = int(np.percentile(TotalPatients['GCS Score'], 25, interpolation = 'midpoint'))
T_Q3GCS = int(np.percentile(TotalPatients['GCS Score'], 75, interpolation = 'midpoint'))
T_GCS = str(T_MedGCS) + ' [' + str(T_Q1GCS) + '-' + str(T_Q3GCS) + ']'
P_MedGCS = int(PositivePatients['GCS Score'].median())
P_Q1GCS = int(np.percentile(PositivePatients['GCS Score'], 25, interpolation = 'midpoint'))
P_Q3GCS = int(np.percentile(PositivePatients['GCS Score'], 75, interpolation = 'midpoint'))
P_GCS = str(P_MedGCS) + ' [' + str(P_Q1GCS) + '-' + str(P_Q3GCS) + ']'
N_MedGCS = int(NegativePatients['GCS Score'].median())
N_Q1GCS = int(np.percentile(NegativePatients['GCS Score'], 25, interpolation = 'midpoint'))
N_Q3GCS = int(np.percentile(NegativePatients['GCS Score'], 75, interpolation = 'midpoint'))
N_GCS = str(N_MedGCS) + ' [' + str(N_Q1GCS) + '-' + str(N_Q3GCS) + ']'

T_MedSegNeutrophil = round(TotalPatients['Seg Neutrophil'].median(),2)
T_Q1SegNeutrophil = round(np.percentile(TotalPatients['Seg Neutrophil'], 25, interpolation = 'midpoint'),2)
T_Q3SegNeutrophil = round(np.percentile(TotalPatients['Seg Neutrophil'], 75, interpolation = 'midpoint'),2)
T_SegNeutrophil = str(T_MedSegNeutrophil) + ' [' + str(T_Q1SegNeutrophil) + '-' + str(T_Q3SegNeutrophil) + ']'
P_MedSegNeutrophil = round(PositivePatients['Seg Neutrophil'].median(),2)
P_Q1SegNeutrophil = round(np.percentile(PositivePatients['Seg Neutrophil'], 25, interpolation = 'midpoint'),2)
P_Q3SegNeutrophil = round(np.percentile(PositivePatients['Seg Neutrophil'], 75, interpolation = 'midpoint'),2)
P_SegNeutrophil = str(P_MedSegNeutrophil) + ' [' + str(P_Q1SegNeutrophil) + '-' + str(P_Q3SegNeutrophil) + ']'
N_MedSegNeutrophil = round(NegativePatients['Seg Neutrophil'].median(),2)
N_Q1SegNeutrophil = round(np.percentile(NegativePatients['Seg Neutrophil'], 25, interpolation = 'midpoint'),2)
N_Q3SegNeutrophil = round(np.percentile(NegativePatients['Seg Neutrophil'], 75, interpolation = 'midpoint'),2)
N_SegNeutrophil = str(N_MedSegNeutrophil) + ' [' + str(N_Q1SegNeutrophil) + '-' + str(N_Q3SegNeutrophil) + ']'

T_MedBandsNeutrophil = round(TotalPatients['Bands Neutrophil'].median(),2)
T_Q1BandsNeutrophil = round(np.percentile(TotalPatients['Bands Neutrophil'], 25, interpolation = 'midpoint'),2)
T_Q3BandsNeutrophil = round(np.percentile(TotalPatients['Bands Neutrophil'], 75, interpolation = 'midpoint'),2)
T_BandsNeutrophil = str(T_MedBandsNeutrophil) + ' [' + str(T_Q1BandsNeutrophil) + '-' + str(T_Q3BandsNeutrophil) + ']'
P_MedBandsNeutrophil = round(PositivePatients['Bands Neutrophil'].median(),2)
P_Q1BandsNeutrophil = round(np.percentile(PositivePatients['Bands Neutrophil'], 25, interpolation = 'midpoint'),2)
P_Q3BandsNeutrophil = round(np.percentile(PositivePatients['Bands Neutrophil'], 75, interpolation = 'midpoint'),2)
P_BandsNeutrophil = str(P_MedBandsNeutrophil) + ' [' + str(P_Q1BandsNeutrophil) + '-' + str(P_Q3BandsNeutrophil) + ']'
N_MedBandsNeutrophil = round(NegativePatients['Bands Neutrophil'].median(),2)
N_Q1BandsNeutrophil = round(np.percentile(NegativePatients['Bands Neutrophil'], 25, interpolation = 'midpoint'),2)
N_Q3BandsNeutrophil = round(np.percentile(NegativePatients['Bands Neutrophil'], 75, interpolation = 'midpoint'),2)
N_BandsNeutrophil = str(N_MedBandsNeutrophil) + ' [' + str(N_Q1BandsNeutrophil) + '-' + str(N_Q3BandsNeutrophil) + ']'

T_MedANC = int(TotalPatients['Absolute Neutrophil Count'].median())
T_Q1ANC = int(np.percentile(TotalPatients['Absolute Neutrophil Count'], 25, interpolation = 'midpoint'))
T_Q3ANC = int(np.percentile(TotalPatients['Absolute Neutrophil Count'], 75, interpolation = 'midpoint'))
T_ANC = str(T_MedANC) + ' [' + str(T_Q1ANC) + '-' + str(T_Q3ANC) + ']'
P_MedANC = int(PositivePatients['Absolute Neutrophil Count'].median())
P_Q1ANC = int(np.percentile(PositivePatients['Absolute Neutrophil Count'], 25, interpolation = 'midpoint'))
P_Q3ANC = int(np.percentile(PositivePatients['Absolute Neutrophil Count'], 75, interpolation = 'midpoint'))
P_ANC = str(P_MedANC) + ' [' + str(P_Q1ANC) + '-' + str(P_Q3ANC) + ']'
N_MedANC = int(NegativePatients['Absolute Neutrophil Count'].median())
N_Q1ANC = int(np.percentile(NegativePatients['Absolute Neutrophil Count'], 25, interpolation = 'midpoint'))
N_Q3ANC = int(np.percentile(NegativePatients['Absolute Neutrophil Count'], 75, interpolation = 'midpoint'))
N_ANC = str(N_MedANC) + ' [' + str(N_Q1ANC) + '-' + str(N_Q3ANC) + ']'

T_MedLymphocyte = int(TotalPatients['Lymphocyte'].median())
T_Q1Lymphocyte = int(np.percentile(TotalPatients['Lymphocyte'], 25, interpolation = 'midpoint'))
T_Q3Lymphocyte = int(np.percentile(TotalPatients['Lymphocyte'], 75, interpolation = 'midpoint'))
T_Lymphocyte = str(T_MedLymphocyte) + ' [' + str(T_Q1Lymphocyte) + '-' + str(T_Q3Lymphocyte) + ']'
P_MedLymphocyte = int(PositivePatients['Lymphocyte'].median())
P_Q1Lymphocyte = int(np.percentile(PositivePatients['Lymphocyte'], 25, interpolation = 'midpoint'))
P_Q3Lymphocyte = int(np.percentile(PositivePatients['Lymphocyte'], 75, interpolation = 'midpoint'))
P_Lymphocyte = str(P_MedLymphocyte) + ' [' + str(P_Q1Lymphocyte) + '-' + str(P_Q3Lymphocyte) + ']'
N_MedLymphocyte = int(NegativePatients['Lymphocyte'].median())
N_Q1Lymphocyte = int(np.percentile(NegativePatients['Lymphocyte'], 25, interpolation = 'midpoint'))
N_Q3Lymphocyte = int(np.percentile(NegativePatients['Lymphocyte'], 75, interpolation = 'midpoint'))
N_Lymphocyte = str(N_MedLymphocyte) + ' [' + str(N_Q1Lymphocyte) + '-' + str(N_Q3Lymphocyte) + ']'

T_MedALC = round(TotalPatients['Absolute Lymphocyte Count'].median(),2)
T_Q1ALC = round(np.percentile(TotalPatients['Absolute Lymphocyte Count'], 25, interpolation = 'midpoint'),2)
T_Q3ALC = round(np.percentile(TotalPatients['Absolute Lymphocyte Count'], 75, interpolation = 'midpoint'),2)
T_ALC = str(T_MedALC) + ' [' + str(T_Q1ALC) + '-' + str(T_Q3ALC) + ']'
P_MedALC = round(PositivePatients['Absolute Lymphocyte Count'].median(),2)
P_Q1ALC = round(np.percentile(PositivePatients['Absolute Lymphocyte Count'], 25, interpolation = 'midpoint'),2)
P_Q3ALC = round(np.percentile(PositivePatients['Absolute Lymphocyte Count'], 75, interpolation = 'midpoint'),2)
P_ALC = str(P_MedALC) + ' [' + str(P_Q1ALC) + '-' + str(P_Q3ALC) + ']'
N_MedALC = round(NegativePatients['Absolute Lymphocyte Count'].median(),2)
N_Q1ALC = round(np.percentile(NegativePatients['Absolute Lymphocyte Count'], 25, interpolation = 'midpoint'),2)
N_Q3ALC = round(np.percentile(NegativePatients['Absolute Lymphocyte Count'], 75, interpolation = 'midpoint'),2)
N_ALC = str(N_MedALC) + ' [' + str(N_Q1ALC) + '-' + str(N_Q3ALC) + ']'

T_MedNLR = round(TotalPatients['Neutrophil Lymphocyte Ratio'].median(),1)
T_Q1NLR = round(np.percentile(TotalPatients['Neutrophil Lymphocyte Ratio'], 25, interpolation = 'midpoint'),1)
T_Q3NLR = round(np.percentile(TotalPatients['Neutrophil Lymphocyte Ratio'], 75, interpolation = 'midpoint'),1)
T_NLR = str(T_MedNLR) + ' [' + str(T_Q1NLR) + '-' + str(T_Q3NLR) + ']'
P_MedNLR = round(PositivePatients['Neutrophil Lymphocyte Ratio'].median(),1)
P_Q1NLR = round(np.percentile(PositivePatients['Neutrophil Lymphocyte Ratio'], 25, interpolation = 'midpoint'),1)
P_Q3NLR = round(np.percentile(PositivePatients['Neutrophil Lymphocyte Ratio'], 75, interpolation = 'midpoint'),1)
P_NLR = str(P_MedNLR) + ' [' + str(P_Q1NLR) + '-' + str(P_Q3NLR) + ']'
N_MedNLR = round(NegativePatients['Neutrophil Lymphocyte Ratio'].median(),1)
N_Q1NLR = round(np.percentile(NegativePatients['Neutrophil Lymphocyte Ratio'], 25, interpolation = 'midpoint'),1)
N_Q3NLR = round(np.percentile(NegativePatients['Neutrophil Lymphocyte Ratio'], 75, interpolation = 'midpoint'),1)
N_NLR = str(N_MedNLR) + ' [' + str(N_Q1NLR) + '-' + str(N_Q3NLR) + ']'

T_MedCreatinine = round(TotalPatients['Creatinine'].median(),2)
T_Q1Creatinine = round(np.percentile(TotalPatients['Creatinine'], 25, interpolation = 'midpoint'),2)
T_Q3Creatinine = round(np.percentile(TotalPatients['Creatinine'], 75, interpolation = 'midpoint'),2)
T_Creatinine = str(T_MedCreatinine) + ' [' + str(T_Q1Creatinine) + '-' + str(T_Q3Creatinine) + ']'
P_MedCreatinine = round(PositivePatients['Creatinine'].median(),2)
P_Q1Creatinine = round(np.percentile(PositivePatients['Creatinine'], 25, interpolation = 'midpoint'),2)
P_Q3Creatinine = round(np.percentile(PositivePatients['Creatinine'], 75, interpolation = 'midpoint'),2)
P_Creatinine = str(P_MedCreatinine) + ' [' + str(P_Q1Creatinine) + '-' + str(P_Q3Creatinine) + ']'
N_MedCreatinine = round(NegativePatients['Creatinine'].median(),2)
N_Q1Creatinine = round(np.percentile(NegativePatients['Creatinine'], 25, interpolation = 'midpoint'),2)
N_Q3Creatinine = round(np.percentile(NegativePatients['Creatinine'], 75, interpolation = 'midpoint'),2)
N_Creatinine = str(N_MedCreatinine) + ' [' + str(N_Q1Creatinine) + '-' + str(N_Q3Creatinine) + ']'

T_MedLactate = round(TotalPatients['Lactate Level'].median(),1)
T_Q1Lactate = round(np.percentile(TotalPatients['Lactate Level'], 25, interpolation = 'midpoint'),1)
T_Q3Lactate = round(np.percentile(TotalPatients['Lactate Level'], 75, interpolation = 'midpoint'),1)
T_Lactate = str(T_MedLactate) + ' [' + str(T_Q1Lactate) + '-' + str(T_Q3Lactate) + ']'
P_MedLactate = round(PositivePatients['Lactate Level'].median(),1)
P_Q1Lactate = round(np.percentile(PositivePatients['Lactate Level'], 25, interpolation = 'midpoint'),1)
P_Q3Lactate = round(np.percentile(PositivePatients['Lactate Level'], 75, interpolation = 'midpoint'),1)
P_Lactate = str(P_MedLactate) + ' [' + str(P_Q1Lactate) + '-' + str(P_Q3Lactate) + ']'
N_MedLactate = round(NegativePatients['Lactate Level'].median(),1)
N_Q1Lactate = round(np.percentile(NegativePatients['Lactate Level'], 25, interpolation = 'midpoint'),1)
N_Q3Lactate = round(np.percentile(NegativePatients['Lactate Level'], 75, interpolation = 'midpoint'),1)
N_Lactate = str(N_MedLactate) + ' [' + str(N_Q1Lactate) + '-' + str(N_Q3Lactate) + ']'

T_MedBilirubin = round(TotalPatients['Bilirubin'].median(),2)
T_Q1Bilirubin = round(np.percentile(TotalPatients['Bilirubin'], 25, interpolation = 'midpoint'),2)
T_Q3Bilirubin = round(np.percentile(TotalPatients['Bilirubin'], 75, interpolation = 'midpoint'),2)
T_Bilirubin = str(T_MedBilirubin) + ' [' + str(T_Q1Bilirubin) + '-' + str(T_Q3Bilirubin) + ']'
P_MedBilirubin = round(PositivePatients['Bilirubin'].median(),2)
P_Q1Bilirubin = round(np.percentile(PositivePatients['Bilirubin'], 25, interpolation = 'midpoint'),2)
P_Q3Bilirubin = round(np.percentile(PositivePatients['Bilirubin'], 75, interpolation = 'midpoint'),2)
P_Bilirubin = str(P_MedBilirubin) + ' [' + str(P_Q1Bilirubin) + '-' + str(P_Q3Bilirubin) + ']'
N_MedBilirubin = round(NegativePatients['Bilirubin'].median(),2)
N_Q1Bilirubin = round(np.percentile(NegativePatients['Bilirubin'], 25, interpolation = 'midpoint'),2)
N_Q3Bilirubin = round(np.percentile(NegativePatients['Bilirubin'], 75, interpolation = 'midpoint'),2)
N_Bilirubin = str(N_MedBilirubin) + ' [' + str(N_Q1Bilirubin) + '-' + str(N_Q3Bilirubin) + ']'

T_MedAST = round(TotalPatients['AST'].median(),1)
T_Q1AST = round(np.percentile(TotalPatients['AST'], 25, interpolation = 'midpoint'),1)
T_Q3AST = round(np.percentile(TotalPatients['AST'], 75, interpolation = 'midpoint'),1)
T_AST = str(T_MedAST) + ' [' + str(T_Q1AST) + '-' + str(T_Q3AST) + ']'
P_MedAST = round(PositivePatients['AST'].median(),1)
P_Q1AST = round(np.percentile(PositivePatients['AST'], 25, interpolation = 'midpoint'),1)
P_Q3AST = round(np.percentile(PositivePatients['AST'], 75, interpolation = 'midpoint'),1)
P_AST = str(P_MedAST) + ' [' + str(P_Q1AST) + '-' + str(P_Q3AST) + ']'
N_MedAST = round(NegativePatients['AST'].median(),1)
N_Q1AST = round(np.percentile(NegativePatients['AST'], 25, interpolation = 'midpoint'),1)
N_Q3AST = round(np.percentile(NegativePatients['AST'], 75, interpolation = 'midpoint'),1)
N_AST = str(N_MedAST) + ' [' + str(N_Q1AST) + '-' + str(N_Q3AST) + ']'

T_MedPLT = int(TotalPatients['Platelets'].median())
T_Q1PLT = int(np.percentile(TotalPatients['Platelets'], 25, interpolation = 'midpoint'))
T_Q3PLT = int(np.percentile(TotalPatients['Platelets'], 75, interpolation = 'midpoint'))
T_PLT = str(T_MedPLT) + ' [' + str(T_Q1PLT) + '-' + str(T_Q3PLT) + ']'
P_MedPLT = int(PositivePatients['Platelets'].median())
P_Q1PLT = int(np.percentile(PositivePatients['Platelets'], 25, interpolation = 'midpoint'))
P_Q3PLT = int(np.percentile(PositivePatients['Platelets'], 75, interpolation = 'midpoint'))
P_PLT = str(P_MedPLT) + ' [' + str(P_Q1PLT) + '-' + str(P_Q3PLT) + ']'
N_MedPLT = int(NegativePatients['Platelets'].median())
N_Q1PLT = int(np.percentile(NegativePatients['Platelets'], 25, interpolation = 'midpoint'))
N_Q3PLT = int(np.percentile(NegativePatients['Platelets'], 75, interpolation = 'midpoint'))
N_PLT = str(N_MedPLT) + ' [' + str(N_Q1PLT) + '-' + str(N_Q3PLT) + ']'

T_MedBUN = round(TotalPatients['BUN'].median(),1)
T_Q1BUN = round(np.percentile(TotalPatients['BUN'], 25, interpolation = 'midpoint'),1)
T_Q3BUN = round(np.percentile(TotalPatients['BUN'], 75, interpolation = 'midpoint'),1)
T_BUN = str(T_MedBUN) + ' [' + str(T_Q1BUN) + '-' + str(T_Q3BUN) + ']'
P_MedBUN = round(PositivePatients['BUN'].median(),1)
P_Q1BUN = round(np.percentile(PositivePatients['BUN'], 25, interpolation = 'midpoint'),1)
P_Q3BUN = round(np.percentile(PositivePatients['BUN'], 75, interpolation = 'midpoint'),1)
P_BUN = str(P_MedBUN) + ' [' + str(P_Q1BUN) + '-' + str(P_Q3BUN) + ']'
N_MedBUN = round(NegativePatients['BUN'].median(),1)
N_Q1BUN = round(np.percentile(NegativePatients['BUN'], 25, interpolation = 'midpoint'),1)
N_Q3BUN = round(np.percentile(NegativePatients['BUN'], 75, interpolation = 'midpoint'),1)
N_BUN = str(N_MedBUN) + ' [' + str(N_Q1BUN) + '-' + str(N_Q3BUN) + ']'

T_MedWBC = round(TotalPatients['WBC'].median(),1)
T_Q1WBC = round(np.percentile(TotalPatients['WBC'], 25, interpolation = 'midpoint'),1)
T_Q3WBC = round(np.percentile(TotalPatients['WBC'], 75, interpolation = 'midpoint'),1)
T_WBC = str(T_MedWBC) + ' [' + str(T_Q1WBC) + '-' + str(T_Q3WBC) + ']'
P_MedWBC = round(PositivePatients['WBC'].median(),1)
P_Q1WBC = round(np.percentile(PositivePatients['WBC'], 25, interpolation = 'midpoint'),1)
P_Q3WBC = round(np.percentile(PositivePatients['WBC'], 75, interpolation = 'midpoint'),1)
P_WBC = str(P_MedWBC) + ' [' + str(P_Q1WBC) + '-' + str(P_Q3WBC) + ']'
N_MedWBC = round(NegativePatients['WBC'].median(),1)
N_Q1WBC = round(np.percentile(NegativePatients['WBC'], 25, interpolation = 'midpoint'),1)
N_Q3WBC = round(np.percentile(NegativePatients['WBC'], 75, interpolation = 'midpoint'),1)
N_WBC = str(N_MedWBC) + ' [' + str(N_Q1WBC) + '-' + str(N_Q3WBC) + ']'

T_MedHematocrite = int(TotalPatients['Hematocrite'].median())
T_Q1Hematocrite = int(np.percentile(TotalPatients['Hematocrite'], 25, interpolation = 'midpoint'))
T_Q3Hematocrite = int(np.percentile(TotalPatients['Hematocrite'], 75, interpolation = 'midpoint'))
T_Hematocrite = str(T_MedHematocrite) + ' [' + str(T_Q1Hematocrite) + '-' + str(T_Q3Hematocrite) + ']'
P_MedHematocrite = int(PositivePatients['Hematocrite'].median())
P_Q1Hematocrite = int(np.percentile(PositivePatients['Hematocrite'], 25, interpolation = 'midpoint'))
P_Q3Hematocrite = int(np.percentile(PositivePatients['Hematocrite'], 75, interpolation = 'midpoint'))
P_Hematocrite = str(P_MedHematocrite) + ' [' + str(P_Q1Hematocrite) + '-' + str(P_Q3Hematocrite) + ']'
N_MedHematocrite = int(NegativePatients['Hematocrite'].median())
N_Q1Hematocrite = int(np.percentile(NegativePatients['Hematocrite'], 25, interpolation = 'midpoint'))
N_Q3Hematocrite = int(np.percentile(NegativePatients['Hematocrite'], 75, interpolation = 'midpoint'))
N_Hematocrite = str(N_MedHematocrite) + ' [' + str(N_Q1Hematocrite) + '-' + str(N_Q3Hematocrite) + ']'

T_MedESR = int(TotalPatients['ESR'].median())
T_Q1ESR = int(np.percentile(TotalPatients['ESR'], 25, interpolation = 'midpoint'))
T_Q3ESR = int(np.percentile(TotalPatients['ESR'], 75, interpolation = 'midpoint'))
T_ESR = str(T_MedESR) + ' [' + str(T_Q1ESR) + '-' + str(T_Q3ESR) + ']'
P_MedESR = int(PositivePatients['ESR'].median())
P_Q1ESR = int(np.percentile(PositivePatients['ESR'], 25, interpolation = 'midpoint'))
P_Q3ESR = int(np.percentile(PositivePatients['ESR'], 75, interpolation = 'midpoint'))
P_ESR = str(P_MedESR) + ' [' + str(P_Q1ESR) + '-' + str(P_Q3ESR) + ']'
N_MedESR = int(NegativePatients['ESR'].median())
N_Q1ESR = int(np.percentile(NegativePatients['ESR'], 25, interpolation = 'midpoint'))
N_Q3ESR = int(np.percentile(NegativePatients['ESR'], 75, interpolation = 'midpoint'))
N_ESR = str(N_MedESR) + ' [' + str(N_Q1ESR) + '-' + str(N_Q3ESR) + ']'

T_MedCRP = round(TotalPatients['CRP(mg/L)'].median(),1)
T_Q1CRP = round(np.percentile(TotalPatients['CRP(mg/L)'], 25, interpolation = 'midpoint'),1)
T_Q3CRP = round(np.percentile(TotalPatients['CRP(mg/L)'], 75, interpolation = 'midpoint'),1)
T_CRP = str(T_MedCRP) + ' [' + str(T_Q1CRP) + '-' + str(T_Q3CRP) + ']'
P_MedCRP = round(PositivePatients['CRP(mg/L)'].median(),1)
P_Q1CRP = round(np.percentile(PositivePatients['CRP(mg/L)'], 25, interpolation = 'midpoint'),1)
P_Q3CRP = round(np.percentile(PositivePatients['CRP(mg/L)'], 75, interpolation = 'midpoint'),1)
P_CRP = str(P_MedCRP) + ' [' + str(P_Q1CRP) + '-' + str(P_Q3CRP) + ']'
N_MedCRP = round(NegativePatients['CRP(mg/L)'].median(),1)
N_Q1CRP = round(np.percentile(NegativePatients['CRP(mg/L)'], 25, interpolation = 'midpoint'),1)
N_Q3CRP = round(np.percentile(NegativePatients['CRP(mg/L)'], 75, interpolation = 'midpoint'),1)
N_CRP = str(N_MedCRP) + ' [' + str(N_Q1CRP) + '-' + str(N_Q3CRP) + ']'

T_MedPCT = round(TotalPatients['PCT(ng/ml)'].median(),2)
T_Q1PCT = round(np.percentile(TotalPatients['PCT(ng/ml)'], 25, interpolation = 'midpoint'),2)
T_Q3PCT = round(np.percentile(TotalPatients['PCT(ng/ml)'], 75, interpolation = 'midpoint'),2)
T_PCT = str(T_MedPCT) + ' [' + str(T_Q1PCT) + '-' + str(T_Q3PCT) + ']'
P_MedPCT = round(PositivePatients['PCT(ng/ml)'].median(),2)
P_Q1PCT = round(np.percentile(PositivePatients['PCT(ng/ml)'], 25, interpolation = 'midpoint'),2)
P_Q3PCT = round(np.percentile(PositivePatients['PCT(ng/ml)'], 75, interpolation = 'midpoint'),2)
P_PCT = str(P_MedPCT) + ' [' + str(P_Q1PCT) + '-' + str(P_Q3PCT) + ']'
N_MedPCT = round(NegativePatients['PCT(ng/ml)'].median(),2)
N_Q1PCT = round(np.percentile(NegativePatients['PCT(ng/ml)'], 25, interpolation = 'midpoint'),2)
N_Q3PCT = round(np.percentile(NegativePatients['PCT(ng/ml)'], 75, interpolation = 'midpoint'),2)
N_PCT = str(N_MedPCT) + ' [' + str(N_Q1PCT) + '-' + str(N_Q3PCT) + ']'

T_MedPB = int(TotalPatients['Pitt Bacteremia score'].median())
T_Q1PB = int(np.percentile(TotalPatients['Pitt Bacteremia score'], 25, interpolation = 'midpoint'))
T_Q3PB = int(np.percentile(TotalPatients['Pitt Bacteremia score'], 75, interpolation = 'midpoint'))
T_PB = str(T_MedPB) + ' [' + str(T_Q1PB) + '-' + str(T_Q3PB) + ']'
P_MedPB = int(PositivePatients['Pitt Bacteremia score'].median())
P_Q1PB = int(np.percentile(PositivePatients['Pitt Bacteremia score'], 25, interpolation = 'midpoint'))
P_Q3PB = int(np.percentile(PositivePatients['Pitt Bacteremia score'], 75, interpolation = 'midpoint'))
P_PB = str(P_MedPB) + ' [' + str(P_Q1PB) + '-' + str(P_Q3PB) + ']'
N_MedPB = int(NegativePatients['Pitt Bacteremia score'].median())
N_Q1PB = int(np.percentile(NegativePatients['Pitt Bacteremia score'], 25, interpolation = 'midpoint'))
N_Q3PB = int(np.percentile(NegativePatients['Pitt Bacteremia score'], 75, interpolation = 'midpoint'))
N_PB = str(N_MedPB) + ' [' + str(N_Q1PB) + '-' + str(N_Q3PB) + ']'

T_MedAPACHE = int(TotalPatients['APACHE II score'].median())
T_Q1APACHE = int(np.percentile(TotalPatients['APACHE II score'], 25, interpolation = 'midpoint'))
T_Q3APACHE = int(np.percentile(TotalPatients['APACHE II score'], 75, interpolation = 'midpoint'))
T_APACHE = str(T_MedAPACHE) + ' [' + str(T_Q1APACHE) + '-' + str(T_Q3APACHE) + ']'
P_MedAPACHE = int(PositivePatients['APACHE II score'].median())
P_Q1APACHE = int(np.percentile(PositivePatients['APACHE II score'], 25, interpolation = 'midpoint'))
P_Q3APACHE = int(np.percentile(PositivePatients['APACHE II score'], 75, interpolation = 'midpoint'))
P_APACHE = str(P_MedAPACHE) + ' [' + str(P_Q1APACHE) + '-' + str(P_Q3APACHE) + ']'
N_MedAPACHE = int(NegativePatients['APACHE II score'].median())
N_Q1APACHE = int(np.percentile(NegativePatients['APACHE II score'], 25, interpolation = 'midpoint'))
N_Q3APACHE = int(np.percentile(NegativePatients['APACHE II score'], 75, interpolation = 'midpoint'))
N_APACHE = str(N_MedAPACHE) + ' [' + str(N_Q1APACHE) + '-' + str(N_Q3APACHE) + ']'

T_MedSOFA = int(TotalPatients['SOFA score'].median())
T_Q1SOFA = int(np.percentile(TotalPatients['SOFA score'], 25, interpolation = 'midpoint'))
T_Q3SOFA = int(np.percentile(TotalPatients['SOFA score'], 75, interpolation = 'midpoint'))
T_SOFA = str(T_MedSOFA) + ' [' + str(T_Q1SOFA) + '-' + str(T_Q3SOFA) + ']'
P_MedSOFA = int(PositivePatients['SOFA score'].median())
P_Q1SOFA = int(np.percentile(PositivePatients['SOFA score'], 25, interpolation = 'midpoint'))
P_Q3SOFA = int(np.percentile(PositivePatients['SOFA score'], 75, interpolation = 'midpoint'))
P_SOFA = str(P_MedSOFA) + ' [' + str(P_Q1SOFA) + '-' + str(P_Q3SOFA) + ']'
N_MedSOFA = int(NegativePatients['SOFA score'].median())
N_Q1SOFA = int(np.percentile(NegativePatients['SOFA score'], 25, interpolation = 'midpoint'))
N_Q3SOFA = int(np.percentile(NegativePatients['SOFA score'], 75, interpolation = 'midpoint'))
N_SOFA = str(N_MedSOFA) + ' [' + str(N_Q1SOFA) + '-' + str(N_Q3SOFA) + ']'

#Categorical Variables: Total counts [Proportion]
#Gender
T_GenderCounts = TotalPatients['Gender'].value_counts()[0]
T_GenderPercent = round(TotalPatients['Gender'].value_counts()[0] / (TotalPatients['Gender'].value_counts()[0]+TotalPatients['Gender'].value_counts()[1])*100,1)
T_Gender = str(T_GenderCounts) + ' [' + str(T_GenderPercent) + '%]'
P_GenderCounts = PositivePatients['Gender'].value_counts()[0]
P_GenderPercent = round(PositivePatients['Gender'].value_counts()[0] / (PositivePatients['Gender'].value_counts()[0]+PositivePatients['Gender'].value_counts()[1])*100,1)
P_Gender = str(P_GenderCounts) + ' [' + str(P_GenderPercent) + '%]'
N_GenderCounts = NegativePatients['Gender'].value_counts()[0]
N_GenderPercent = round(NegativePatients['Gender'].value_counts()[0] / (NegativePatients['Gender'].value_counts()[0]+NegativePatients['Gender'].value_counts()[1])*100,1)
N_Gender = str(N_GenderCounts) + ' [' + str(N_GenderPercent) + '%]'

#28-day Mortality Rate
T_MortalityCounts = TotalPatients['Death Status'].value_counts()[1]
T_MortalityPercent = round(TotalPatients['Death Status'].value_counts()[1] / (TotalPatients['Death Status'].value_counts()[0]+TotalPatients['Death Status'].value_counts()[1])*100,1)
T_Mortality = str(T_MortalityCounts) + ' [' + str(T_MortalityPercent) + '%]'
P_MortalityCounts = PositivePatients['Death Status'].value_counts()[1]
P_MortalityPercent = round(PositivePatients['Death Status'].value_counts()[1] / (PositivePatients['Death Status'].value_counts()[0]+PositivePatients['Death Status'].value_counts()[1])*100,1)
P_Mortality = str(P_MortalityCounts) + ' [' + str(P_MortalityPercent) + '%]'
N_MortalityCounts = NegativePatients['Death Status'].value_counts()[1]
N_MortalityPercent = round(NegativePatients['Death Status'].value_counts()[1] / (NegativePatients['Death Status'].value_counts()[0]+NegativePatients['Death Status'].value_counts()[1])*100,1)
N_Mortality = str(N_MortalityCounts) + ' [' + str(N_MortalityPercent) + '%]'

#Prior Antibiotics Status
T_AntibioticsCounts = TotalPatients['Prior Antibiotics'].value_counts()[1]
T_AntibioticsPercent = round(TotalPatients['Prior Antibiotics'].value_counts()[1] / (TotalPatients['Prior Antibiotics'].value_counts()[0]+TotalPatients['Prior Antibiotics'].value_counts()[1])*100,1)
T_Antibiotics = str(T_AntibioticsCounts) + ' [' + str(T_AntibioticsPercent) + '%]'
P_AntibioticsCounts = PositivePatients['Prior Antibiotics'].value_counts()[1]
P_AntibioticsPercent = round(PositivePatients['Prior Antibiotics'].value_counts()[1] / (PositivePatients['Prior Antibiotics'].value_counts()[0]+PositivePatients['Prior Antibiotics'].value_counts()[1])*100,1)
P_Antibiotics = str(P_AntibioticsCounts) + ' [' + str(P_AntibioticsPercent) + '%]'
N_AntibioticsCounts = NegativePatients['Prior Antibiotics'].value_counts()[1]
N_AntibioticsPercent = round(NegativePatients['Prior Antibiotics'].value_counts()[1] / (NegativePatients['Prior Antibiotics'].value_counts()[0]+NegativePatients['Prior Antibiotics'].value_counts()[1])*100,1)
N_Antibiotics = str(N_AntibioticsCounts) + ' [' + str(N_AntibioticsPercent) + '%]'

#DM : Diabetes Mellintus
T_DMCounts = TotalPatients['Diabetes Mellintus'].value_counts()[1]
T_DMPercent = round(TotalPatients['Diabetes Mellintus'].value_counts()[1] / (TotalPatients['Diabetes Mellintus'].value_counts()[0]+TotalPatients['Diabetes Mellintus'].value_counts()[1])*100,1)
T_DM = str(T_DMCounts) + ' [' + str(T_DMPercent) + '%]'
P_DMCounts = PositivePatients['Diabetes Mellintus'].value_counts()[1]
P_DMPercent = round(PositivePatients['Diabetes Mellintus'].value_counts()[1] / (PositivePatients['Diabetes Mellintus'].value_counts()[0]+PositivePatients['Diabetes Mellintus'].value_counts()[1])*100,1)
P_DM = str(P_DMCounts) + ' [' + str(P_DMPercent) + '%]'
N_DMCounts = NegativePatients['Diabetes Mellintus'].value_counts()[1]
N_DMPercent = round(NegativePatients['Diabetes Mellintus'].value_counts()[1] / (NegativePatients['Diabetes Mellintus'].value_counts()[0]+NegativePatients['Diabetes Mellintus'].value_counts()[1])*100,1)
N_DM = str(N_DMCounts) + ' [' + str(N_DMPercent) + '%]'

#HTN : Hypertension
T_HTNCounts = TotalPatients['Hypertension'].value_counts()[1]
T_HTNPercent = round(TotalPatients['Hypertension'].value_counts()[1] / (TotalPatients['Hypertension'].value_counts()[0]+TotalPatients['Hypertension'].value_counts()[1])*100,1)
T_HTN = str(T_HTNCounts) + ' [' + str(T_HTNPercent) + '%]'
P_HTNCounts = PositivePatients['Hypertension'].value_counts()[1]
P_HTNPercent = round(PositivePatients['Hypertension'].value_counts()[1] / (PositivePatients['Hypertension'].value_counts()[0]+PositivePatients['Hypertension'].value_counts()[1])*100,1)
P_HTN = str(P_HTNCounts) + ' [' + str(P_HTNPercent) + '%]'
N_HTNCounts = NegativePatients['Hypertension'].value_counts()[0]
N_HTNPercent = round(NegativePatients['Hypertension'].value_counts()[0] / (NegativePatients['Hypertension'].value_counts()[0]+NegativePatients['Hypertension'].value_counts()[1])*100,1)
N_HTN = str(N_HTNCounts) + ' [' + str(N_HTNPercent) + '%]'

#HF : Heart Failure
T_HFCounts = TotalPatients['Heart Failure'].value_counts()[1]
T_HFPercent = round(TotalPatients['Heart Failure'].value_counts()[1] / (TotalPatients['Heart Failure'].value_counts()[0]+TotalPatients['Heart Failure'].value_counts()[1])*100,1)
T_HF = str(T_HFCounts) + ' [' + str(T_HFPercent) + '%]'
P_HFCounts = PositivePatients['Heart Failure'].value_counts()[1]
P_HFPercent = round(PositivePatients['Heart Failure'].value_counts()[1] / (PositivePatients['Heart Failure'].value_counts()[0]+PositivePatients['Heart Failure'].value_counts()[1])*100,1)
P_HF = str(P_HFCounts) + ' [' + str(P_HFPercent) + '%]'
N_HFCounts = NegativePatients['Heart Failure'].value_counts()[1]
N_HFPercent = round(NegativePatients['Heart Failure'].value_counts()[1] / (NegativePatients['Heart Failure'].value_counts()[0]+NegativePatients['Heart Failure'].value_counts()[1])*100,1)
N_HF = str(N_HFCounts) + ' [' + str(N_HFPercent) + '%]'

#CEVD : Cerebrovascular Disease
T_CEVDCounts = TotalPatients['Cerebrovascular Disease'].value_counts()[1]
T_CEVDPercent = round(TotalPatients['Cerebrovascular Disease'].value_counts()[1] / (TotalPatients['Cerebrovascular Disease'].value_counts()[0]+TotalPatients['Cerebrovascular Disease'].value_counts()[1])*100,1)
T_CEVD = str(T_CEVDCounts) + ' [' + str(T_CEVDPercent) + '%]'
P_CEVDCounts = PositivePatients['Cerebrovascular Disease'].value_counts()[1]
P_CEVDPercent = round(PositivePatients['Cerebrovascular Disease'].value_counts()[1] / (PositivePatients['Cerebrovascular Disease'].value_counts()[0]+PositivePatients['Cerebrovascular Disease'].value_counts()[1])*100,1)
P_CEVD = str(P_CEVDCounts) + ' [' + str(P_CEVDPercent) + '%]'
N_CEVDCounts = NegativePatients['Cerebrovascular Disease'].value_counts()[1]
N_CEVDPercent = round(NegativePatients['Cerebrovascular Disease'].value_counts()[1] / (NegativePatients['Cerebrovascular Disease'].value_counts()[0]+NegativePatients['Cerebrovascular Disease'].value_counts()[1])*100,1)
N_CEVD = str(N_CEVDCounts) + ' [' + str(N_CEVDPercent) + '%]'

#RD : Renal Disease
T_RDCounts = TotalPatients['Renal Disease'].value_counts()[1]
T_RDPercent = round(TotalPatients['Renal Disease'].value_counts()[1] / (TotalPatients['Renal Disease'].value_counts()[0]+TotalPatients['Renal Disease'].value_counts()[1])*100,1)
T_RD = str(T_RDCounts) + ' [' + str(T_RDPercent) + '%]'
P_RDCounts = PositivePatients['Renal Disease'].value_counts()[1]
P_RDPercent = round(PositivePatients['Renal Disease'].value_counts()[1] / (PositivePatients['Renal Disease'].value_counts()[0]+PositivePatients['Renal Disease'].value_counts()[1])*100,1)
P_RD = str(P_RDCounts) + ' [' + str(P_RDPercent) + '%]'
N_RDCounts = NegativePatients['Renal Disease'].value_counts()[1]
N_RDPercent = round(NegativePatients['Renal Disease'].value_counts()[1] / (NegativePatients['Renal Disease'].value_counts()[0]+NegativePatients['Renal Disease'].value_counts()[1])*100,1)
N_RD = str(N_RDCounts) + ' [' + str(N_RDPercent) + '%]'

#LD : Liver Disease
T_LDCounts = TotalPatients['Liver Disease'].value_counts()[1]
T_LDPercent = round(TotalPatients['Liver Disease'].value_counts()[1] / (TotalPatients['Liver Disease'].value_counts()[0]+TotalPatients['Liver Disease'].value_counts()[1])*100,1)
T_LD = str(T_LDCounts) + ' [' + str(T_LDPercent) + '%]'
P_LDCounts = PositivePatients['Liver Disease'].value_counts()[1]
P_LDPercent = round(PositivePatients['Liver Disease'].value_counts()[1] / (PositivePatients['Liver Disease'].value_counts()[0]+PositivePatients['Liver Disease'].value_counts()[1])*100,1)
P_LD = str(P_LDCounts) + ' [' + str(P_LDPercent) + '%]'
N_LDCounts = NegativePatients['Liver Disease'].value_counts()[1]
N_LDPercent = round(NegativePatients['Liver Disease'].value_counts()[1] / (NegativePatients['Liver Disease'].value_counts()[0]+NegativePatients['Liver Disease'].value_counts()[1])*100,1)
N_LD = str(N_LDCounts) + ' [' + str(N_LDPercent) + '%]'

#COPD 
T_COPDCounts = TotalPatients['COPD'].value_counts()[1]
T_COPDPercent = round(TotalPatients['COPD'].value_counts()[1] / (TotalPatients['COPD'].value_counts()[0]+TotalPatients['COPD'].value_counts()[1])*100,1)
T_COPD = str(T_COPDCounts) + ' [' + str(T_COPDPercent) + '%]'
P_COPDCounts = PositivePatients['COPD'].value_counts()[1]
P_COPDPercent = round(PositivePatients['COPD'].value_counts()[1] / (PositivePatients['COPD'].value_counts()[0]+PositivePatients['COPD'].value_counts()[1])*100,1)
P_COPD = str(P_COPDCounts) + ' [' + str(P_COPDPercent) + '%]'
N_COPDCounts = NegativePatients['COPD'].value_counts()[1]
N_COPDPercent = round(NegativePatients['COPD'].value_counts()[1] / (NegativePatients['COPD'].value_counts()[0]+NegativePatients['COPD'].value_counts()[1])*100,1)
N_COPD = str(N_COPDCounts) + ' [' + str(N_COPDPercent) + '%]'

#KN : Known Neoplasm
T_KNCounts = TotalPatients['Known Neoplasm'].value_counts()[1]
T_KNPercent = round(TotalPatients['Known Neoplasm'].value_counts()[1] / (TotalPatients['Known Neoplasm'].value_counts()[0]+TotalPatients['Known Neoplasm'].value_counts()[1])*100,1)
T_KN = str(T_KNCounts) + ' [' + str(T_KNPercent) + '%]'
P_KNCounts = PositivePatients['Known Neoplasm'].value_counts()[1]
P_KNPercent = round(PositivePatients['Known Neoplasm'].value_counts()[1] / (PositivePatients['Known Neoplasm'].value_counts()[0]+PositivePatients['Known Neoplasm'].value_counts()[1])*100,1)
P_KN = str(P_KNCounts) + ' [' + str(P_KNPercent) + '%]'
N_KNCounts = NegativePatients['Known Neoplasm'].value_counts()[1]
N_KNPercent = round(NegativePatients['Known Neoplasm'].value_counts()[1] / (NegativePatients['Known Neoplasm'].value_counts()[0]+NegativePatients['Known Neoplasm'].value_counts()[1])*100,1)
N_KN = str(N_KNCounts) + ' [' + str(N_KNPercent) + '%]'

#CRBI : Catheter Related Bloodstream Infection
T_CRBICounts = TotalPatients['Catheter Related Bloodstream Infection'].value_counts()[1]
T_CRBIPercent = round(TotalPatients['Catheter Related Bloodstream Infection'].value_counts()[1] / (TotalPatients['Catheter Related Bloodstream Infection'].value_counts()[0]+TotalPatients['Catheter Related Bloodstream Infection'].value_counts()[1])*100,1)
T_CRBI = str(T_CRBICounts) + ' [' + str(T_CRBIPercent) + '%]'
P_CRBICounts = PositivePatients['Catheter Related Bloodstream Infection'].value_counts()[1]
P_CRBIPercent = round(PositivePatients['Catheter Related Bloodstream Infection'].value_counts()[1] / (PositivePatients['Catheter Related Bloodstream Infection'].value_counts()[0]+PositivePatients['Catheter Related Bloodstream Infection'].value_counts()[1])*100,1)
P_CRBI = str(P_CRBICounts) + ' [' + str(P_CRBIPercent) + '%]'
N_CRBICounts = NegativePatients['Catheter Related Bloodstream Infection'].value_counts()[1]
N_CRBIPercent = round(NegativePatients['Catheter Related Bloodstream Infection'].value_counts()[1] / (NegativePatients['Catheter Related Bloodstream Infection'].value_counts()[0]+NegativePatients['Catheter Related Bloodstream Infection'].value_counts()[1])*100,1)
N_CRBI = str(N_CRBICounts) + ' [' + str(N_CRBIPercent) + '%]'

#IAI : Intra Abdominal Infection
T_IAICounts = TotalPatients['Intra Abdominal Infection'].value_counts()[1]
T_IAIPercent = round(TotalPatients['Intra Abdominal Infection'].value_counts()[1] / (TotalPatients['Intra Abdominal Infection'].value_counts()[0]+TotalPatients['Intra Abdominal Infection'].value_counts()[1])*100,1)
T_IAI = str(T_IAICounts) + ' [' + str(T_IAIPercent) + '%]'
P_IAICounts = PositivePatients['Intra Abdominal Infection'].value_counts()[1]
P_IAIPercent = round(PositivePatients['Intra Abdominal Infection'].value_counts()[1] / (PositivePatients['Intra Abdominal Infection'].value_counts()[0]+PositivePatients['Intra Abdominal Infection'].value_counts()[1])*100,1)
P_IAI = str(P_IAICounts) + ' [' + str(P_IAIPercent) + '%]'
N_IAICounts = NegativePatients['Intra Abdominal Infection'].value_counts()[1]
N_IAIPercent = round(NegativePatients['Intra Abdominal Infection'].value_counts()[1] / (NegativePatients['Intra Abdominal Infection'].value_counts()[0]+NegativePatients['Intra Abdominal Infection'].value_counts()[1])*100,1)
N_IAI = str(N_IAICounts) + ' [' + str(N_IAIPercent) + '%]'

#RTI : Respiratory Tract Infection *
T_RTICounts = TotalPatients['Respiratory Tract Infection'].value_counts()[0]
T_RTIPercent = round(TotalPatients['Respiratory Tract Infection'].value_counts()[0] / (TotalPatients['Respiratory Tract Infection'].value_counts()[0]+TotalPatients['Respiratory Tract Infection'].value_counts()[1])*100,1)
T_RTI = str(T_RTICounts) + ' [' + str(T_RTIPercent) + '%]'
P_RTICounts = PositivePatients['Respiratory Tract Infection'].value_counts()[1]
P_RTIPercent = round(PositivePatients['Respiratory Tract Infection'].value_counts()[1] / (PositivePatients['Respiratory Tract Infection'].value_counts()[0]+PositivePatients['Respiratory Tract Infection'].value_counts()[1])*100,1)
P_RTI = str(P_RTICounts) + ' [' + str(P_RTIPercent) + '%]'
N_RTICounts = NegativePatients['Respiratory Tract Infection'].value_counts()[0]
N_RTIPercent = round(NegativePatients['Respiratory Tract Infection'].value_counts()[0] / (NegativePatients['Respiratory Tract Infection'].value_counts()[0]+NegativePatients['Respiratory Tract Infection'].value_counts()[1])*100,1)
N_RTI = str(N_RTICounts) + ' [' + str(N_RTIPercent) + '%]'

#SSTI: Skin and Soft Tissue Infection
T_SSTICounts = TotalPatients['Skin and Soft Tissue Infection'].value_counts()[1]
T_SSTIPercent = round(TotalPatients['Skin and Soft Tissue Infection'].value_counts()[1] / (TotalPatients['Skin and Soft Tissue Infection'].value_counts()[0]+TotalPatients['Skin and Soft Tissue Infection'].value_counts()[1])*100,1)
T_SSTI = str(T_SSTICounts) + ' [' + str(T_SSTIPercent) + '%]'
P_SSTICounts = PositivePatients['Skin and Soft Tissue Infection'].value_counts()[1]
P_SSTIPercent = round(PositivePatients['Skin and Soft Tissue Infection'].value_counts()[1] / (PositivePatients['Skin and Soft Tissue Infection'].value_counts()[0]+PositivePatients['Skin and Soft Tissue Infection'].value_counts()[1])*100,1)
P_SSTI = str(P_SSTICounts) + ' [' + str(P_SSTIPercent) + '%]'
N_SSTICounts = NegativePatients['Skin and Soft Tissue Infection'].value_counts()[1]
N_SSTIPercent = round(NegativePatients['Skin and Soft Tissue Infection'].value_counts()[1] / (NegativePatients['Skin and Soft Tissue Infection'].value_counts()[0]+NegativePatients['Skin and Soft Tissue Infection'].value_counts()[1])*100,1)
N_SSTI = str(N_SSTICounts) + ' [' + str(N_SSTIPercent) + '%]'

#UTI: Urinary Tract Infection
T_UTICounts = TotalPatients['Urinary Tract Infection'].value_counts()[1]
T_UTIPercent = round(TotalPatients['Urinary Tract Infection'].value_counts()[1] / (TotalPatients['Urinary Tract Infection'].value_counts()[0]+TotalPatients['Urinary Tract Infection'].value_counts()[1])*100,1)
T_UTI = str(T_UTICounts) + ' [' + str(T_UTIPercent) + '%]'
P_UTICounts = PositivePatients['Urinary Tract Infection'].value_counts()[1]
P_UTIPercent = round(PositivePatients['Urinary Tract Infection'].value_counts()[1] / (PositivePatients['Urinary Tract Infection'].value_counts()[0]+PositivePatients['Urinary Tract Infection'].value_counts()[1])*100,1)
P_UTI = str(P_UTICounts) + ' [' + str(P_UTIPercent) + '%]'
N_UTICounts = NegativePatients['Urinary Tract Infection'].value_counts()[1]
N_UTIPercent = round(NegativePatients['Urinary Tract Infection'].value_counts()[1] / (NegativePatients['Urinary Tract Infection'].value_counts()[0]+NegativePatients['Urinary Tract Infection'].value_counts()[1])*100,1)
N_UTI = str(N_UTICounts) + ' [' + str(N_UTIPercent) + '%]'

#Others: Others
T_OthersCounts = TotalPatients['Others'].value_counts()[1]
T_OthersPercent = round(TotalPatients['Others'].value_counts()[1] / (TotalPatients['Others'].value_counts()[0]+TotalPatients['Others'].value_counts()[1])*100,1)
T_Others = str(T_OthersCounts) + ' [' + str(T_OthersPercent) + '%]'
P_OthersCounts = PositivePatients['Others'].value_counts()[1]
P_OthersPercent = round(PositivePatients['Others'].value_counts()[1] / (PositivePatients['Others'].value_counts()[0]+PositivePatients['Others'].value_counts()[1])*100,1)
P_Others = str(P_OthersCounts) + ' [' + str(P_OthersPercent) + '%]'
N_OthersCounts = NegativePatients['Others'].value_counts()[1]
N_OthersPercent = round(NegativePatients['Others'].value_counts()[1] / (NegativePatients['Others'].value_counts()[0]+NegativePatients['Others'].value_counts()[1])*100,1)
N_Others = str(N_OthersCounts) + ' [' + str(N_OthersPercent) + '%]'

#FOU : Fever of Unknown Origin
T_FOUCounts = TotalPatients['Fever of Unknown Origin'].value_counts()[1]
T_FOUPercent = round(TotalPatients['Fever of Unknown Origin'].value_counts()[1] / (TotalPatients['Fever of Unknown Origin'].value_counts()[0]+TotalPatients['Fever of Unknown Origin'].value_counts()[1])*100,1)
T_FOU = str(T_FOUCounts) + ' [' + str(T_FOUPercent) + '%]'
P_FOUCounts = PositivePatients['Fever of Unknown Origin'].value_counts()[1]
P_FOUPercent = round(PositivePatients['Fever of Unknown Origin'].value_counts()[1] / (PositivePatients['Fever of Unknown Origin'].value_counts()[0]+PositivePatients['Fever of Unknown Origin'].value_counts()[1])*100,1)
P_FOU = str(P_FOUCounts) + ' [' + str(P_FOUPercent) + '%]'
N_FOUCounts = NegativePatients['Fever of Unknown Origin'].value_counts()[1]
N_FOUPercent = round(NegativePatients['Fever of Unknown Origin'].value_counts()[1] / (NegativePatients['Fever of Unknown Origin'].value_counts()[0]+NegativePatients['Fever of Unknown Origin'].value_counts()[1])*100,1)
N_FOU = str(N_FOUCounts) + ' [' + str(N_FOUPercent) + '%]'

#________________Running T-test, MannWhitney Test, Chi-square Test_______________________________________________________
#Running a T-test/Mann-Whitney U test to identify the statistical significance between Bacteremia Positive and Negative patients
#tTest: 0.384, uTest:0.335
tTestResultAge = stats.ttest_ind(PositivePatients['Age'], NegativePatients['Age'], equal_var=False)
UTestResultAge = mannwhitneyu(PositivePatients['Age'], NegativePatients['Age'], alternative='two-sided')
U_Age = round(UTestResultAge[1],3)

#tTest: 0.0016, uTest:0.0020
tTestResultVolume = stats.ttest_ind(PositivePatients['Volume'], NegativePatients['Volume'], equal_var=False)
UTestResultVolume = mannwhitneyu(PositivePatients['Volume'], NegativePatients['Volume'], alternative='two-sided')
U_Volume = round(UTestResultVolume[1],3)

#tTest: 0.042, uTest:0.061
tTestResultBT = stats.ttest_ind(PositivePatients['Body Temperature'], NegativePatients['Body Temperature'], equal_var=False)
UTestResultBT = mannwhitneyu(PositivePatients['Body Temperature'], NegativePatients['Body Temperature'], alternative='two-sided')
U_BT = round(UTestResultBT[1],3)

#tTest: 0.833, uTest:0.827
tTestResultBW = stats.ttest_ind(PositivePatients['Body Weight'], NegativePatients['Body Weight'], equal_var=False)
UTestResultBW = mannwhitneyu(PositivePatients['Body Weight'], NegativePatients['Body Weight'], alternative='two-sided')
U_BW = round(UTestResultBW[1],3)

#tTest: 0.803, uTest:0.479
tTestResultPaO2 = stats.ttest_ind(PositivePatients['PaO2'], NegativePatients['PaO2'], equal_var=False)
UTestResultPaO2 = mannwhitneyu(PositivePatients['PaO2'], NegativePatients['PaO2'], alternative='two-sided')
U_PaO2 = round(UTestResultPaO2[1],3)

#tTest: 0.505, uTest:0.358
tTestResultFiO2 = stats.ttest_ind(PositivePatients['FiO2'], NegativePatients['FiO2'], equal_var=False)
UTestResultFiO2 = mannwhitneyu(PositivePatients['FiO2'], NegativePatients['FiO2'], alternative='two-sided')
U_FiO2 = round(UTestResultFiO2[1],3)

#tTest: 0.358, uTest:0.515
tTestResultPaO2FiO2 = stats.ttest_ind(PositivePatients['PaO2/FiO2'], NegativePatients['PaO2/FiO2'], equal_var=False)
UTestResultPaO2FiO2 = mannwhitneyu(PositivePatients['PaO2/FiO2'], NegativePatients['PaO2/FiO2'], alternative='two-sided')
U_PaO2FiO2 = round(UTestResultPaO2FiO2[1],3)

#tTest: 0.259, uTest:0.294
tTestResultRR = stats.ttest_ind(PositivePatients['Respiratory Rate'], NegativePatients['Respiratory Rate'], equal_var=False)
UTestResultRR = mannwhitneyu(PositivePatients['Respiratory Rate'], NegativePatients['Respiratory Rate'], alternative='two-sided')
U_RR = round(UTestResultRR[1],3)

#tTest: 0.208, uTest:0.092
tTestResultHR = stats.ttest_ind(PositivePatients['Heart Rate'], NegativePatients['Heart Rate'], equal_var=False)
UTestResultHR = mannwhitneyu(PositivePatients['Heart Rate'], NegativePatients['Heart Rate'], alternative='two-sided')
U_HR = round(UTestResultHR[1],3)

#uTest:0.008
tTestResultSBP = stats.ttest_ind(PositivePatients['SBP'], NegativePatients['SBP'], equal_var=False)
UTestResultSBP = mannwhitneyu(PositivePatients['SBP'], NegativePatients['SBP'], alternative='two-sided')
U_SBP = round(UTestResultSBP[1],3)

#uTest:0.0003
tTestResultDBP = stats.ttest_ind(PositivePatients['DBP'], NegativePatients['DBP'], equal_var=False)
UTestResultDBP = mannwhitneyu(PositivePatients['DBP'], NegativePatients['DBP'], alternative='two-sided')
U_DBP = round(UTestResultDBP[1],4)

#tTest: 0.160, uTest:0.130
tTestResultMAP = stats.ttest_ind(PositivePatients['MAP'], NegativePatients['MAP'], equal_var=False)
UTestResultMAP = mannwhitneyu(PositivePatients['MAP'], NegativePatients['MAP'], alternative='two-sided')
U_MAP = round(UTestResultMAP[1],3)

#tTest: 0.243 , uTest:0.082
tTestResultPH = stats.ttest_ind(PositivePatients['Arterial pH'], NegativePatients['Arterial pH'], equal_var=False)
UTestResultPH = mannwhitneyu(PositivePatients['Arterial pH'], NegativePatients['Arterial pH'], alternative='two-sided')
U_PH = round(UTestResultPH[1],3)

#tTest: 0.426 , uTest:0.354
tTestResultK = stats.ttest_ind(PositivePatients['K'], NegativePatients['K'], equal_var=False)
UTestResultK = mannwhitneyu(PositivePatients['K'], NegativePatients['K'], alternative='two-sided')
U_K = round(UTestResultK[1],3)

#tTest: 0.0057 , uTest:0.000052
tTestResultNa = stats.ttest_ind(PositivePatients['Na'], NegativePatients['Na'], equal_var=False)
UTestResultNa = mannwhitneyu(PositivePatients['Na'], NegativePatients['Na'], alternative='two-sided')
U_NA = round(UTestResultNa[1],4)

#tTest: 0.00569 , uTest:0.000157
tTestResultGCS = stats.ttest_ind(PositivePatients['GCS Score'], NegativePatients['GCS Score'], equal_var=False)
UTestResultGCS = mannwhitneyu(PositivePatients['GCS Score'], NegativePatients['GCS Score'], alternative='two-sided')
U_GCS = round(UTestResultGCS[1],3)

#tTest: 0.027 , uTest:0.000794
tTestResultLymphocyte = stats.ttest_ind(PositivePatients['Lymphocyte'], NegativePatients['Lymphocyte'], equal_var=False)
UTestResultLymphocyte = mannwhitneyu(PositivePatients['Lymphocyte'], NegativePatients['Lymphocyte'], alternative='two-sided')
U_Lymphocyte = round(UTestResultLymphocyte[1],6)

#tTest: 0.00462 , uTest: 0.00000111
tTestResultCreatinine = stats.ttest_ind(PositivePatients['Creatinine'], NegativePatients['Creatinine'], equal_var=False)
UTestResultCreatinine = mannwhitneyu(PositivePatients['Creatinine'], NegativePatients['Creatinine'], alternative='two-sided')
U_Creatinine = round(UTestResultCreatinine[1],6)

#tTest: 0.00506 , uTest: 0.000642
tTestResultLactate = stats.ttest_ind(PositivePatients['Lactate Level'], NegativePatients['Lactate Level'], equal_var=False)
UTestResultLactate = mannwhitneyu(PositivePatients['Lactate Level'], NegativePatients['Lactate Level'], alternative='two-sided')
U_Lactate = round(UTestResultLactate[1],3)

#tTest: 0.009 , uTest: 0.000342
tTestResultBilirubin = stats.ttest_ind(PositivePatients['Bilirubin'], NegativePatients['Bilirubin'], equal_var=False)
UTestResultBilirubin = mannwhitneyu(PositivePatients['Bilirubin'], NegativePatients['Bilirubin'], alternative='two-sided')
U_Bilirubin = round(UTestResultBilirubin[1],8)

#tTest: 0.0905 , uTest: 0.004
tTestResultAST = stats.ttest_ind(PositivePatients['AST'], NegativePatients['AST'], equal_var=False)
UTestResultAST = mannwhitneyu(PositivePatients['AST'], NegativePatients['AST'], alternative='two-sided')
U_AST = round(UTestResultAST[1],4)

#tTest: 0.0319 , uTest: 0.00334
tTestResultPLT = stats.ttest_ind(PositivePatients['Platelets'], NegativePatients['Platelets'], equal_var=False)
UTestResultPLT = mannwhitneyu(PositivePatients['Platelets'], NegativePatients['Platelets'], alternative='two-sided')
U_PLT = round(UTestResultPLT[1],6)

#tTest: 0.0005 , uTest: 0.00000836
tTestResultBUN = stats.ttest_ind(PositivePatients['BUN'], NegativePatients['BUN'], equal_var=False)
UTestResultBUN = mannwhitneyu(PositivePatients['BUN'], NegativePatients['BUN'], alternative='two-sided')
U_BUN = round(UTestResultBUN[1],8)

#tTest: 0.242 , uTest: 0.285
tTestResultWBC = stats.ttest_ind(PositivePatients['WBC'], NegativePatients['WBC'], equal_var=False)
UTestResultWBC = mannwhitneyu(PositivePatients['WBC'], NegativePatients['WBC'], alternative='two-sided')
U_WBC = round(UTestResultWBC[1],3)

#tTest: 0., uTest: 0.
tTestResultHematocrite = stats.ttest_ind(PositivePatients['Hematocrite'], NegativePatients['Hematocrite'], equal_var=False)
UTestResultHematocrite = mannwhitneyu(PositivePatients['Hematocrite'], NegativePatients['Hematocrite'], alternative='two-sided')
U_Hematocrite = round(UTestResultHematocrite[1],3)

#tTest: 0.0346 , uTest: 0.0699
tTestResultESR = stats.ttest_ind(PositivePatients['ESR'], NegativePatients['ESR'], equal_var=False)
UTestResultESR = mannwhitneyu(PositivePatients['ESR'], NegativePatients['ESR'], alternative='two-sided')
U_ESR = round(UTestResultESR[1],3)

#tTest: 0.00000268 , uTest: 0.00000265
tTestResultCRP = stats.ttest_ind(PositivePatients['CRP(mg/L)'], NegativePatients['CRP(mg/L)'], equal_var=False)
UTestResultCRP = mannwhitneyu(PositivePatients['CRP(mg/L)'], NegativePatients['CRP(mg/L)'], alternative='two-sided')
U_CRP = round(UTestResultCRP[1],10)

#tTest: 0.00000457 , uTest: 0.00000000037
tTestResultPCT = stats.ttest_ind(PositivePatients['PCT(ng/ml)'], NegativePatients['PCT(ng/ml)'], equal_var=False)
UTestResultPCT = mannwhitneyu(PositivePatients['PCT(ng/ml)'], NegativePatients['PCT(ng/ml)'], alternative='two-sided')
U_PCT = round(UTestResultPCT[1],15)

#tTest: 0. , uTest: 0.
tTestResultPB = stats.ttest_ind(PositivePatients['Pitt Bacteremia score'], NegativePatients['Pitt Bacteremia score'], equal_var=False)
UTestResultPB = mannwhitneyu(PositivePatients['Pitt Bacteremia score'], NegativePatients['Pitt Bacteremia score'], alternative='two-sided')
U_PB = round(UTestResultPB[1],3)

#tTest: 0. , uTest: 0.
tTestResultAPACHE = stats.ttest_ind(PositivePatients['APACHE II score'], NegativePatients['APACHE II score'], equal_var=False)
UTestResultAPACHE = mannwhitneyu(PositivePatients['APACHE II score'], NegativePatients['APACHE II score'], alternative='two-sided')
U_APACHE = round(UTestResultAPACHE[1],6)

#tTest: 0.00000265 , uTest: 0.000000257
tTestResultSOFA = stats.ttest_ind(PositivePatients['SOFA score'], NegativePatients['SOFA score'], equal_var=False)
UTestResultSOFA = mannwhitneyu(PositivePatients['SOFA score'], NegativePatients['SOFA score'], alternative='two-sided')
U_SOFA = round(UTestResultSOFA[1],9)

#tTest: 0.0796 , uTest: 0.0161
tTestResultSegNeutrophil = stats.ttest_ind(PositivePatients['Seg Neutrophil'], NegativePatients['Seg Neutrophil'], equal_var=False)
UTestResultSegNeutrophil = mannwhitneyu(PositivePatients['Seg Neutrophil'], NegativePatients['Seg Neutrophil'], alternative='two-sided')
U_SegNeutrophil = round(UTestResultSegNeutrophil[1],3)

#tTest: 0.107 , uTest: 0.149
tTestResultBandsNeutrophil = stats.ttest_ind(PositivePatients['Bands Neutrophil'], NegativePatients['Bands Neutrophil'], equal_var=False)
UTestResultBandsNeutrophil = mannwhitneyu(PositivePatients['Bands Neutrophil'], NegativePatients['Bands Neutrophil'], alternative='two-sided')
U_BandsNeutrophil = round(UTestResultBandsNeutrophil[1],3)

#tTest: 0.055 , uTest: 0.064
tTestResultANC = stats.ttest_ind(PositivePatients['Absolute Neutrophil Count'], NegativePatients['Absolute Neutrophil Count'], equal_var=False)
UTestResultANC = mannwhitneyu(PositivePatients['Absolute Neutrophil Count'], NegativePatients['Absolute Neutrophil Count'], alternative='two-sided')
U_ANC = round(UTestResultANC[1],3)

#tTest: 0.0039 , uTest: 0.0027
tTestResultALC = stats.ttest_ind(PositivePatients['Absolute Lymphocyte Count'], NegativePatients['Absolute Lymphocyte Count'], equal_var=False)
UTestResultALC = mannwhitneyu(PositivePatients['Absolute Lymphocyte Count'], NegativePatients['Absolute Lymphocyte Count'], alternative='two-sided')
U_ALC = round(UTestResultALC[1],6)

#tTest: 0.00026, uTest: 0.000238
tTestResultNLR = stats.ttest_ind(PositivePatients['Neutrophil Lymphocyte Ratio'], NegativePatients['Neutrophil Lymphocyte Ratio'], equal_var=False)
UTestResultNLR = mannwhitneyu(PositivePatients['Neutrophil Lymphocyte Ratio'], NegativePatients['Neutrophil Lymphocyte Ratio'], alternative='two-sided')
U_NLR = round(UTestResultNLR[1],6)

#Running a Chi-square test to identify the statistical significance between Bacteremia Positive and Negative patients
A0, B0 = PositivePatients['Gender'].value_counts()
C0, D0 = NegativePatients['Gender'].value_counts()
ChiTable0 = [[A0,B0],[C0,D0]]
chiResultGender = chi2_contingency(ChiTable0)[1]
C_Gender = round(chiResultGender,3)

A1, B1 = PositivePatients['Death Status'].value_counts()
C1, D1 = NegativePatients['Death Status'].value_counts()
ChiTable1 = [[A1,B1],[C1,D1]]
chiResultMortality = chi2_contingency(ChiTable1)[1]
C_Mortality = round(chiResultMortality,3)

A2, B2 = PositivePatients['Prior Antibiotics'].value_counts()
C2, D2 = NegativePatients['Prior Antibiotics'].value_counts()
ChiTable2 = [[A2,B2],[C2,D2]]
chiResultAntibiotics = chi2_contingency(ChiTable2)[1]
C_Antibiotics = round(chiResultAntibiotics,3)

A3, B3 = PositivePatients['Diabetes Mellintus'].value_counts()
C3, D3 = NegativePatients['Diabetes Mellintus'].value_counts()
ChiTable3 = [[A3,B3],[C3,D3]]
chiResultDM = chi2_contingency(ChiTable3)[1]
C_DM = round(chiResultDM,3)

A4, B4 = PositivePatients['Hypertension'].value_counts()
C4, D4 = NegativePatients['Hypertension'].value_counts()
ChiTable4 = [[A4,B4],[C4,D4]]
chiResultHTN = chi2_contingency(ChiTable4)[1]
C_HTN = round(chiResultHTN,3)

A5, B5 = PositivePatients['Heart Failure'].value_counts()
C5, D5 = NegativePatients['Heart Failure'].value_counts()
ChiTable5 = [[A5,B5],[C5,D5]]
chiResultHF = chi2_contingency(ChiTable5)[1]
C_HF = round(chiResultHF,3)

A6, B6 = PositivePatients['Cerebrovascular Disease'].value_counts()
C6, D6 = NegativePatients['Cerebrovascular Disease'].value_counts()
ChiTable6 = [[A6,B6],[C6,D6]]
chiResultCEVD = chi2_contingency(ChiTable6)[1]
C_CEVD = round(chiResultCEVD,3)

A7, B7 = PositivePatients['Renal Disease'].value_counts()
C7, D7 = NegativePatients['Renal Disease'].value_counts()
ChiTable7 = [[A7,B7],[C7,D7]]
chiResultRD = chi2_contingency(ChiTable7)[1]
C_RD = round(chiResultRD,3)

A9, B9 = PositivePatients['Liver Disease'].value_counts()
C9, D9 = NegativePatients['Liver Disease'].value_counts()
ChiTable9 = [[A9,B9],[C9,D9]]
chiResultLD = chi2_contingency(ChiTable9)[1]
C_LD = round(chiResultLD,3)

A10, B10 = PositivePatients['COPD'].value_counts()
C10, D10 = NegativePatients['COPD'].value_counts()
ChiTable10 = [[A10,B10],[C10,D10]]
chiResultCOPD = chi2_contingency(ChiTable10)[1]
C_COPD = round(chiResultCOPD,3)

A11, B11 = PositivePatients['Known Neoplasm'].value_counts()
C11, D11 = NegativePatients['Known Neoplasm'].value_counts()
ChiTable11 = [[A11,B11],[C11,D11]]
chiResultKN = chi2_contingency(ChiTable11)[1]
C_KN = round(chiResultKN,3)

#Fisher's exact
A12, B12 = PositivePatients['Catheter Related Bloodstream Infection'].value_counts()
C12, D12 = NegativePatients['Catheter Related Bloodstream Infection'].value_counts()
ChiTable12 = [[A12,B12],[C12,D12]]
orCRBI, pCRBI = fisher_exact(ChiTable12, alternative='two-sided')
C_CRBI = round(pCRBI,3)

A13, B13 = PositivePatients['Intra Abdominal Infection'].value_counts()
C13, D13 = NegativePatients['Intra Abdominal Infection'].value_counts()
ChiTable13 = [[A13,B13],[C13,D13]]
chiResultIAI = chi2_contingency(ChiTable13)[1]
C_IAI = round(chiResultIAI,5)

A14, B14 = PositivePatients['Respiratory Tract Infection'].value_counts()
C14, D14 = NegativePatients['Respiratory Tract Infection'].value_counts()
ChiTable14 = [[A14,B14],[C14,D14]]
chiResultRTI = chi2_contingency(ChiTable14)[1]
C_RTI = round(chiResultRTI,3)

#Fisher's exact
A15, B15 = PositivePatients['Skin and Soft Tissue Infection'].value_counts()
C15, D15 = NegativePatients['Skin and Soft Tissue Infection'].value_counts()
ChiTable15 = [[A15,B15],[C15,D15]]
orSSTI, pSSTI = fisher_exact(ChiTable15, alternative='two-sided')
C_SSTI = round(pSSTI,3)

A16, B16 = PositivePatients['Urinary Tract Infection'].value_counts()
C16, D16 = NegativePatients['Urinary Tract Infection'].value_counts()
ChiTable16 = [[A16,B16],[C16,D16]]
chiResultUTI = chi2_contingency(ChiTable16)[1]
C_UTI = round(chiResultUTI,5)

#Fisher's exact
A8, B8 = PositivePatients['Others'].value_counts()
C8, D8 = NegativePatients['Others'].value_counts()
ChiTable8 = [[A8,B8],[C8,D8]]
orOthers, pOthers = fisher_exact(ChiTable8, alternative='two-sided')
C_Others = round(pOthers,5)

#Fisher's exact
A17, B17 = PositivePatients['Fever of Unknown Origin'].value_counts()
C17, D17 = NegativePatients['Fever of Unknown Origin'].value_counts()
ChiTable17 = [[A17,B17],[C17,D17]]
orFOU, pFOU = fisher_exact(ChiTable17, alternative='two-sided')
C_FOU = round(pFOU,5)

data = {'Total Patients' : pd.Series([T_Age, T_Gender, T_BT, T_Mortality, T_Antibiotics, T_Volume, T_PaO2, T_FiO2, T_PaO2FiO2, T_RR, T_HR, T_SBP, T_DBP, T_MAP, T_PH, T_K, T_NA, T_GCS, T_ANC, T_ALC, T_NLR, T_Creatinine, T_Lactate, T_Bilirubin, T_AST, T_PLT, T_BUN, T_WBC, T_Hematocrite, T_ESR, T_CRP, T_PCT, T_PB, T_APACHE, T_SOFA,
                                      T_DM, T_HTN, T_HF, T_CEVD, T_RD, T_LD, T_COPD, T_KN, T_CRBI, T_IAI, T_RTI, T_SSTI, T_UTI, T_Others, T_FOU], 
                                         index =['Age', 'Gender', 'Body Temperature', 'In-hospital Mortality', 'Prior Antibiotics', 'Volume', 'PaO2', 'FiO2', 'PaO2/FiO2', 'Respiratory Rate', 'Heart Rate', 'SBP', 'DBP', 'MAP', 'Arterial pH', 'K', 'Na', 'GCS Score', 
                                                 'Absolute Neutrophil Count', 'Absolute Lymphocyte Count', 'Neutrophil Lymphocyte Ratio','Creatinine', 'Lactate Level', 'Bilirubin', 'AST', 'Platelets', 'BUN', 'WBC', 'Hematocrite', 'ESR', 'CRP(mg/L)', 'PCT(ng/ml)', 'Pitt Bacteremia score', 'APACHE II score', 'SOFA score',
                                                 'Diabetes Mellintus', 'Hypertension', 'Heart Failure', 'Cerebrovascular Disease', 'Renal Disease', 'Liver Disease', 'COPD', 'Known Neoplasm', 'Catheter Related Bloodstream Infection', 'Intra Abdominal Infection', 'Respiratory Tract Infection', 'Skin and Soft Tissue Infection', 'Urinary Tract Infection', 'Others', 'Fever of Unknown Origin']),
        'Bacteremia Positive Patients' : pd.Series([P_Age, P_Gender, P_BT, P_Mortality,  P_Antibiotics, P_Volume, P_PaO2, P_FiO2, P_PaO2FiO2, P_RR, P_HR, P_SBP, P_DBP, P_MAP, P_PH, P_K, P_NA, P_GCS, P_ANC, P_ALC, P_NLR, P_Creatinine, P_Lactate, P_Bilirubin, P_AST, P_PLT, P_BUN, P_WBC, P_Hematocrite, P_ESR, P_CRP, P_PCT, P_PB, P_APACHE, P_SOFA,
                                                    P_DM, P_HTN, P_HF, P_CEVD, P_RD, P_LD, P_COPD, P_KN, P_CRBI, P_IAI, P_RTI, P_SSTI, P_UTI, P_Others, P_FOU], 
                                         index =['Age', 'Gender', 'Body Temperature', 'In-hospital Mortality', 'Prior Antibiotics', 'Volume', 'PaO2', 'FiO2', 'PaO2/FiO2', 'Respiratory Rate', 'Heart Rate', 'SBP', 'DBP', 'MAP', 
                                                 'Arterial pH', 'K', 'Na', 'GCS Score', 'Absolute Neutrophil Count', 'Absolute Lymphocyte Count', 'Neutrophil Lymphocyte Ratio','Creatinine', 'Lactate Level', 'Bilirubin', 'AST', 'Platelets', 'BUN', 'WBC', 'Hematocrite', 'ESR', 'CRP(mg/L)', 'PCT(ng/ml)', 'Pitt Bacteremia score', 'APACHE II score','SOFA score',
                                                 'Diabetes Mellintus', 'Hypertension', 'Heart Failure', 'Cerebrovascular Disease', 'Renal Disease', 'Liver Disease', 'COPD', 'Known Neoplasm', 'Catheter Related Bloodstream Infection', 'Intra Abdominal Infection', 'Respiratory Tract Infection', 'Skin and Soft Tissue Infection', 'Urinary Tract Infection', 'Others', 'Fever of Unknown Origin']),
        'Bacteremia Negative Patients' : pd.Series([N_Age, N_Gender, N_BT, N_Mortality,  N_Antibiotics, N_Volume, N_PaO2, N_FiO2, N_PaO2FiO2, N_RR, N_HR, N_SBP, N_DBP, N_MAP, N_PH, N_K, N_NA, N_GCS, N_ANC, N_ALC, N_NLR, N_Creatinine, N_Lactate, N_Bilirubin, N_AST, N_PLT, N_BUN, N_WBC, N_Hematocrite, N_ESR, N_CRP, N_PCT, N_PB, N_APACHE, N_SOFA,
                                                    N_DM, N_HTN, N_HF, N_CEVD, N_RD, N_LD, N_COPD, N_KN, N_CRBI, N_IAI, N_RTI, N_SSTI, N_UTI, N_Others, N_FOU], 
                                         index =['Age', 'Gender', 'Body Temperature', 'In-hospital Mortality', 'Prior Antibiotics', 'Volume', 'PaO2', 'FiO2', 'PaO2/FiO2', 'Respiratory Rate', 'Heart Rate', 'SBP', 'DBP', 'MAP', 
                                                 'Arterial pH', 'K', 'Na', 'GCS Score', 'Absolute Neutrophil Count', 'Absolute Lymphocyte Count', 'Neutrophil Lymphocyte Ratio', 'Creatinine', 'Lactate Level', 'Bilirubin', 'AST', 'Platelets', 'BUN', 'WBC', 'Hematocrite', 'ESR', 'CRP(mg/L)', 'PCT(ng/ml)', 'Pitt Bacteremia score', 'APACHE II score', 'SOFA score',
                                                 'Diabetes Mellintus', 'Hypertension', 'Heart Failure', 'Cerebrovascular Disease', 'Renal Disease', 'Liver Disease', 'COPD', 'Known Neoplasm', 'Catheter Related Bloodstream Infection', 'Intra Abdominal Infection', 'Respiratory Tract Infection', 'Skin and Soft Tissue Infection', 'Urinary Tract Infection', 'Others', 'Fever of Unknown Origin']),
        'P-Value' : pd.Series([U_Age, C_Gender, U_BT, C_Mortality,  C_Antibiotics, U_Volume, U_PaO2, U_FiO2, U_PaO2FiO2, U_RR, U_HR, U_SBP, U_DBP, U_MAP, U_PH, U_K, U_NA, U_GCS, U_ANC, U_ALC, U_NLR, U_Creatinine, U_Lactate, U_Bilirubin, U_AST, U_PLT, U_BUN, U_WBC, U_Hematocrite, U_ESR, U_CRP, U_PCT, U_PB, U_APACHE, U_SOFA,
                               C_DM, C_HTN, C_HF, C_CEVD, C_RD, C_LD, C_COPD, C_KN, C_CRBI, C_IAI, C_RTI, C_SSTI, C_UTI, C_Others, C_FOU], 
                    index =['Age', 'Gender', 'Body Temperature', 'In-hospital Mortality', 'Prior Antibiotics', 'Volume', 'PaO2', 'FiO2', 'PaO2/FiO2', 'Respiratory Rate', 'Heart Rate', 'SBP', 'DBP', 'MAP', 'Arterial pH', 'K', 'Na', 
                            'GCS Score', 'Absolute Neutrophil Count', 'Absolute Lymphocyte Count', 'Neutrophil Lymphocyte Ratio', 'Creatinine', 'Lactate Level', 'Bilirubin', 'AST', 'Platelets', 'BUN', 'WBC', 'Hematocrite', 'ESR', 'CRP(mg/L)', 'PCT(ng/ml)', 'Pitt Bacteremia score', 'APACHE II score', 'SOFA score',
                            'Diabetes Mellintus', 'Hypertension', 'Heart Failure', 'Cerebrovascular Disease', 'Renal Disease', 'Liver Disease', 'COPD', 'Known Neoplasm', 'Catheter Related Bloodstream Infection', 'Intra Abdominal Infection', 'Respiratory Tract Infection', 'Skin and Soft Tissue Infection', 'Urinary Tract Infection', 'Others', 'Fever of Unknown Origin'])}
ClinicalTable = pd.DataFrame(data)
ClinicalTable.to_csv('ClinicalTable1.csv')

#Remove combination infection from the subgroup analysis
BacteremiaSubgroup = PositivePatients.copy()
BacteremiaSubgroup = BacteremiaSubgroup[['Serial Number', 'Age', 'Gender', 'Bacteremia Type','Death Status', 'MAP', 'Na', 'GCS Score', 'Neutrophil Lymphocyte Ratio', 'Creatinine', 'Lactate Level', 'Bilirubin', 'AST', 'ESR', 'Platelets', 'BUN', 'CRP(mg/L)', 'PCT(ng/ml)', 'APACHE II score', 'SOFA score']]

#Select only the Bacteremia Positivse Patients from Total Patients from BacteremiaData data frame
GramPositivePatients = BacteremiaSubgroup.loc[BacteremiaSubgroup["Bacteremia Type"]=='GPC']

#Select only the Bacteremia Negative Patients from Total Patients from BacteremiaData data frame
GramNegativePatients = BacteremiaSubgroup.loc[BacteremiaSubgroup["Bacteremia Type"]=='GNR']

GP_MortalityCounts = GramPositivePatients['Death Status'].value_counts()[0]
GP_MortalityPercent = round(GramPositivePatients['Death Status'].value_counts()[0] / (GramPositivePatients['Death Status'].value_counts()[0]+GramPositivePatients['Death Status'].value_counts()[1])*100,1)
GP_Mortality = str(GP_MortalityCounts) + ' [' + str(GP_MortalityPercent) + '%]'
GN_MortalityCounts = GramNegativePatients['Death Status'].value_counts()[0]
GN_MortalityPercent = round(GramNegativePatients['Death Status'].value_counts()[0] / (GramNegativePatients['Death Status'].value_counts()[0]+GramNegativePatients['Death Status'].value_counts()[1])*100,1)
GN_Mortality = str(GN_MortalityCounts) + ' [' + str(GN_MortalityPercent) + '%]'
GA1, GB1 = GramPositivePatients['Death Status'].value_counts()
GC1, GD1 = GramNegativePatients['Death Status'].value_counts()
ChiTableG1 = [[GA1,GB1],[GC1,GD1]]
chiResultGMortality = chi2_contingency(ChiTableG1)[1]
GC_Mortality = round(chiResultGMortality,3)

GP_MedMAP = int(GramPositivePatients['MAP'].median())
GP_Q1MAP = int(np.percentile(GramPositivePatients['MAP'], 25, interpolation = 'midpoint'))
GP_Q3MAP = int(np.percentile(GramPositivePatients['MAP'], 75, interpolation = 'midpoint'))
GP_MAP = str(GP_MedMAP) + ' [' + str(GP_Q1MAP) + '-' + str(GP_Q3MAP) + ']'
GN_MedMAP = int(GramNegativePatients['MAP'].median())
GN_Q1MAP = int(np.percentile(GramNegativePatients['MAP'], 25, interpolation = 'midpoint'))
GN_Q3MAP = int(np.percentile(GramNegativePatients['MAP'], 75, interpolation = 'midpoint'))
GN_MAP = str(GN_MedMAP) + ' [' + str(GN_Q1MAP) + '-' + str(GN_Q3MAP) + ']'
GUTestResultMAP = mannwhitneyu(GramPositivePatients['MAP'], GramNegativePatients['MAP'], alternative='two-sided')
GU_MAP = round(GUTestResultMAP[1],3)

GP_MedNa = round(GramPositivePatients['Na'].median(),1)
GP_Q1Na = round(np.percentile(GramPositivePatients['Na'], 25, interpolation = 'midpoint'),1)
GP_Q3Na = round(np.percentile(GramPositivePatients['Na'], 75, interpolation = 'midpoint'),1)
GP_Na = str(GP_MedNa) + ' [' + str(GP_Q1Na) + '-' + str(GP_Q3Na) + ']'
GN_MedNa = round(GramNegativePatients['Na'].median(),1)
GN_Q1Na = round(np.percentile(GramNegativePatients['Na'], 25, interpolation = 'midpoint'),1)
GN_Q3Na = round(np.percentile(GramNegativePatients['Na'], 75, interpolation = 'midpoint'),1)
GN_Na = str(GN_MedNa) + ' [' + str(GN_Q1Na) + '-' + str(GN_Q3Na) + ']'
GUTestResultNa = mannwhitneyu(GramPositivePatients['Na'], GramNegativePatients['Na'], alternative='two-sided')
GU_Na = round(GUTestResultNa[1],3)

GP_MedGCS = int(GramPositivePatients['GCS Score'].median())
GP_Q1GCS = int(np.percentile(GramPositivePatients['GCS Score'], 25, interpolation = 'midpoint'))
GP_Q3GCS = int(np.percentile(GramPositivePatients['GCS Score'], 75, interpolation = 'midpoint'))
GP_GCS = str(GP_MedGCS) + ' [' + str(GP_Q1GCS) + '-' + str(GP_Q3GCS) + ']'
GN_MedGCS = int(GramNegativePatients['GCS Score'].median())
GN_Q1GCS = int(np.percentile(GramNegativePatients['GCS Score'], 25, interpolation = 'midpoint'))
GN_Q3GCS = int(np.percentile(GramNegativePatients['GCS Score'], 75, interpolation = 'midpoint'))
GN_GCS = str(GN_MedGCS) + ' [' + str(GN_Q1GCS) + '-' + str(GN_Q3GCS) + ']'
GUTestResultGCS = mannwhitneyu(GramPositivePatients['GCS Score'], GramNegativePatients['GCS Score'], alternative='two-sided')
GU_GCS = round(GUTestResultGCS[1],3)

GP_MedNLR = round(GramPositivePatients['Neutrophil Lymphocyte Ratio'].median(),1)
GP_Q1NLR = round(np.percentile(GramPositivePatients['Neutrophil Lymphocyte Ratio'], 25, interpolation = 'midpoint'),1)
GP_Q3NLR = round(np.percentile(GramPositivePatients['Neutrophil Lymphocyte Ratio'], 75, interpolation = 'midpoint'),1)
GP_NLR = str(GP_MedNLR) + ' [' + str(GP_Q1NLR) + '-' + str(GP_Q3NLR) + ']'
GN_MedNLR = round(GramNegativePatients['Neutrophil Lymphocyte Ratio'].median(),1)
GN_Q1NLR = round(np.percentile(GramNegativePatients['Neutrophil Lymphocyte Ratio'], 25, interpolation = 'midpoint'),1)
GN_Q3NLR = round(np.percentile(GramNegativePatients['Neutrophil Lymphocyte Ratio'], 75, interpolation = 'midpoint'),1)
GN_NLR = str(GN_MedNLR) + ' [' + str(GN_Q1NLR) + '-' + str(GN_Q3NLR) + ']'
GUTestResultNLR = mannwhitneyu(GramPositivePatients['Neutrophil Lymphocyte Ratio'], GramNegativePatients['Neutrophil Lymphocyte Ratio'], alternative='two-sided')
GU_NLR = round(GUTestResultNLR[1],3)

GP_MedCreatinine = round(GramPositivePatients['Creatinine'].median(),2)
GP_Q1Creatinine = round(np.percentile(GramPositivePatients['Creatinine'], 25, interpolation = 'midpoint'),2)
GP_Q3Creatinine = round(np.percentile(GramPositivePatients['Creatinine'], 75, interpolation = 'midpoint'),2)
GP_Creatinine = str(GP_MedCreatinine) + ' [' + str(GP_Q1Creatinine) + '-' + str(GP_Q3Creatinine) + ']'
GN_MedCreatinine = round(GramNegativePatients['Creatinine'].median(),2)
GN_Q1Creatinine = round(np.percentile(GramNegativePatients['Creatinine'], 25, interpolation = 'midpoint'),2)
GN_Q3Creatinine = round(np.percentile(GramNegativePatients['Creatinine'], 75, interpolation = 'midpoint'),2)
GN_Creatinine = str(GN_MedCreatinine) + ' [' + str(GN_Q1Creatinine) + '-' + str(GN_Q3Creatinine) + ']'
GUTestResultCreatinine = mannwhitneyu(GramPositivePatients['Creatinine'], GramNegativePatients['Creatinine'], alternative='two-sided')
GU_Creatinine = round(GUTestResultCreatinine[1],3)

GP_MedLactate = round(GramPositivePatients['Lactate Level'].median(),1)
GP_Q1Lactate = round(np.percentile(GramPositivePatients['Lactate Level'], 25, interpolation = 'midpoint'),1)
GP_Q3Lactate = round(np.percentile(GramPositivePatients['Lactate Level'], 75, interpolation = 'midpoint'),1)
GP_Lactate = str(GP_MedLactate) + ' [' + str(GP_Q1Lactate) + '-' + str(GP_Q3Lactate) + ']'
GN_MedLactate = round(GramNegativePatients['Lactate Level'].median(),1)
GN_Q1Lactate = round(np.percentile(GramNegativePatients['Lactate Level'], 25, interpolation = 'midpoint'),1)
GN_Q3Lactate = round(np.percentile(GramNegativePatients['Lactate Level'], 75, interpolation = 'midpoint'),1)
GN_Lactate = str(GN_MedLactate) + ' [' + str(GN_Q1Lactate) + '-' + str(GN_Q3Lactate) + ']'
GUTestResultLactate = mannwhitneyu(GramPositivePatients['Lactate Level'], GramNegativePatients['Lactate Level'], alternative='two-sided')
GU_Lactate = round(GUTestResultLactate[1],3)

GP_MedBilirubin = round(GramPositivePatients['Bilirubin'].median(),2)
GP_Q1Bilirubin = round(np.percentile(GramPositivePatients['Bilirubin'], 25, interpolation = 'midpoint'),2)
GP_Q3Bilirubin = round(np.percentile(GramPositivePatients['Bilirubin'], 75, interpolation = 'midpoint'),2)
GP_Bilirubin = str(GP_MedBilirubin) + ' [' + str(GP_Q1Bilirubin) + '-' + str(GP_Q3Bilirubin) + ']'
GN_MedBilirubin = round(GramNegativePatients['Bilirubin'].median(),2)
GN_Q1Bilirubin = round(np.percentile(GramNegativePatients['Bilirubin'], 25, interpolation = 'midpoint'),2)
GN_Q3Bilirubin = round(np.percentile(GramNegativePatients['Bilirubin'], 75, interpolation = 'midpoint'),2)
GN_Bilirubin = str(GN_MedBilirubin) + ' [' + str(GN_Q1Bilirubin) + '-' + str(GN_Q3Bilirubin) + ']'
GUTestResultBilirubin = mannwhitneyu(GramPositivePatients['Bilirubin'], GramNegativePatients['Bilirubin'], alternative='two-sided')
GU_Bilirubin = round(GUTestResultBilirubin[1],3)

GP_MedAST = round(GramPositivePatients['AST'].median(),1)
GP_Q1AST = round(np.percentile(GramPositivePatients['AST'], 25, interpolation = 'midpoint'),1)
GP_Q3AST = round(np.percentile(GramPositivePatients['AST'], 75, interpolation = 'midpoint'),1)
GP_AST = str(GP_MedAST) + ' [' + str(GP_Q1AST) + '-' + str(GP_Q3AST) + ']'
GN_MedAST = round(GramNegativePatients['AST'].median(),1)
GN_Q1AST = round(np.percentile(GramNegativePatients['AST'], 25, interpolation = 'midpoint'),1)
GN_Q3AST = round(np.percentile(GramNegativePatients['AST'], 75, interpolation = 'midpoint'),1)
GN_AST = str(GN_MedAST) + ' [' + str(GN_Q1AST) + '-' + str(GN_Q3AST) + ']'
GUTestResultAST = mannwhitneyu(GramPositivePatients['AST'], GramNegativePatients['AST'], alternative='two-sided')
GU_AST = round(GUTestResultAST[1],3)

GP_MedESR = int(GramPositivePatients['ESR'].median())
GP_Q1ESR = int(np.percentile(GramPositivePatients['ESR'], 25, interpolation = 'midpoint'))
GP_Q3ESR = int(np.percentile(GramPositivePatients['ESR'], 75, interpolation = 'midpoint'))
GP_ESR = str(GP_MedESR) + ' [' + str(GP_Q1ESR) + '-' + str(GP_Q3ESR) + ']'
GN_MedESR = int(GramNegativePatients['ESR'].median())
GN_Q1ESR = int(np.percentile(GramNegativePatients['ESR'], 25, interpolation = 'midpoint'))
GN_Q3ESR = int(np.percentile(GramNegativePatients['ESR'], 75, interpolation = 'midpoint'))
GN_ESR = str(GN_MedESR) + ' [' + str(GN_Q1ESR) + '-' + str(GN_Q3ESR) + ']'
GUTestResultESR = mannwhitneyu(GramPositivePatients['ESR'], GramNegativePatients['ESR'], alternative='two-sided')
GU_ESR = round(GUTestResultESR[1],3)

GP_MedPLT = int(GramPositivePatients['Platelets'].median())
GP_Q1PLT = int(np.percentile(GramPositivePatients['Platelets'], 25, interpolation = 'midpoint'))
GP_Q3PLT = int(np.percentile(GramPositivePatients['Platelets'], 75, interpolation = 'midpoint'))
GP_PLT = str(GP_MedPLT) + ' [' + str(GP_Q1PLT) + '-' + str(GP_Q3PLT) + ']'
GN_MedPLT = int(GramNegativePatients['Platelets'].median())
GN_Q1PLT = int(np.percentile(GramNegativePatients['Platelets'], 25, interpolation = 'midpoint'))
GN_Q3PLT = int(np.percentile(GramNegativePatients['Platelets'], 75, interpolation = 'midpoint'))
GN_PLT = str(GN_MedPLT) + ' [' + str(GN_Q1PLT) + '-' + str(GN_Q3PLT) + ']'
GUTestResultPLT = mannwhitneyu(GramPositivePatients['Platelets'], GramNegativePatients['Platelets'], alternative='two-sided')
GU_PLT = round(GUTestResultPLT[1],3)

GP_MedBUN = round(GramPositivePatients['BUN'].median(),1)
GP_Q1BUN = round(np.percentile(GramPositivePatients['BUN'], 25, interpolation = 'midpoint'),1)
GP_Q3BUN = round(np.percentile(GramPositivePatients['BUN'], 75, interpolation = 'midpoint'),1)
GP_BUN = str(GP_MedBUN) + ' [' + str(GP_Q1BUN) + '-' + str(GP_Q3BUN) + ']'
GN_MedBUN = round(GramNegativePatients['BUN'].median(),1)
GN_Q1BUN = round(np.percentile(GramNegativePatients['BUN'], 25, interpolation = 'midpoint'),1)
GN_Q3BUN = round(np.percentile(GramNegativePatients['BUN'], 75, interpolation = 'midpoint'),1)
GN_BUN = str(GN_MedBUN) + ' [' + str(GN_Q1BUN) + '-' + str(GN_Q3BUN) + ']'
GUTestResultBUN = mannwhitneyu(GramPositivePatients['BUN'], GramNegativePatients['BUN'], alternative='two-sided')
GU_BUN = round(GUTestResultBUN[1],3)

GP_MedCRP = round(GramPositivePatients['CRP(mg/L)'].median(),1)
GP_Q1CRP = round(np.percentile(GramPositivePatients['CRP(mg/L)'], 25, interpolation = 'midpoint'),1)
GP_Q3CRP = round(np.percentile(GramPositivePatients['CRP(mg/L)'], 75, interpolation = 'midpoint'),1)
GP_CRP = str(GP_MedCRP) + ' [' + str(GP_Q1CRP) + '-' + str(GP_Q3CRP) + ']'
GN_MedCRP = round(GramNegativePatients['CRP(mg/L)'].median(),1)
GN_Q1CRP = round(np.percentile(GramNegativePatients['CRP(mg/L)'], 25, interpolation = 'midpoint'),1)
GN_Q3CRP = round(np.percentile(GramNegativePatients['CRP(mg/L)'], 75, interpolation = 'midpoint'),1)
GN_CRP = str(GN_MedCRP) + ' [' + str(GN_Q1CRP) + '-' + str(GN_Q3CRP) + ']'
GUTestResultCRP = mannwhitneyu(GramPositivePatients['CRP(mg/L)'], GramNegativePatients['CRP(mg/L)'], alternative='two-sided')
GU_CRP = round(GUTestResultCRP[1],3)

GP_MedPCT = round(GramPositivePatients['PCT(ng/ml)'].median(),2)
GP_Q1PCT = round(np.percentile(GramPositivePatients['PCT(ng/ml)'], 25, interpolation = 'midpoint'),2)
GP_Q3PCT = round(np.percentile(GramPositivePatients['PCT(ng/ml)'], 75, interpolation = 'midpoint'),2)
GP_PCT = str(GP_MedPCT) + ' [' + str(GP_Q1PCT) + '-' + str(GP_Q3PCT) + ']'
GN_MedPCT = round(GramNegativePatients['PCT(ng/ml)'].median(),2)
GN_Q1PCT = round(np.percentile(GramNegativePatients['PCT(ng/ml)'], 25, interpolation = 'midpoint'),2)
GN_Q3PCT = round(np.percentile(GramNegativePatients['PCT(ng/ml)'], 75, interpolation = 'midpoint'),2)
GN_PCT = str(GN_MedPCT) + ' [' + str(GN_Q1PCT) + '-' + str(GN_Q3PCT) + ']'
GUTestResultPCT = mannwhitneyu(GramPositivePatients['PCT(ng/ml)'], GramNegativePatients['PCT(ng/ml)'], alternative='two-sided')
GU_PCT = round(GUTestResultPCT[1],3)

GP_MedAPACHE = int(GramPositivePatients['APACHE II score'].median())
GP_Q1APACHE = int(np.percentile(GramPositivePatients['APACHE II score'], 25, interpolation = 'midpoint'))
GP_Q3APACHE = int(np.percentile(GramPositivePatients['APACHE II score'], 75, interpolation = 'midpoint'))
GP_APACHE = str(GP_MedAPACHE) + ' [' + str(GP_Q1APACHE) + '-' + str(GP_Q3APACHE) + ']'
GN_MedAPACHE = int(GramNegativePatients['APACHE II score'].median())
GN_Q1APACHE = int(np.percentile(GramNegativePatients['APACHE II score'], 25, interpolation = 'midpoint'))
GN_Q3APACHE = int(np.percentile(GramNegativePatients['APACHE II score'], 75, interpolation = 'midpoint'))
GN_APACHE = str(GN_MedAPACHE) + ' [' + str(GN_Q1APACHE) + '-' + str(GN_Q3APACHE) + ']'
GUTestResultAPACHE = mannwhitneyu(GramPositivePatients['APACHE II score'], GramNegativePatients['APACHE II score'], alternative='two-sided')
GU_APACHE = round(GUTestResultAPACHE[1],3)

GP_MedSOFA = int(GramPositivePatients['SOFA score'].median())
GP_Q1SOFA = int(np.percentile(GramPositivePatients['SOFA score'], 25, interpolation = 'midpoint'))
GP_Q3SOFA = int(np.percentile(GramPositivePatients['SOFA score'], 75, interpolation = 'midpoint'))
GP_SOFA = str(GP_MedSOFA) + ' [' + str(GP_Q1SOFA) + '-' + str(GP_Q3SOFA) + ']'
GN_MedSOFA = int(GramNegativePatients['SOFA score'].median())
GN_Q1SOFA = int(np.percentile(GramNegativePatients['SOFA score'], 25, interpolation = 'midpoint'))
GN_Q3SOFA = int(np.percentile(GramNegativePatients['SOFA score'], 75, interpolation = 'midpoint'))
GN_SOFA = str(GN_MedSOFA) + ' [' + str(GN_Q1SOFA) + '-' + str(GN_Q3SOFA) + ']'
GUTestResultSOFA = mannwhitneyu(GramPositivePatients['SOFA score'], GramNegativePatients['SOFA score'], alternative='two-sided')
GU_SOFA = round(GUTestResultSOFA[1],3)


data2 = {'Gram-Positive Cocci' : pd.Series([GP_MAP, GP_Na, GP_GCS, GP_NLR, GP_Creatinine, GP_Lactate, GP_Bilirubin, GP_AST, GP_BUN, GP_ESR, GP_PLT, GP_CRP, GP_PCT, GP_APACHE, GP_SOFA, GP_Mortality], 
                                         index =['MAP', 'Na', 'GCS Score','Neutrophil Lymphocyte Ratio','Creatinine', 'Lactate Level', 
                                                 'Bilirubin', 'AST', 'BUN', 'ESR', 'Platelets', 'CRP', 'PCT', 'APACHE II score','SOFA score', 'In-hospital Mortality']),
        'Gram-Negative Rods' : pd.Series([GN_MAP, GN_Na, GN_GCS, GN_NLR, GN_Creatinine, GN_Lactate, GN_Bilirubin, GN_AST, GN_BUN, GN_ESR, GN_PLT, GN_CRP, GN_PCT, GN_APACHE, GN_SOFA, GN_Mortality], 
                                         index =['MAP', 'Na', 'GCS Score', 'Neutrophil Lymphocyte Ratio', 'Creatinine', 'Lactate Level', 
                                                 'Bilirubin', 'AST', 'BUN', 'ESR', 'Platelets', 'CRP', 'PCT', 'APACHE II score','SOFA score', 'In-hospital Mortality']),
        'P-Value' : pd.Series([GU_MAP, GU_Na, GU_GCS, GU_NLR, GU_Creatinine, GU_Lactate, GU_Bilirubin, GU_AST, GU_BUN, GU_ESR, GU_PLT, GU_CRP, GU_PCT, GU_APACHE, GU_SOFA, GC_Mortality], 
                    index =['MAP', 'Na', 'GCS Score', 'Neutrophil Lymphocyte Ratio', 'Creatinine', 
                            'Lactate Level', 'Bilirubin', 'AST', 'BUN', 'ESR', 'Platelets', 'CRP', 'PCT', 'APACHE II score','SOFA score', 'In-hospital Mortality'])}
ClinicalTable2 = pd.DataFrame(data2)
ClinicalTable2.to_csv('ClinicalTable2.csv')

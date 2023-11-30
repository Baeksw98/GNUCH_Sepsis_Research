"""
Created on November 16th, 2021
@author: Sangwon Baek
KyeongsangUniversity Sepsis Severity score calculation code
"""

import numpy as np
import pandas as pd 
import os

#setting working directory to the folder
path=r"C:\Users\user-pc\Desktop\KSU\Sepsis Research\Data\CSV"
os.chdir(path)
os.getcwd()
pd.options.mode.chained_assignment = None 

#Reading the files
Data = pd.read_csv('ClinicalData.csv')

#Convert the PaO2/FiO2 values into Respiratory system SOFA subscore
binsRespSOFA = [-np.inf, 99, 199, 299, 399, np.inf]
labelsRespSOFA=[4,3,2,1,0]
Data['Respiratory SOFA Subscore'] = pd.cut(Data['PaO2/FiO2'], bins=binsRespSOFA, labels=labelsRespSOFA)

#Convert the GCS score values into Nervous system SOFA subscore
binsCNSSOFA = [-np.inf, 5.9, 9.9, 12.9, 14.9, np.inf]
labelsCNSSOFA=[4,3,2,1,0]
Data['CNS SOFA Subscore'] = pd.cut(Data['GCS Score'], bins=binsCNSSOFA, labels=labelsCNSSOFA)

CardioScore = []
for i in range(len(Data)):
    if Data['Dosage Type'].loc[i] == 'None':
        if Data['MAP'].loc[i] > 70:
            CardioScore.append(0)
        else:
            CardioScore.append(1)
    elif Data['Dosage Type'].loc[i]=='Dopamine' and Data['Total Dosage'].loc[i]<=5 or Data['Dosage Type'].loc[i]=='Dobutamine' and Data['Total Dosage'].loc[i]<=5:
        CardioScore.append(2)
    elif Data['Dosage Type'].loc[i]=='Dobutamine' and  Data['Total Dosage'].loc[i]>5  or  Data['Dosage Type'].loc[i]=='Norpin' and Data['Total Dosage'].loc[i]<=0.1 or Data['Dosage Type'].loc[i]=='Epinephrine' and Data['Total Dosage'].loc[i]<=0.1:
        CardioScore.append(3)
    else:
        CardioScore.append(4)       
Data['Cardio SOFA Subscore']=CardioScore  

#Convert the Bilirubin values into Liver system SOFA subscore
binsLiverSOFA = [-np.inf, 1.19,1.99,5.99,11.99, np.inf]
labelsLiverSOFA=[0,1,2,3,4]
Data['Liver SOFA Subscore'] = pd.cut(Data['Bilirubin'], bins=binsLiverSOFA, labels=labelsLiverSOFA)

#Convert the Platelet values into Coagulation system SOFA subscore
CoagulationScore = []
for i in range(len(Data)):
    if Data['Platelets'].loc[i] >= 150 and Data['Platelet Concentration Transfusion Status'].loc[i]=='N':
        CoagulationScore.append(0)
    elif Data['Platelets'].loc[i] >= 100 and Data['Platelets'].loc[i] < 150 and Data['Platelet Concentration Transfusion Status'].loc[i]=='N':
        CoagulationScore.append(1)
    elif Data['Platelets'].loc[i] >= 50 and Data['Platelets'].loc[i] < 100 and Data['Platelet Concentration Transfusion Status'].loc[i]=='N':
        CoagulationScore.append(2)
    elif Data['Platelets'].loc[i] >= 20 and Data['Platelets'].loc[i] < 50 and Data['Platelet Concentration Transfusion Status'].loc[i]=='N':
        CoagulationScore.append(3)
    else:
        CoagulationScore.append(4)        
Data['Coagulation SOFA Subscore']=CoagulationScore  

#Convert the Creatinine values into Kidney system SOFA subscore
KidneyScore = []
for i in range(len(Data)):
    if Data['Creatinine'].loc[i] < 1.2 and Data['Hemodialysis Status'].loc[i]=='N':
        KidneyScore.append(0)
    elif Data['Creatinine'].loc[i] < 2 and Data['Creatinine'].loc[i] >= 1.2 and Data['Hemodialysis Status'].loc[i]=='N':
        KidneyScore.append(1)
    elif Data['Creatinine'].loc[i] < 3.5 and Data['Creatinine'].loc[i] >= 2 and Data['Hemodialysis Status'].loc[i]=='N':
        KidneyScore.append(2)
    elif Data['Creatinine'].loc[i] < 5 and Data['Creatinine'].loc[i] >= 3.5 and Data['Hemodialysis Status'].loc[i]=='N':
        KidneyScore.append(3)
    else:
        KidneyScore.append(4)        
Data['Kidney SOFA Subscore']=KidneyScore  

#Calculate the SOFA score
Checker = Data[['Serial Number', 'PaO2/FiO2', 'Respiratory SOFA Subscore', 'GCS Score', 'CNS SOFA Subscore' , 'MAP', 'Dosage Type', 'Total Dosage', 
                'Cardio SOFA Subscore', 'Bilirubin', 'Liver SOFA Subscore', 'Platelets', 'Platelet Concentration Transfusion Status', 
                'Coagulation SOFA Subscore', 'Creatinine', 'Hemodialysis Status', 'Kidney SOFA Subscore']]
SOFAcalculator = Data[['Respiratory SOFA Subscore', 'CNS SOFA Subscore' , 'Cardio SOFA Subscore', 'Liver SOFA Subscore', 'Coagulation SOFA Subscore', 'Kidney SOFA Subscore']]
SOFAcalculator['SOFA score']=SOFAcalculator.sum(axis=1, skipna=True)
Data['SOFA score']= SOFAcalculator['SOFA score']
Data=Data.drop(columns=['Respiratory SOFA Subscore', 'CNS SOFA Subscore' , 'Cardio SOFA Subscore', 'Liver SOFA Subscore', 'Coagulation SOFA Subscore', 'Kidney SOFA Subscore'])

#APACHE II Score = acute physiology score + age score + chronic health status score
#The score is between 0-71, increasing score associated with increasing risk of mortality
#Temperature score
TempScore = []
for i in range(len(Data)):
    if 36 <= Data['Body Temperature'].loc[i] < 38.5:
        TempScore.append(0)
    elif 38.5 <= Data['Body Temperature'].loc[i] < 39 or 34 <= Data['Body Temperature'].loc[i] < 36:
        TempScore.append(1)
    elif 32 <= Data['Body Temperature'].loc[i] < 34:
        TempScore.append(2)
    elif 39 <= Data['Body Temperature'].loc[i] < 41 or 30 <= Data['Body Temperature'].loc[i] < 32:
        TempScore.append(3)
    else:
        TempScore.append(4)        
Data['Temp Subscore']=TempScore  
TempScoreChecker = Data[['Serial Number','Body Temperature','Temp Subscore']]

#Mean arterial pressure score
MAPScore = []
for i in range(len(Data)):
    if 70 <= Data['MAP'].loc[i] < 110:
        MAPScore.append(0)
    elif 110 <= Data['MAP'].loc[i] < 130 or 50 <= Data['MAP'].loc[i] < 70:
        MAPScore.append(2)
    elif 130 <= Data['MAP'].loc[i] < 160:
        MAPScore.append(3)
    else:
        MAPScore.append(4)        
Data['MAP Subscore']=MAPScore  
MAPScoreChecker = Data[['Serial Number','MAP','MAP Subscore']]

#Heart rate score
HRScore = []
for i in range(len(Data)):
    if 70 <= Data['Heart Rate'].loc[i] < 110:
        HRScore.append(0)
    elif 110 <= Data['Heart Rate'].loc[i] < 140 or 55 <= Data['Heart Rate'].loc[i] < 70:
        HRScore.append(2)
    elif 140 <= Data['Heart Rate'].loc[i] < 180 or 40 <= Data['Heart Rate'].loc[i] < 55:
        HRScore.append(3)
    else:
        HRScore.append(4)        
Data['HR Subscore']=HRScore  
HRScoreChecker = Data[['Serial Number','Heart Rate','HR Subscore']]

#Respiratory rate score
RRScore = []
for i in range(len(Data)):
    if 12 <= Data['Respiratory Rate'].loc[i] < 25:
        RRScore.append(0)
    elif 25 <= Data['Respiratory Rate'].loc[i] < 35 or 10 <= Data['Respiratory Rate'].loc[i] < 12:
        RRScore.append(1)
    elif 6 <= Data['Respiratory Rate'].loc[i] < 10:
        RRScore.append(2)
    elif 35 <= Data['Respiratory Rate'].loc[i] < 50:
        RRScore.append(3)
    else:
        RRScore.append(4)        
Data['RR Subscore']=RRScore  
RRScoreChecker = Data[['Serial Number','Respiratory Rate','RR Subscore']]

#Oxygenation score
OxygenationScore = []
for i in range(len(Data)):
    if 0.5 <= Data['FiO2'].loc[i]:
        if Data['A-aDO2'].loc[i] < 200: 
            OxygenationScore.append(0)
        elif 200 <= Data['A-aDO2'].loc[i] < 350:
            OxygenationScore.append(2)
        elif 350 <= Data['A-aDO2'].loc[i] < 500:
            OxygenationScore.append(3)
        elif 500 <= Data['A-aDO2'].loc[i]:
            OxygenationScore.append(4)
    else:
        if Data['PaO2'].loc[i] > 70: 
            OxygenationScore.append(0)
        elif 61 <= Data['PaO2'].loc[i] <= 70:
            OxygenationScore.append(1)
        elif 55 <= Data['PaO2'].loc[i] < 61:
            OxygenationScore.append(3)
        elif Data['PaO2'].loc[i] < 55:
            OxygenationScore.append(4)     
Data['Oxygenation Subscore']=OxygenationScore  
OxygenationScoreChecker = Data[['Serial Number','FiO2','PaO2','Oxygenation Subscore']]

#Arterial pH score
PHScore = []
for i in range(len(Data)):
    if 7.33 <= Data['Arterial pH'].loc[i] < 7.5:
        PHScore.append(0)
    elif 7.5 <= Data['Arterial pH'].loc[i] < 7.6:
        PHScore.append(1)
    elif 7.25 <= Data['Arterial pH'].loc[i] < 7.33:
        PHScore.append(2)
    elif 7.6 <= Data['Arterial pH'].loc[i] < 7.7 or 7.15 <= Data['Arterial pH'].loc[i] < 7.25:
        PHScore.append(3)
    else:
        PHScore.append(4)        
Data['PH Subscore']=PHScore  
PHScoreChecker = Data[['Serial Number','Arterial pH','PH Subscore']]

#Serum Na score
NAScore = []
for i in range(len(Data)):
    if 130 <= Data['Na'].loc[i] < 150:
        NAScore.append(0)
    elif 150 <= Data['Na'].loc[i] < 155:
        NAScore.append(1)
    elif 155 <= Data['Na'].loc[i] < 160 or 120 <= Data['Na'].loc[i] < 130:
        NAScore.append(2)
    elif 160 <= Data['Na'].loc[i] < 180 or 111 <= Data['Na'].loc[i] < 120:
        NAScore.append(3)
    else:
        NAScore.append(4)        
Data['NA Subscore']=NAScore  
NAScoreChecker = Data[['Serial Number','Na','NA Subscore']]

#Serum K score
KScore = []
for i in range(len(Data)):
    if 3.5 <= Data['K'].loc[i] < 5.5:
        KScore.append(0)
    elif 5.5 <= Data['K'].loc[i] < 6 or 3 <= Data['K'].loc[i] < 3.5:
        KScore.append(1)
    elif 2.5 <= Data['K'].loc[i] < 3:
        KScore.append(2)
    elif 6 <= Data['K'].loc[i] < 7:
        KScore.append(3)
    else:
        KScore.append(4)        
Data['K Subscore']=KScore  
KScoreChecker = Data[['Serial Number','K','K Subscore']]

#Serum creatinine score *Double point for acute renal failure
CreatinineScore = []
for i in range(len(Data)):
    if 0.6 <= Data['Creatinine'].loc[i] < 1.5:
        CreatinineScore.append(0)
    elif 1.5 <= Data['Creatinine'].loc[i] < 2.0 or Data['Creatinine'].loc[i] < 0.6:
        CreatinineScore.append(2)
    elif 2.0 <= Data['Creatinine'].loc[i] < 3.5:
        CreatinineScore.append(3)
    else:
        CreatinineScore.append(4)     
Data['Creatinine Subscore']=CreatinineScore  
for i in range(len(Data)):
    if Data['AKI Status'].loc[i]=='Y':
        Data['Creatinine Subscore'].loc[i] = Data['Creatinine Subscore'].loc[i]*2
CreatinineScoreChecker = Data[['Serial Number','Creatinine','Creatinine Subscore', 'AKI Status']]

#Hct score
HctScore = []
for i in range(len(Data)):
    if 30 <= Data['Hematocrite'].loc[i] < 46:
        HctScore.append(0)
    elif 46 <= Data['Hematocrite'].loc[i] < 50:
        HctScore.append(1)
    elif 50 <= Data['Hematocrite'].loc[i] < 60 or 20 <= Data['Hematocrite'].loc[i] < 30:
        HctScore.append(2)
    elif 60 <= Data['Hematocrite'].loc[i] or Data['Hematocrite'].loc[i] < 20:
        HctScore.append(4)      
Data['Hct Subscore']=HctScore  
HctScoreChecker = Data[['Serial Number','Hematocrite','Hct Subscore']]

#WBC score
WBCScore = []
for i in range(len(Data)):
    if 3 <= Data['WBC'].loc[i] < 15:
        WBCScore.append(0)
    elif 15 <= Data['WBC'].loc[i] < 20:
        WBCScore.append(1)
    elif 20 <= Data['WBC'].loc[i] < 40 or 1 <= Data['WBC'].loc[i] < 3:
        WBCScore.append(2)
    elif 40 <= Data['WBC'].loc[i] or Data['WBC'].loc[i] < 1:
        WBCScore.append(4)      
Data['WBC Subscore']=WBCScore  
WBCScoreChecker = Data[['Serial Number','WBC','WBC Subscore']]

#GCS score 
GCSscore = []
for i in range(len(Data)):
    GCSscore.append(15-Data['GCS Score'].loc[i])
Data['GCS Subscore']=GCSscore  
GCSScoreChecker = Data[['Serial Number','GCS Score','GCS Subscore']]

#Age score
AgeScore = []
for i in range(len(Data)):
    if Data['Age'].loc[i] < 44:
        AgeScore.append(0)
    elif 44 <= Data['Age'].loc[i] < 55:
        AgeScore.append(2)
    elif 55 <= Data['Age'].loc[i] < 65:
        AgeScore.append(3)
    elif 65 <= Data['Age'].loc[i] < 75:
        AgeScore.append(5)
    else:
        AgeScore.append(6)        
Data['Age Subscore']=AgeScore  
AgeScoreChecker = Data[['Serial Number','Age','Age Subscore']]

#Chronic Health status score 
#Organ sufficiency include (hepatic, cardiovascular, renal, pulmonary) failure
#Elective postoperative patient with immunocompromise or history of severe organ failure: +2
#Nonoperative patient or emergency postperative patient with immunocompromise or severe organ inssuficiency: +5
ChronicHealthScore = []
for i in range(len(Data)):
    if Data['Elective Postoperative Patients'].loc[i]=='Y' and Data['Immunocompromised State'].loc[i]=='Y':
        ChronicHealthScore.append(2)
    elif Data['Emergency Postoperative Patients'].loc[i]=='Y' and Data['Immunocompromised State'].loc[i]=='Y':
        ChronicHealthScore.append(5)   
    elif Data['Non Operative Patients'].loc[i]=='Y' and Data['Immunocompromised State'].loc[i]=='Y':
        ChronicHealthScore.append(5)
    else:
        ChronicHealthScore.append(0)
Data['ChronicHealth Subscore'] = ChronicHealthScore
ChronicHealthScoreChecker = Data[['Elective Postoperative Patients', 'Emergency Postoperative Patients', 'Non Operative Patients', 'Immunocompromised State', 'ChronicHealth Subscore']]


#Calculate the APACHE score
APACHEcalculator = Data[['Temp Subscore','MAP Subscore', 'HR Subscore', 'RR Subscore', 'Oxygenation Subscore', 'PH Subscore', 
                         'NA Subscore', 'K Subscore', 'Creatinine Subscore', 'Hct Subscore', 'WBC Subscore', 'GCS Subscore', 'Age Subscore', 'ChronicHealth Subscore']]
APACHEcalculator['APACHE II score']=APACHEcalculator.sum(axis=1, skipna=True)
Data['APACHE II score']= APACHEcalculator['APACHE II score']
Data=Data.drop(columns=['Temp Subscore','MAP Subscore', 'HR Subscore', 'RR Subscore', 'Oxygenation Subscore', 'PH Subscore', 
           'NA Subscore', 'K Subscore', 'Creatinine Subscore', 'Hct Subscore', 'WBC Subscore', 'GCS Subscore', 'Age Subscore', 'ChronicHealth Subscore'])

#Pitt Bacteremia Score
#The score is between 0-14, increasing score associated with increasing risk of mortality
TempScorePB = []
for i in range(len(Data)):
    if 36.1 <= Data['Body Temperature'].loc[i] < 39:
        TempScorePB.append(0)
    elif 35.1 <= Data['Body Temperature'].loc[i] < 36.1 or 39 <= Data['Body Temperature'].loc[i] < 40:
        TempScorePB.append(1)
    else:
        TempScorePB.append(2)   
Data['PB Temp Subscore']=TempScorePB  
TempScoreCheckerPB = Data[['Serial Number','Body Temperature','PB Temp Subscore']]

HypotensionScorePB = []
for i in range(len(Data)):
    if Data['SBP'].loc[i] < 90 or Data['Total Dosage'].loc[i] > 0:
        HypotensionScorePB.append(2)
    else:
        HypotensionScorePB.append(0)   
Data['PB HTN Subscore']=HypotensionScorePB  
HTNScoreCheckerPB = Data[['Serial Number','SBP', 'Total Dosage', 'PB HTN Subscore']]

MechnicalVentilationPB = []
word = 'Ventilator'
for i in range(len(Data)):
    if word in Data['O2 Inhalation Method']:
        MechnicalVentilationPB.append(2)
    else:
        MechnicalVentilationPB.append(0)   
Data['PB Ventilation Subscore']=MechnicalVentilationPB  
VentilationScoreCheckerPB = Data[['Serial Number','O2 Inhalation Method', 'PB Ventilation Subscore']]

CardiacArrestPB = []
for i in range(len(Data)):
    if Data['Cardiac Arrest Status'].loc[i]=='Y':
        CardiacArrestPB.append(4)
    else:
        CardiacArrestPB.append(0)   
Data['PB Cardiac Arrest Subscore']=CardiacArrestPB  
CardiacArrestScoreCheckerPB = Data[['Serial Number','Cardiac Arrest Status', 'PB Cardiac Arrest Subscore']]

MentalStatusPB = []
for i in range(len(Data)):
    if Data['Mental Status'].loc[i]=='Comatose':
        MentalStatusPB.append(4)
    elif Data['Mental Status'].loc[i]=='Stuporous':
        MentalStatusPB.append(2)
    elif Data['Mental Status'].loc[i]=='Disoriented':
        MentalStatusPB.append(1)
    else:
        MentalStatusPB.append(0)   
Data['PB Mental Subscore']=MentalStatusPB  
MentalScoreCheckerPB = Data[['Serial Number','Mental Status', 'PB Mental Subscore']]

#Calculate the Pitt Bacteremia score
PittBacteremiacalculator = Data[['PB Temp Subscore', 'PB HTN Subscore', 'PB Ventilation Subscore', 'PB Cardiac Arrest Subscore', 'PB Mental Subscore']]
PittBacteremiacalculator['Pitt Bacteremia score']=PittBacteremiacalculator.sum(axis=1, skipna=True)
Data['Pitt Bacteremia score']= PittBacteremiacalculator['Pitt Bacteremia score']
Data=Data.drop(columns=['PB Temp Subscore', 'PB HTN Subscore', 'PB Ventilation Subscore', 'PB Cardiac Arrest Subscore', 'PB Mental Subscore'])

#Create a CSV file of resulting Bacteremia Data 
Data.to_csv('ClinicalDataWithScores.csv', encoding='cp949')
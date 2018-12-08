---
nav_include: 2
title: Models
notebook: Models_combined.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}


## Final Project: Alzheimer's Disease and Cognitive Impairment Prediction

**Harvard University**<br/>
**Fall 2018**<br/>
**Instructors**: Pavlos Protopapas, Kevin Rader

**Team Members**: Zeo Liu, Connor Mccann, David Melancon

<hr style="height:2pt">





```python
'''NOTEBOOK STYLE'''
import requests
from IPython.core.display import HTML
styles = requests.get("https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/cs109.css").text
HTML(styles)
```





<style>
blockquote { background: #AEDE94; }
h1 { 
    padding-top: 25px;
    padding-bottom: 25px;
    text-align: left; 
    padding-left: 10px;
    background-color: #DDDDDD; 
    color: black;
}
h2 { 
    padding-top: 10px;
    padding-bottom: 10px;
    text-align: left; 
    padding-left: 5px;
    background-color: #EEEEEE; 
    color: black;
}

div.exercise {
	background-color: #ffcccc;
	border-color: #E9967A; 	
	border-left: 5px solid #800080; 
	padding: 0.5em;
}
div.theme {
	background-color: #DDDDDD;
	border-color: #E9967A; 	
	border-left: 5px solid #800080; 
	padding: 0.5em;
	font-size: 18pt;
}
div.gc { 
	background-color: #AEDE94;
	border-color: #E9967A; 	 
	border-left: 5px solid #800080; 
	padding: 0.5em;
	font-size: 12pt;
}
p.q1 { 
    padding-top: 5px;
    padding-bottom: 5px;
    text-align: left; 
    padding-left: 5px;
    background-color: #EEEEEE; 
    color: black;
}
header {
   padding-top: 35px;
    padding-bottom: 35px;
    text-align: left; 
    padding-left: 10px;
    background-color: #DDDDDD; 
    color: black;
}
</style>







```python
'''IMPORT THE LIBRARIES'''
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor 
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors.kde import KernelDensity

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()
matplotlib.rcParams['figure.figsize'] = (13.0, 6.0)

from time import clock

from IPython.display import display
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 500)
sns.set_style('whitegrid')
sns.set_context('talk')

from collections import defaultdict
```


    C:\Users\david\Anaconda3\lib\site-packages\sklearn\ensemble\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d
    

## Modeling Longitudinal Component of the Data

As previously discussed, in the development of a predictive model based on medical studies, one of the key challenges is the longitudinal component of the data. In the ADNI study, patients have multiple entries in the database, each one corresponding to a different visit. One has to take into account this progression over time in order predict future data.

Let us recall the objective of the project: predict over time CDRSB score and diagnosis.

### Progression of CDRSB ver Time

We can plot the progression of the CDRSB score over time for a subset of patients.



```python
'''LOAD THE DATABASE'''
f = 'ADNIDataFiles/ADNIMERGE.csv'
df = pd.read_csv(f)
```




```python
'''CLEAN THE DATABASE'''

df_dummy = df.copy()

#Create a new cleaned df
df_clean = pd.DataFrame()

#Diagnosis
df_clean['Diagnosis'] = df_dummy.DX.replace({'Dementia':'AD'})

#CDRSB
df_clean['CDRSB'] = df_dummy.CDRSB

#Patient ID
df_clean['ID'] = pd.Categorical(df_dummy.PTID)
df_clean.ID = df_clean.ID.cat.codes

#Age at each visit
df_dummy.EXAMDATE = pd.to_datetime(df_dummy['EXAMDATE'])
df_dummy['FIRSTDATE'] = df_dummy.groupby('PTID')['EXAMDATE'].transform(min)
df_dummy['TIMESINCEFIRST'] = (df_dummy.EXAMDATE - df_dummy.FIRSTDATE).dt.days
df_clean['AGE'] = df_dummy.AGE + df_dummy.TIMESINCEFIRST/365.
df_clean['TIME'] = df_dummy.TIMESINCEFIRST/365

#Gender one-hot encoding on female
df_clean['Female'] = pd.get_dummies(df_dummy.PTGENDER)['Female']

#Education years
df_clean['Education'] = df_dummy.PTEDUCAT

#Ethicity one-hot encoding on unknown
df_clean[['Hispanic','Non_Hispanic']] = pd.get_dummies(df_dummy.PTETHCAT)[['Hisp/Latino','Not Hisp/Latino']]

#Race as categorical data
df_clean['Race'] = pd.Categorical(df_dummy.PTRACCAT)

#Race as categorical data
df_clean['Marital_status'] = pd.Categorical(df_dummy.PTMARRY)

#Gene APOE4 (Important gene with different types)
df_clean['APOE4'] = pd.Categorical(df_dummy.APOE4)

#FDG (PET scan marker)

#Create a list of predictors for which we want to calculate delta
remainingColumns = ['ADAS11', 'ADAS13', 'MMSE', 'RAVLT_immediate',
 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'FAQ', 'MOCA',
 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan',
 'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat',
 'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal',
 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal',
 'Fusiform', 'MidTemp', 'ICV']

for c in remainingColumns:
    df_clean[c] = df_dummy[c]
    
df_clean = df_clean[pd.notna(df_clean['CDRSB'])]
```




```python
'''PROGRESSION OF CDRSB'''

#Line plot CDRSB vs Age by patient
fig, ax = plt.subplots(figsize=(14,7))
grouped = df_clean.sort_values(by=['AGE']).groupby('ID')

#Select a subset of patient
n_patients = 200
for i in df_clean['ID'].sample(n_patients):
    grouped.get_group(i).plot(x='AGE',y='CDRSB', ax=ax, legend=False, alpha=0.5)

plt.xlabel('Age')
plt.ylabel('CDRSB')
plt.title('CDRSB vs. Age',fontweight='bold');
```



![png](Models_combined_files/Models_combined_9_0.png)


From the plot above, we assume linear relationship between patient age and CDRSB score. The CDRSB score over time is simply

\begin{equation}
CDRSB(t) = mt + CDRSB_0,
\end{equation}

where $t$ is time, $m$ is the slope, and $CDRSB_0$ is the baseline score.

### New Response Variable: CDRSB Slope

To remove the longitudinal component of the data, we engineer the features $m = \texttt{slope_overall}$ as well as $m_i = \texttt{slope}_i$ where $i$ correspond to the $i$-th visit. The response variable becomes $\texttt{slope_overall}$ and the CDRSB score is predicted based on the formula above. Finally, the diagnosis is predicted from the classifier discussed in the EDA

- CDRSB $\le 0$: Final diagnosis is CN
- $0 <$ CDRSB$ < 4.0$: Final diagnosis is MCI
- CDRSB$\ge 4.0$: Final diagnosis is AD

### Feature Engineering

First, we add the new features to the main dataframe. In addition to computing the overall CDRSB slope, we also compute the current slope at each increment of time for each patient.



```python
'''FEATURE ENGINEERING'''

#Sort and group to compute current slopes
grouped = df_clean.sort_values(by=['AGE']).groupby('ID')

#Initialize dictionaries of new features
slope_dict = defaultdict(dict)
base_cdrsb_dict = dict()
slope_overall_dict = dict()
final_cdrsb_dict = dict()
final_time_dict = dict()
final_diagnosis_dict = dict()

#Feature engineering
for i in df_clean['ID'].unique():
    group = grouped.get_group(i)
        
    #Save values to use for model assessment later
    base_cdrsb_dict[i] = group['CDRSB'].values[0]
    final_cdrsb_dict[i] = group['CDRSB'].values[-1]
    final_time_dict[i] = group['TIME'].values[-1]
    final_diagnosis_dict[i] = group['Diagnosis'].values[-1]
    
    #Save overall slope
    X = np.vander(group['TIME'],2)
    slope_overall = np.linalg.lstsq(X,group['CDRSB'])[0][0]
    slope_overall_dict[i] = slope_overall
    
    #Save incremental slope
    for j in range(2,group.shape[0]+1):
        group_upToVisit = group.head(j)
        visit = round(group_upToVisit['TIME'].values[-1]*2)/2
        X_upToVisit = np.vander(group_upToVisit['TIME'],2)
        slope_upToVisit = np.linalg.lstsq(X_upToVisit,group_upToVisit['CDRSB'])[0][0]
        slope_dict[visit][i] = slope_upToVisit
    
#Set of all possible visit intervals observed
visits = sorted([key for key in slope_dict.keys() if key != 'default'])

#Initialize new columns to NaNs before filling them
df_clean["base_cdrsb"] = np.nan
df_clean["slope_overall"] = np.nan
df_clean["final_cdrsb"] = np.nan
df_clean["final_diagnosis"] = None
df_clean["final_time"] = np.nan
for v in visits:
    col_name = "slope_{:0.1f}".format(v)
    df_clean[col_name] = np.nan

#Add new columns to df_clean
for i,row in df_clean.iterrows():
    time_round = round(df_clean.loc[i,'TIME']*2)/2.
    df_clean.loc[i,'base_cdrsb'] = base_cdrsb_dict[row['ID']]
    df_clean.loc[i,'slope_overall'] = slope_overall_dict[row['ID']]
    df_clean.loc[i,'final_cdrsb'] = final_cdrsb_dict[row['ID']]
    df_clean.loc[i,'final_diagnosis'] = final_diagnosis_dict[row['ID']]
    df_clean.loc[i,'final_time'] = final_time_dict[row['ID']]
    for v in visits:
        if row['ID'] in slope_dict[v] and time_round == v:
            col_name = "slope_{:0.1f}".format(v)
            df_clean.loc[i,col_name]= slope_dict[v][row['ID']]
```


    C:\Users\david\Anaconda3\lib\site-packages\ipykernel_launcher.py:26: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
    To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
    C:\Users\david\Anaconda3\lib\site-packages\ipykernel_launcher.py:34: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
    To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
    

### Dictionary of Dataframes

!!!! EXPLAIN BETTER THE CURRENT SLOPE DFS? !!!!!!

A critical objective of the final model is early detection. Ideally, the model would predict with high accuracy the CDRSB score and diagnosis over time after the first few visits and with the least expensive feature subset. Therefore, we built a table of dataframes to explore the impact of (1) number of visits considered and (2) feature subset selection. The table of dataframes has dimensions $m \times n$ where $m$ is equal to $23$ (maximum number of visits of the study with increments of about $6$ months) and $n$ is equal to $4$ (different subsets of feature) that we split in the following way:

1. Demographics + Cognitive Tests
2. Demographics + Cognitive Tests + Ecog Tests
3. Demographics + Cognitive Tests + Imaging Data
4. Demographics + Cognitive Tests + Ecog Test + Imaging Data

Let us prepare the split of the different feature subsets.



```python
'''FEATURE SUBSETS'''

imaging_columns = ['ICV','MidTemp','Fusiform','Entorhinal','WholeBrain','Hippocampus','Ventricles'] #removed FDG!
ecog_columns = ['EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan',
                'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem',
                'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan', 'EcogSPOrgan',
                'EcogSPDivatt', 'EcogSPTotal']
final_columns = ['slope_overall','final_cdrsb','final_diagnosis','final_time']
slope_columns = []
for v in visits:
    col_name = "slope_{:0.1f}".format(v)
    slope_columns.append(col_name)
standard_columns = list(set(df_clean.columns.values) - set(imaging_columns) - set(ecog_columns) \
                        - set(slope_columns) - set(final_columns))

subsets = []
subsets.append(standard_columns + final_columns)
subsets.append(standard_columns + final_columns + ecog_columns)
subsets.append(standard_columns + final_columns + imaging_columns)
subsets.append(standard_columns + final_columns + ecog_columns + imaging_columns)
```


We can now create the dictionary of dataframes to explore the impact of the time since the first visit and the feature selection on the efficiency of the model to predict future CDRSB score and diagnosis.



```python
'''DICTIONARY OF DFS'''

dfs = defaultdict(dict)
visits_new = np.linspace(0.,11.,23)
n_subsets = len(subsets)
for v in visits_new:
    col_name = "slope_{:0.1f}".format(v)
    for s in range(n_subsets):
        if v > 0:
            dfs[s][v] = df_clean[subsets[s]+[col_name]].dropna().sort_values('ID')
        else:
            dfs[s][v] = df_clean[subsets[s]].dropna().groupby('ID',as_index=False).first().sort_values('ID')
```


## Predicting Future CDRSB Score and Diagnosis

### Function Implementations

To assess the performance of the different models to predict the slope of the CDRSB, the future CDRSB score, and the future diagnosis, we implemented the following functions.



```python
def run_model(X,y,assessment,seed=40,size=0.3,kernel_opt = False,weight=0,max_depth = 3,bandwidth=0.5,n_estimators=10):
    ''' 
    **Train model with RF regressor on the CDRSB slope and return different scores
    * param[in] X predictor matrix
    * param[in] y response variable (CDRSB slope)
    * param[in] assessment df to save predictors not used for regression
    * param[in] seed random seed of rf and train-test-split
    * param[in] size test/(train+test)
    * param[in] weight hyper-parameter exponent to apply to regression weights
    * param[in] kernel_opt bool to activate/deactivate the kernel weighting penalty
    * param[in] max_depth hyper-parameter of the rf tree max depth
    * param[in] bandwidth hyper-parameter of the kernel distribution bandwidth
    * param[in] n_estimator hyper-parameter of the number of trees used for rf
    * param[out] model rf fitted model
    * param[out] training_score test_score R2 on the rf with CDRSB slope as response variable
    * param[out] cdrsb_error_train,cdrsb_error_test CDRSB mse between true value and compute with linear slope
    * param[out] diagnosis_error_train,diagnosis_error_test diagnosis accuracy based on CDRSB classification
    '''
    
    #Stratify the database for current diagnoses
    df_stratify = pd.DataFrame()
    df_stratify[['Diagnosis_CN','Diagnosis_MCI']] = X[['Diagnosis_CN','Diagnosis_MCI']]

    #Split train and test
    X_train, X_test, y_train, y_test, assessment_train, assessment_test =\
                        train_test_split(X, y, assessment, test_size=size, random_state=seed,\
                                        stratify=df_stratify[['Diagnosis_CN','Diagnosis_MCI']])
    
    #Generate kernel density function to apply to weight
    if kernel_opt:
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(assessment_train['final_time'].values.reshape(-1,1))
        kde_score = np.exp(kde.score_samples(assessment_train['final_time'].values.reshape(-1,1)))
    else:
        kde_score = np.ones(assessment_train.shape[0])
    
    #Fit Random Forest with weight
    if weight > 0:
        w = [0. if f == 0 else (1-t/f)**weight/k\
             for f,t,k in zip(assessment_train['final_time'],\
            assessment_train['TIME'],kde_score)]      
        model =\
        RandomForestRegressor(n_estimators=n_estimators,random_state=seed,max_depth=max_depth,n_jobs=4).fit(X_train, y_train, sample_weight=w)
    else:
        model =\
       RandomForestRegressor(n_estimators=n_estimators,random_state=seed,max_depth=max_depth,n_jobs=4).fit(X_train, y_train)
    
    #Get train and test scores
    training_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    #Get CDRSB error
    cdrsb_pred_train = \
    [i+s*t for i,s,t in zip(assessment_train['base_cdrsb'],model.predict(X_train),assessment_train['final_time'])]
    cdrsb_pred_test  = \
    [i+s*t for i,s,t in zip(assessment_test['base_cdrsb'],model.predict(X_test),assessment_test['final_time'])]
    cdrsb_error_train = mean_squared_error(assessment_train['final_cdrsb'],cdrsb_pred_train)
    cdrsb_error_test  = mean_squared_error(assessment_test['final_cdrsb'],cdrsb_pred_test)

    #Get diagnosis accuracy score
    diagnosis_pred_train = ['CN' if i <0.5 else 'MCI' if i < 2.5 else 'AD' for i in cdrsb_pred_train]
    diagnosis_pred_test = ['CN' if i  <0.5 else 'MCI' if i < 2.5 else 'AD' for i in cdrsb_pred_test]
    diagnosis_error_train = accuracy_score(assessment_train['final_diagnosis'],diagnosis_pred_train)
    diagnosis_error_test  = accuracy_score(assessment_test['final_diagnosis'],diagnosis_pred_test)
                                 
    return model,training_score,test_score,cdrsb_error_train,cdrsb_error_test,diagnosis_error_train,diagnosis_error_test

def GridSearch(max_depths,weights,bandwidths,n_estimators,subset,year):
    ''' 
    **Hyper-parameters tuning through gridsearch
    * param[in] max_depths array of hyper-parameters of the rf tree max depth
    * param[in] weights array of  hyper-parameters exponent to apply to regression weights
    * param[in] bandwidths array of hyper-parameters of the kernel distribution bandwidth
    * param[in] n_estimators array of  hyper-parameters of the number of trees used for rf
    * param[in] subset integer representing the selected feature subset (0,1,2 or 3)
    * param[in] year float representing the selected year (0,0.5,...10.5,11)
    * param[out] best_cdrsb_error CDRSB mse between true value and compute with linear slope for best hyper-params
    * param[out] best_diagnosis_accuracy diagnosis accuracy based on CDRSB classification for best hyper-params
    * param[out] best_R2 test R2 score on the rf with CDRSB slope as response variable for best hyper-params
    * param[out] best_params list of best hyper param (order is same as inputs)
    '''
    best_cdrsb_error = np.inf
    best_diagnosis_accuracy = 0
    best_R2 = 0
    best_params = []
    for max_depth in max_depths:
        for weight in weights:
            for bandwidth in bandwidths:
                for n_estimator in n_estimators:
                    _,_,R2,_,cdrsb_error_test,_,diagnosis_accuracy=\
                    run_model(Xs[subset][year],ys[subset][year],assessments[subset][year],\
                            seed=40,weight=weight,n_estimators=n_estimator,max_depth=max_depth)
                    if cdrsb_error_test < best_cdrsb_error:
                        best_cdrsb_error = cdrsb_error_test
                        best_diagnosis_accuracy = diagnosis_accuracy
                        best_R2 = R2
                        best_params = [max_depth,weight,bandwidth,n_estimator]
                        
    return best_cdrsb_error, best_diagnosis_accuracy,best_R2,best_params

def plotModelPerformance(ax,title,years,train_score,test_score,mse_train,mse_test,acc_train,acc_test):
    ''' 
    **Plot model performance (R2 of CDRSB slope, MSE of future CDRSB, and accuracy of diagnosis)
      as a function of the number of years for a fixed feature subset
    * param[in] ax array of axis handle
    * param[in] title of the subplot
    * param[in] years array of time since first visit in year
    * param[in] train_score, test_score array of R2 scores of the CDRSB slope for rf
    * param[in] mse_train, mse_test array of mse errors on the final CDRSB
    * param[in] acc_train, acc_test array of accuracies of the final diagnosis
    * param[out] ax array of axis handle
    '''
    
    #Plot the train and test score vs time
    ax[0].plot(years,train_score,'b-',label='train')
    ax[0].plot(years,test_score,'r-',label='test')
    ax[0].set_xlabel('Time [years]')
    ax[0].set_ylabel(r'$R^2$')
    ax[0].set_title(r'$R^2$ Slope ('+title+')')
    ax[0].set_xlim(0,10)
    ax[0].set_ylim(0.2,1)
    ax[0].legend()
    
    #Plot the train and test mse vs number of visits
    ax[1].plot(years,cdrsb_error_train,'b-',label='train')
    ax[1].plot(years,cdrsb_error_test,'r-',label='test')
    ax[1].set_xlabel('Time [years]')
    ax[1].set_ylabel('MSE')
    ax[1].set_title(r'MSE CDRSB ('+title+')')
    ax[1].set_xlim(0,10)
    ax[1].set_ylim(0,10)
    ax[1].legend()
    
    #Plot the train and test accuarcy vs number of visits
    ax[2].plot(years,diagnosis_error_train,'b-',label='train')
    ax[2].plot(years,diagnosis_error_test,'r-',label='test')
    ax[2].set_xlabel('Time [years]')
    ax[2].set_ylabel('Accuracy')
    ax[2].set_title('Accuracy ('+title+')')
    ax[2].set_xlim(0,10)
    ax[2].set_ylim(0.4,1.0)
    ax[2].legend()
    
    return ax

def plotHeatMap(mat,ax,title,reverse=False):
    ''' 
    **Plot heat maps
    * param[in] matrix of the heatmap
    * param[in] ax axis handle
    * param[in] title of the heatmap
    * param[in] reverse option to reverse the heatmap
    * param[out] ax axis handle
    '''
    #Properties of the heatmap
    cmap = 'winter'
    if reverse:
        cmap = cmap + "_r"
        
    #Declare years
    years = [0,0.5]+list(np.linspace(1,10,10))
    
    #Set axis handle
    ax.imshow(mat,cmap=plt.get_cmap(cmap))
    ax.grid(False)
    ax.set_title(title)

    # Show all ticks
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(12))
    
    # Label ticks
    ax.set_yticklabels(["{} years".format(y) for y in years])
    ax.set_xticklabels(["Subset {}".format(i) for i in range(4)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(4):
        for j in range(12):
            text = ax.text(i, j, "{:0.3f}".format(mat[j,i]),
                           ha="center", va="center", color="w",fontsize=12)
```


### Prepare data for training and testing

To prepare the data for training and testing, we have to create the predictor matrices $Xs$ as well as the response variable vectors $ys$ corresponding to the different dataframes. We also need to one-hot encode the predictor matrices and to save important predictors for models assesssment, e.g. $\texttt{base_cdrsb}$, $\texttt{final_cdrsb}$, $\texttt{final_diagnosis}$, $\texttt{final_time}$, $\texttt{TIME}$, $\texttt{ID}$.



```python
n_subset = len(dfs)
n_visits = len(dfs[0])
Xs = defaultdict(dict)
ys = defaultdict(dict)
assessments = defaultdict(dict)
for i in range(n_subset):
    for j in range(n_visits):
        v  = 0.5*j
        ys[i][v] = dfs[i][v]['slope_overall']
        assessments[i][v] = dfs[i][v][['base_cdrsb','final_cdrsb','final_diagnosis','final_time','TIME','ID']]
        temp_df = dfs[i][v].drop(['final_cdrsb','final_diagnosis','final_time','slope_overall','base_cdrsb','TIME','ID'], axis=1)
        Xs[i][v] = pd.get_dummies(temp_df,drop_first=True)
```


### Baseline Model

We first start investigating how well we can predict the CDRSB slope, the future CDRSB, and the future diagnosis with the original dataframes without hyper-parameter tuning. We can first plot the relation between the model scores ($R^2$ on the CDRSB slope, MSE on the final CDRSB, and accuracy on the final diagnosis) as a function of the number of visits for different feature subsets.



```python
''''BASELINE MODELS'''

#Loop over time and the subsets
n_years = [10,10,8,7]
subsets = [0,1,2,3]

#Set up the figure
fig = plt.figure(figsize=(15,10))
axs = []
axs_subs = []
for i in range(len(subsets)):
    axs.append(fig.add_subplot(3,4,3*i+1))
    axs.append(fig.add_subplot(3,4,3*i+2))
    axs.append(fig.add_subplot(3,4,3*i+3))

axs_subs.append([axs[0],axs[4],axs[8]])
axs_subs.append([axs[1],axs[5],axs[9]])
axs_subs.append([axs[2],axs[6],axs[10]])
axs_subs.append([axs[3],axs[7],axs[11]])

#Run model and plot figure
for f in subsets:
    years = [0,0.5]+list(np.linspace(1,n_years[f],n_years[f]))
    test_score = np.zeros(len(years))
    train_score = np.zeros(len(years))
    cdrsb_error_train = np.zeros(len(years))
    cdrsb_error_test = np.zeros(len(years))
    diagnosis_error_train = np.zeros(len(years))
    diagnosis_error_test = np.zeros(len(years))
    for i,s in enumerate(years):
        a,train_score[i],test_score[i],cdrsb_error_train[i],cdrsb_error_test[i],\
        diagnosis_error_train[i],diagnosis_error_test[i]=\
        run_model(Xs[f][s],ys[f][s],assessments[f][s])
    title = 'subset'+str(f)
    axs_subs[f] = plotModelPerformance(axs_subs[f],title,years,train_score,test_score,cdrsb_error_train,\
                               cdrsb_error_test,diagnosis_error_train,diagnosis_error_test)
plt.tight_layout()
plt.suptitle('Baseline Models Based on Time and Subset Selection',fontsize='20')
plt.subplots_adjust(top=0.9);
```



![png](Models_combined_files/Models_combined_30_0.png)


In the Figure above, the model scores only go up until $10$ years for subsets $0$ and $1$ and up until $8$ and $7$ years for subsets $2$ and $3$ due to missing data.

!!!!!!!!!!!!!!!!!!!SOME ANALYSIS IS MISSING HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

In additon, we can plot the heat map of the model scores as a function of the selected subset and the time in years.



```python
'''BASELINE HEATMAP'''

#Declare properties to map
cdrsb_error_mat = np.zeros((12,4))
R2_mat = np.zeros((12,4))
diagnosis_accuracy_mat = np.zeros((12,4))
years = [0,0.5]+list(np.linspace(1,10,10))

#Compute property matrices
for i in range(4):
    for j,y in enumerate(years):
        numNonFinalPatients = sum((assessments[i][y]['TIME'] - assessments[i][y]['final_time'])!=0)
        if numNonFinalPatients > 10:
            _,_,R2_mat[j][i],_,cdrsb_error_mat[j][i],\
            _,diagnosis_accuracy_mat[j][i]=\
            run_model(Xs[i][y],ys[i][y],assessments[i][y])
        else:
            cdrsb_error_mat[j][i] = np.NaN
            R2_mat[j][i] = np.NaN
            diagnosis_accuracy_mat[j][i] = np.NaN
            
#Plot heatmaps
fig,axes = plt.subplots(ncols=3,nrows=1,figsize=(15,10))
plotHeatMap(cdrsb_error_mat,axes[0],'CDRSB Mean-Squared-Error',reverse=True)
plotHeatMap(R2_mat,axes[1],'$R^2$ of CDRSB Slope')
plotHeatMap(diagnosis_accuracy_mat,axes[2],'Final Diagnosis Accuracy')
plt.suptitle('Baseline Heatmaps',fontsize=20);
```



![png](Models_combined_files/Models_combined_33_0.png)


!!!!!!!!!!!!!!!!!!!SOME ANALYSIS IS MISSING HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

### Improving Baseline Model: Adding Sample Weight

In order to improve the performance of the baseline model, we have to account first for the high correlation between the response variable overall CDRSB-slope and the predictor current CDRSB-slope when the patient is approaching his final visit. In fact, at the final visit, the predictor current CDRSB-slope will be exactly equal to the response variable overall  CDRSB-slope. This will lead to massive over-fitting on the training set.

Therefore, we introduce sample weight to the random forest model. The array of sample weights are modeled as follows:

$$ w_i = \left(1 - \frac{t_i}{f_i}\right)^p,$$
 
where $w_i$ corresponds to the $i^{th}$ weight to apply to the train response variable, $t_i$ is the current time of the patient's visit, $f_i$ is the final time of the patient's history, and $p$ is a power. The sample weight varies from $0$ when $t_i = f_i$ to $1$ when $t_i = 0$. The exponent $p$ becomes an hyper-parameter that we will explore in a later section.

Another problem we have to deal with is the discrepancy between patients' history. The Figure below highlights a big drop in the number of patients involved in the study after $4$ years. We also overlay on the histogram the estimae kernel density function.



```python
'''HISTOGRAME AND ESTIMATE KERNEL OF FINAL TIME'''

#Kernel density
kde = KernelDensity(kernel='gaussian', bandwidth=.5).fit(assessments[2][0.]['final_time'].values.reshape(-1,1))
t = np.linspace(0,10,100)
tt = np.exp(kde.score_samples(t.reshape(-1,1)))

#Plot hist. + kernel
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(assessments[0][0.]['final_time'],density=True)
ax.set_xlabel('Final Time [year]')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Final Time');
ax.plot(t,tt,'-',label='Kernel Estimate')
ax.legend();
```



![png](Models_combined_files/Models_combined_37_0.png)


A way to deal with this issue is to apply sample weight based on the kernel density estimate. To do this, we update our sample weight function to

$$ w_i = \frac{\left(1 - \frac{t_i}{f_i}\right)^p}{k(f_i)},$$

where $k(f_i)$ is the value of the kernel estimate varying with the final time of the $i^{th}$ observation in the dataframe. This introduces a new hyper-parameter $b$ corresponding to the bandwidth of the kernel density function.

The sample weight adjustments are implemented in the function $\texttt{run_model}$. Similar to the baseline models, we can plot the heatmaps of the model performance for fixed hyper-parameters. We choose a weight power of $p = 2$ and a bandwidth of $k=2$.



```python
'''HEATMAPS OF THE WEIGHTED MODELS'''

#Declare properties to map
cdrsb_error_mat = np.zeros((12,4))
R2_mat = np.zeros((12,4))
diagnosis_accuracy_mat = np.zeros((12,4))
years = [0,0.5]+list(np.linspace(1,10,10))

#Hyper-parameters
p = 2
b = 0.5

#Compute property matrices
for i in range(4):
    for j,y in enumerate(years):
        numNonFinalPatients = sum((assessments[i][y]['TIME'] - assessments[i][y]['final_time'])!=0)
        if numNonFinalPatients > 10:
            _,_,R2_mat[j][i],_,cdrsb_error_mat[j][i],\
            _,diagnosis_accuracy_mat[j][i]=\
            run_model(Xs[i][y],ys[i][y],assessments[i][y],weight=p,bandwidth=b)
        else:
            cdrsb_error_mat[j][i] = np.NaN
            R2_mat[j][i] = np.NaN
            diagnosis_accuracy_mat[j][i] = np.NaN
            
#Plot heatmaps
fig,axes = plt.subplots(ncols=3,nrows=1,figsize=(15,10))
plotHeatMap(cdrsb_error_mat,axes[0],'CDRSB Mean-Squared-Error',reverse=True)
plotHeatMap(R2_mat,axes[1],'$R^2$ of CDRSB Slope')
plotHeatMap(diagnosis_accuracy_mat,axes[2],'Final Diagnosis Accuracy')
plt.suptitle('Heatmaps of Weighted Models',fontsize=20);
```



![png](Models_combined_files/Models_combined_39_0.png)


### Improving Baseline Model: Hyper-Parameter Tuning

To further improve the model perofmance, we perform hyper-parameter tuning. We will tune $4$ different hyper-parameters:

1. Random Forest Tree Maximum Depth (2,5,10,20)
2. Weight Power (1,2,3)
3. Kernel Estimate Bandwidth (0.25,0.5,0.75)
4. Random Forest Number of Trees (10,50,100)

To perform the hyper-parameter tuning, we have implemented the function $\texttt{GridSearch}$.



```python
'''HYPER-PARAMETERS TUNING'''

#Declare bound of hp
max_depths = [2,5,10,20]
weights = [1,2,3]
bandwidths = [0.25, 0.5, 0.75]
n_estimators = [10,50,100]

#Declare outputs of the search
best_cdrsb_error = defaultdict(dict)
best_diagnosis_accuracy = defaultdict(dict)
best_R2 = defaultdict(dict)
best_params = defaultdict(dict)

#Number of years of the study
years = [0,0.5]+list(np.linspace(1,10,10))

#Perform grid search
t0 = clock()
for subset in range(4):
    for y in years:
        numNonFinalPatients = sum((assessments[subset][y]['TIME'] - assessments[subset][y]['final_time'])!=0)
        if  numNonFinalPatients > 10:
            best_cdrsb_error[subset][y], best_diagnosis_accuracy[subset][y], \
            best_R2[subset][y], best_params[subset][y] = \
            GridSearch(max_depths,weights,bandwidths,n_estimators,subset,y)
            print((subset,y,clock()-t0,'Successful'))
        else:
            best_cdrsb_error[subset][y] = np.NaN
            best_diagnosis_accuracy[subset][y] = np.NaN
            best_R2[subset][y] = np.NaN
            best_params[subset][y] = np.NaN
            print((subset,y,clock()-t0,'Too few people'))
```




```python
'''SAVE ENVIRONMENT'''

#Grid search is long so we save results ...
import pickle
with open('grid_search.pkl','wb') as f:
    pickle.dump([best_cdrsb_error,best_diagnosis_accuracy,best_R2,best_params],f)
```


We can plot the heatmaps of the model performance for tuned hyper-parameters.



```python
'''HEATMAPS OF THE TUNED MODELS'''

#Declare properties to map
cdrsb_error_mat = np.zeros((12,4))
R2_mat = np.zeros((12,4))
diagnosis_accuracy_mat = np.zeros((12,4))

#Compute property matrices
for i in range(4):
    for j,v in enumerate(visits):
        cdrsb_error_mat[j,i] = best_cdrsb_error[i][v]
        R2_mat[j,i] = best_R2[i][v]
        diagnosis_accuracy_mat[j,i] = best_diagnosis_accuracy[i][v]

#Plot heatmaps
fig,axes = plt.subplots(ncols=3,nrows=1,figsize=(15,10))
plotHeatMap(cdrsb_error_mat,axes[0],'CDRSB Mean-Squared-Error',reverse=True)
plotHeatMap(R2_mat,axes[1],'$R^2$ of CDRSB Slope')
plotHeatMap(diagnosis_accuracy_mat,axes[2],'Final Diagnosis Accuracy')
plt.suptitle('Tuned Model Heatmaps',fontsize=20);
```


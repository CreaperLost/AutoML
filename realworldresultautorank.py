import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table


res=pd.read_csv('Real-World-Results.csv',sep=';',na_values='?')
res.fillna(0,inplace=True)
print(res.head())

#Without feature selection.
no_feature_selection_res = pd.DataFrame()
for i in res.columns:
    if 'no_indicators_and_No_Feature_Selection' == i or 'Indicator_Variables_and_No_Feature_Selection' == i:
        continue
    if 'No_Feature_Selection' in i:
        no_feature_selection_res = pd.concat([no_feature_selection_res,res[i]],axis=1)
print(no_feature_selection_res)
res_no_Ft=pd.DataFrame(no_feature_selection_res)
res_no_Ft.columns=['GAIN','MF','MM','PPCA','SOFT','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT']
print(res_no_Ft)
result = autorank(res_no_Ft, alpha=0.05, verbose=False)
print(result)
plot_stats(result)
plt.show()

#With feature selection.
feature_selection_res = pd.DataFrame()
for i in res.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res = pd.concat([feature_selection_res,res[i]],axis=1)
res_Ft=pd.DataFrame(feature_selection_res)
res_Ft.columns=['GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
print(res_Ft)
result = autorank(res_Ft, alpha=0.05, verbose=False)
print(result)
plot_stats(result,allow_insignificant=True)
plt.show()

"""
#With feature selection.
overall_res = pd.DataFrame()
for i in res.columns:
    if 'ppca' == i or 'softimp' == i or 'missforest' == i or 'Gain_Imputer' or 'meanmode' == i:
        feature_selection_res = pd.concat([feature_selection_res,res[i]],axis=1)
res_Ft=pd.DataFrame(feature_selection_res,columns=['GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF'])
print(res_Ft)
result = autorank(res_Ft, alpha=0.05, verbose=False)
print(result)
plot_stats(result,allow_insignificant=True)
plt.show()"""
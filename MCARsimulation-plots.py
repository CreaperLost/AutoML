import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
import matplotlib.pyplot as plt
import numpy as np


res=pd.read_csv('MCAR-Results.csv',sep=';',na_values='?')
res_complete = pd.read_csv('Complete-Results.csv',sep=';',na_values='?')
res_complete['Analysis']=res_complete['Analysis'].str.replace('jad_',"").str.replace('_outcome','')


#colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black']
colors = ['tab:blue','Lime','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black']

medianprops_normal = dict(linestyle='-.', linewidth=2.5,color='white')
medianprops_diff = dict(linestyle='-.', linewidth=2.5,color='black')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20
Y_AXIS_SIZE = 24
ULTRA_SIZE = 28


plt.rc('font', size=ULTRA_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=ULTRA_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=ULTRA_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=Y_AXIS_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=Y_AXIS_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#Without feature selection.
no_feature_selection_res = pd.DataFrame(res['Analysis'])
no_feature_selection_res = pd.concat([no_feature_selection_res,res['nofs']],axis=1)
for i in res.columns:
    if 'no_indicators_and_No_Feature_Selection' == i or 'Indicator_Variables_and_No_Feature_Selection' == i:
        continue
    if 'No_Feature_Selection' in i:
        no_feature_selection_res = pd.concat([no_feature_selection_res,res[i]],axis=1)

res_no_Ft=pd.DataFrame(no_feature_selection_res)
res_no_Ft.columns=['Analysis','Best','GAIN','MF','MM','PPCA','SOFT','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT']
res_no_Ft.dropna(axis=1,how='all',inplace=True)
row_id=list(res_no_Ft['Analysis'])
list_10 =  [x for x in [row_id.index(i) if '10' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_25 =  [x for x in [row_id.index(i) if '25' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_50 =  [x for x in [row_id.index(i) if '50' in i else np.nan for i in  row_id] if ~np.isnan(x)]

res_no_Ft_tradeoff = res_no_Ft


"""
fig, ax = plt.subplots()
plt.title('AUC Boxplot for MCAR case when feature selection is not considered')
boxplots = list()
j=0
i=0
res_no_Ft = res_no_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]
for x in res_no_Ft.columns:
    data =  list([np.array(res_no_Ft.loc[list_10][x].dropna(axis=0)),np.array(res_no_Ft.loc[list_25][x].dropna(axis=0)),np.array(res_no_Ft.loc[list_50][x].dropna(axis=0))])
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[0.5+j,2.7+j,4.9+j],widths = 0.15,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_normal))
    j+= 0.18
    i+=1
plt.xlim(0,9)
locs, labels = plt.xticks()
plt.xticks([1.5, 3.7, 5.9],['10%','25%','50%'])
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_no_Ft.columns], loc='lower right')
plt.ylabel('AUC')
plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()
"""




# AUC difference in MCAR case when feature selection is excluded

res_no_Ft_tradeoff.index = res_no_Ft_tradeoff['Analysis']
res_no_Ft_tradeoff.drop('Analysis',axis=1 , inplace=True)
res_complete.index = res_complete['Analysis']
res_complete.drop('Analysis',axis=1 , inplace=True)

for ind in res_no_Ft_tradeoff.index:
    for indx in res_complete.index:
        if indx in ind:
            res_no_Ft_tradeoff.loc[ind] = res_no_Ft_tradeoff.loc[ind].subtract(res_complete.loc[indx]['nofs'])
            break

fig, ax = plt.subplots()
plt.title('AUC Difference for MCAR case when feature selection is excluded')
boxplots = list()
j=0
i=0
res_no_Ft_tradeoff.reset_index(inplace=True)
res_no_Ft_tradeoff = res_no_Ft_tradeoff[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]
print(res_no_Ft_tradeoff)
for x in res_no_Ft_tradeoff.columns:
    data =  list([np.array(res_no_Ft_tradeoff.loc[list_10][x].dropna(axis=0)),np.array(res_no_Ft_tradeoff.loc[list_25][x].dropna(axis=0)),np.array(res_no_Ft_tradeoff.loc[list_50][x].dropna(axis=0))])
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[0.5+j,2.7+j,4.9+j],widths = 0.15,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_normal))
    j+= 0.18
    i+=1
plt.xlim(0,9)
locs, labels = plt.xticks()
plt.xticks([1.5, 3.7, 5.9],['10%','25%','50%'])
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_no_Ft_tradeoff.columns], loc='lower right')
plt.ylabel('AUC')
#plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()






#With feature selection.
feature_selection_res = pd.DataFrame(res['Analysis'])
feature_selection_res = pd.concat([feature_selection_res,res['fs']],axis=1)
for i in res.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res = pd.concat([feature_selection_res,res[i]],axis=1)

res_Ft=pd.DataFrame(feature_selection_res)
res_Ft.columns=['Analysis','Best','GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
res_Ft.dropna(axis=1,how='all',inplace=True)
row_id  =  list(res_Ft['Analysis'])
list_10 =  [x for x in [row_id.index(i) if '10' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_25 =  [x for x in [row_id.index(i) if '25' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_50 =  [x for x in [row_id.index(i) if '50' in i else np.nan for i in  row_id] if ~np.isnan(x)]


res_ft_tradeoff = res_Ft

"""
fig, ax = plt.subplots()
plt.title('AUC Boxplot for MCAR case when feature selection is enforced')
boxplots = list()
j=0
i=0
res_Ft = res_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]
for x in res_Ft.columns:
    data =  list([np.array(res_Ft.loc[list_10][x].dropna(axis=0)),np.array(res_Ft.loc[list_25][x].dropna(axis=0)),np.array(res_Ft.loc[list_50][x].dropna(axis=0))])
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[0.5+j,2.7+j,4.9+j],widths = 0.15,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_normal))
    j+= 0.18
    i+=1

plt.xlim(0,9)
locs, labels = plt.xticks()
plt.xticks([1.5, 3.7, 5.9],['10%','25%','50%'])
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_Ft.columns], loc='lower right')
plt.ylabel('AUC')
plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()
"""



res_ft_tradeoff.index = res_ft_tradeoff['Analysis']
res_ft_tradeoff.drop('Analysis',axis=1 , inplace=True)

for ind in res_ft_tradeoff.index:
    for indx in res_complete.index:
        if indx in ind:
            res_ft_tradeoff.loc[ind] = res_ft_tradeoff.loc[ind].subtract(res_complete.loc[indx]['fs'])
            break

fig, ax = plt.subplots()
plt.title('AUC difference for MCAR case when feature selection is enforced')
boxplots = list()
j=0
i=0
print(res_ft_tradeoff)
res_ft_tradeoff.reset_index(inplace=True)
res_ft_tradeoff = res_ft_tradeoff[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]

for x in res_ft_tradeoff.columns:
    data =  list([np.array(res_ft_tradeoff.loc[list_10][x].dropna(axis=0)),np.array(res_ft_tradeoff.loc[list_25][x].dropna(axis=0)),np.array(res_ft_tradeoff.loc[list_50][x].dropna(axis=0))])
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[0.5+j,2.7+j,4.9+j],widths = 0.15,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_normal))
    j+= 0.18
    i+=1
plt.xlim(0,9)
locs, labels = plt.xticks()
plt.xticks([1.5, 3.7, 5.9],['10%','25%','50%'])
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_ft_tradeoff.columns], loc='lower right')
plt.ylabel('AUC')
#plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()

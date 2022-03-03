import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
import numpy as np

res=pd.read_csv('Real-World-Results.csv',sep=';',na_values='?')

colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black']
medianprops_normal = dict(linestyle='-.', linewidth=2.5,color='white')
medianprops_diff = dict(linestyle='-.', linewidth=2.5,color='black')

colors = ['tab:blue','Lime','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black']

def x(a,b):
    return a - b

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
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



#Without feature selection.
no_feature_selection_res = pd.DataFrame()
no_feature_selection_res = pd.concat([no_feature_selection_res,res['nofs']],axis=1)
for i in res.columns:
    if 'no_indicators_and_No_Feature_Selection' == i or 'Indicator_Variables_and_No_Feature_Selection' == i:
        continue
    if 'No_Feature_Selection' in i:
        no_feature_selection_res = pd.concat([no_feature_selection_res,res[i]],axis=1)

res_no_Ft=pd.DataFrame(no_feature_selection_res)
res_no_Ft.columns=['Best','GAIN','MF','MM','PPCA','SOFT','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT']
res_no_Ft.dropna(axis=1,how='all',inplace=True)
res_no_Ft = res_no_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]

feature_selection_res = pd.DataFrame()
feature_selection_res = pd.concat([feature_selection_res,res['fs']],axis=1)
for i in res.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res = pd.concat([feature_selection_res,res[i]],axis=1)
res_Ft=pd.DataFrame(feature_selection_res)
res_Ft.columns=['Best','GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
res_Ft.dropna(axis=1,how='all',inplace=True)
res_Ft = res_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]

for i in res_Ft.columns:
    res_Ft[i]=res_Ft[i].subtract(res_no_Ft[i],axis=0)
print(res_Ft)



fig, ax = plt.subplots()
boxplots = list()
i=0
pos = 0.2
for x in res_Ft.columns:
    data = list([np.array(res_Ft[x].dropna(axis=0))])    #print(data)
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[pos],widths = 0.26,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_diff))
    i+=1
    pos += 0.28
plt.xlim(0,4)
locs, labels = plt.xticks()
#plt.xticks([3],['Imputation method'])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('AUC_Diff = AUC_fs - Auc_no-fs')
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_Ft.columns],loc='lower right') #, 
# Multiple box plots on one Axes
#plt.tight_layout()
plt.title('AUC Difference from excluding feature selection')
plt.rcParams["savefig.bbox"] = "tight"
plt.show()




#MCAR case

res=pd.read_csv('MCAR-results.csv',sep=';',na_values='?')



#Without feature selection.
no_feature_selection_res = pd.DataFrame()
no_feature_selection_res = pd.concat([no_feature_selection_res,res['nofs']],axis=1)
for i in res.columns:
    if 'no_indicators_and_No_Feature_Selection' == i or 'Indicator_Variables_and_No_Feature_Selection' == i:
        continue
    if 'No_Feature_Selection' in i:
        no_feature_selection_res = pd.concat([no_feature_selection_res,res[i]],axis=1)

res_no_Ft=pd.DataFrame(no_feature_selection_res)
res_no_Ft.columns=['Best','GAIN','MF','MM','PPCA','SOFT','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT']
res_no_Ft.dropna(axis=1,how='all',inplace=True)
res_no_Ft = res_no_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]

feature_selection_res = pd.DataFrame()
feature_selection_res = pd.concat([feature_selection_res,res['fs']],axis=1)
for i in res.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res = pd.concat([feature_selection_res,res[i]],axis=1)
res_Ft=pd.DataFrame(feature_selection_res)
res_Ft.columns=['Best','GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
res_Ft.dropna(axis=1,how='all',inplace=True)
res_Ft = res_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]

for i in res_Ft.columns:
    res_Ft[i]=res_Ft[i].subtract(res_no_Ft[i],axis=0)
print(res_Ft)



fig, ax = plt.subplots()
boxplots = list()
i=0
pos = 0.2
for x in res_Ft.columns:
    data = list([np.array(res_Ft[x].dropna(axis=0))])    #print(data)
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[pos],widths = 0.26,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_diff))
    i+=1
    pos += 0.28
plt.xlim(0,4)
locs, labels = plt.xticks()
#plt.xticks([3],['Imputation method'])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('AUC')
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_Ft.columns],loc='lower right') #, 
# Multiple box plots on one Axes
#plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()


#MCAR case

res=pd.read_csv('MAR-results.csv',sep=';',na_values='?')

#Without feature selection.
no_feature_selection_res = pd.DataFrame()
no_feature_selection_res = pd.concat([no_feature_selection_res,res['nofs']],axis=1)
for i in res.columns:
    if 'no_indicators_and_No_Feature_Selection' == i or 'Indicator_Variables_and_No_Feature_Selection' == i:
        continue
    if 'No_Feature_Selection' in i:
        no_feature_selection_res = pd.concat([no_feature_selection_res,res[i]],axis=1)

res_no_Ft=pd.DataFrame(no_feature_selection_res)
res_no_Ft.columns=['Best','GAIN','MF','MM','PPCA','SOFT','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT']
res_no_Ft.dropna(axis=1,how='all',inplace=True)
res_no_Ft = res_no_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]

feature_selection_res = pd.DataFrame()
feature_selection_res = pd.concat([feature_selection_res,res['fs']],axis=1)
for i in res.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res = pd.concat([feature_selection_res,res[i]],axis=1)
res_Ft=pd.DataFrame(feature_selection_res)
res_Ft.columns=['Best','GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
res_Ft.dropna(axis=1,how='all',inplace=True)
res_Ft = res_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]

for i in res_Ft.columns:
    res_Ft[i]=res_Ft[i].subtract(res_no_Ft[i],axis=0)
print(res_Ft)



fig, ax = plt.subplots()
boxplots = list()
i=0
pos = 0.2
for x in res_Ft.columns:
    data = list([np.array(res_Ft[x].dropna(axis=0))])    #print(data)
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[pos],widths = 0.26,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_diff))
    i+=1
    pos += 0.28
plt.xlim(0,4)
locs, labels = plt.xticks()
#plt.xticks([3],['Imputation method'])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('AUC')
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_Ft.columns],loc='lower right') #, 
# Multiple box plots on one Axes
#plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()
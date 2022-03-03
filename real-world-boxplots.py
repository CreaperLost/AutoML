import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
import numpy as np

res=pd.read_csv('Real-World-Results.csv',sep=';',na_values='?')
print(res.head())


colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black']
medianprops_normal = dict(linestyle='-.', linewidth=2.5,color='white')
medianprops_diff = dict(linestyle='-.', linewidth=2.5,color='black')
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
print(res_no_Ft)


"""
fig, ax = plt.subplots()
plt.title('AUC Boxplot when feature selection is excluded')
boxplots = list()
i=0
pos = 0.2
#['MM','BI+MM','GAIN','BI+GAIN','MF','BI+MF','SOFT','BI+SOFT','PPCA','BI+PPCA']
res_no_Ft = res_no_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]
for x in res_no_Ft.columns:
    data =  list([np.array(res_no_Ft[x].dropna(axis=0))])
    #print(data)
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[pos],widths = 0.3,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_normal))
    i+=1
    pos += 0.31
plt.xlim(0,4.3)
locs, labels = plt.xticks()
#plt.xticks([2.5],[''])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('AUC')
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_no_Ft.columns],loc='lower right')
# Multiple box plots on one Axes
plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()
"""

"""
#auc difference using mean/mode imputation as baseline.

res_no_Ft=res_no_Ft.subtract(res_no_Ft['MM'],axis=0)
res_no_Ft.drop('MM',axis=1,inplace=True)

fig, ax = plt.subplots()
plt.title('AUC Difference plot when feature selection is excluded')
boxplots = list()
i=0
pos = 0.2
#['MM','BI+MM','GAIN','BI+GAIN','MF','BI+MF','SOFT','BI+SOFT','PPCA','BI+PPCA']
res_no_Ft = res_no_Ft[['Best','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]
for x in res_no_Ft.columns:
    data =  list([np.array(res_no_Ft[x].dropna(axis=0))])
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[pos],widths = 0.3,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_diff))
    i+=1
    pos += 0.31
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
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_no_Ft.columns],loc='lower right')
# Multiple box plots on one Axes
plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()
"""





colors_best = ['Lime','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black']


res_no_Ft=res_no_Ft.subtract(res_no_Ft['Best'],axis=0)
res_no_Ft.drop('Best',axis=1,inplace=True)

fig, ax = plt.subplots()
plt.title('AUC Difference plot when feature selection is excluded')
boxplots = list()
i=0
pos = 0.2
#['MM','BI+MM','GAIN','BI+GAIN','MF','BI+MF','SOFT','BI+SOFT','PPCA','BI+PPCA']
res_no_Ft = res_no_Ft[['MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]
for x in res_no_Ft.columns:
    data =  list([np.array(res_no_Ft[x].dropna(axis=0))])
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[pos],widths = 0.3,boxprops=dict(facecolor=colors_best[i]), medianprops=medianprops_diff))
    i+=1
    pos += 0.31
plt.xlim(0,4)
locs, labels = plt.xticks()
#plt.xticks([3],['Imputation method'])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('AUC_Diff = AUC_Imputor - AUC_Best')
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_no_Ft.columns],loc='lower right')
# Multiple box plots on one Axes
#plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()





 


#With feature selection.
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

"""
fig, ax = plt.subplots()
plt.title('AUC Boxplot when feature selection is enforced')
boxplots = list()
i=0
pos = 0.2
#['MM','BI+MM','GAIN','BI+GAIN','MF','BI+MF','SOFT','BI+SOFT','PPCA','BI+PPCA']
res_Ft = res_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]
for x in res_Ft.columns:
    data = list([np.array(res_Ft[x].dropna(axis=0))])    #print(data)
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[pos],widths = 0.3,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_normal))
    i+=1
    pos += 0.31
plt.xlim(0,4.3)
locs, labels = plt.xticks()
#plt.xticks([3],['Imputation method'])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('AUC')
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_Ft.columns],loc='lower right')
# Multiple box plots on one Axes
plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()
"""

"""
#TRADEOFF PLOT
res_Ft=res_Ft.subtract(res_Ft['MM'],axis=0)
res_Ft.drop('MM',axis=1,inplace=True)

fig, ax = plt.subplots()
plt.title('AUC difference boxplot when feature selection is enforced')
boxplots = list()
i=0
pos = 0.2
#['MM','BI+MM','GAIN','BI+GAIN','MF','BI+MF','SOFT','BI+SOFT','PPCA','BI+PPCA']
res_Ft = res_Ft[['Best','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]
for x in res_Ft.columns:
    data = list([np.array(res_Ft[x].dropna(axis=0))])    #print(data)
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[pos],widths = 0.3,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_diff))
    i+=1
    pos += 0.31
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
plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()
"""



res_Ft=res_Ft.subtract(res_Ft['Best'],axis=0)
res_Ft.drop('Best',axis=1,inplace=True)

fig, ax = plt.subplots()
plt.title('AUC difference boxplot when feature selection is enforced')
boxplots = list()
i=0
pos = 0.2
#['MM','BI+MM','GAIN','BI+GAIN','MF','BI+MF','SOFT','BI+SOFT','PPCA','BI+PPCA']
res_Ft = res_Ft[['MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]
for x in res_Ft.columns:
    data = list([np.array(res_Ft[x].dropna(axis=0))])    #print(data)
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[pos],widths = 0.3,boxprops=dict(facecolor=colors_best[i]), medianprops=medianprops_diff))
    i+=1
    pos += 0.31
plt.xlim(0,4)
locs, labels = plt.xticks()
#plt.xticks([3],['Imputation method'])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('AUC_Diff = AUC_Imputor - AUC_Best')
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_Ft.columns],loc='lower right') #, 
# Multiple box plots on one Axes
#plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()

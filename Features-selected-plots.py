import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
import matplotlib.pyplot as plt
import numpy as np


res_mcar=pd.read_csv('MCAR-FT-Results.csv',sep=';',na_values='?')
res_mar=pd.read_csv('MAR-FT-Results.csv',sep=';',na_values='?')



colors = ['tab:blue','Lime','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black']
medianprops_normal = dict(linestyle='-.', linewidth=2.5,color='red')
medianprops_diff = dict(linestyle='-.', linewidth=2.5,color='black')
meanprops_normal = dict(linestyle='--', linewidth=2.5,color='black')

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
plt.rc('legend', fontsize=Y_AXIS_SIZE-2)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#With feature selection.
feature_selection_res = pd.DataFrame(res_mcar['Analysis'])
feature_selection_res = pd.concat([feature_selection_res,res_mcar['fs']],axis=1)
for i in res_mcar.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res = pd.concat([feature_selection_res,res_mcar[i]],axis=1)
res_Ft=pd.DataFrame(feature_selection_res)
res_Ft.columns=['Analysis','Best','GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
res_Ft.dropna(axis=1,how='all',inplace=True)

row_id=list(res_Ft['Analysis'])
list_10 =  [x for x in [row_id.index(i) if '10' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_25 =  [x for x in [row_id.index(i) if '25' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_50 =  [x for x in [row_id.index(i) if '50' in i else np.nan for i in  row_id] if ~np.isnan(x)]

"""
fig, ax = plt.subplots()
plt.title('Features Selected, MCAR case')
boxplots = list()
j=0
i=0
#['MM','BI+MM','GAIN','BI+GAIN','MF','BI+MF','SOFT','BI+SOFT','PPCA','BI+PPCA']
res_Ft = res_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]
for x in res_Ft.columns:
    data =  list([np.array(res_Ft.loc[list_10][x].dropna(axis=0)),np.array(res_Ft.loc[list_25][x].dropna(axis=0)),np.array(res_Ft.loc[list_50][x].dropna(axis=0))])
    #print(data)
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[0.5+j,2.7+j,4.9+j],widths = 0.15,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_normal))
    j+= 0.18
    i+=1

plt.xlim(0,9)
locs, labels = plt.xticks()
plt.xticks([1.5, 3.7, 5.9],['10%','25%','50%'])
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_Ft.columns], loc='lower right')
# Multiple box plots on one Axes
plt.ylabel('Features Selected')
plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()
"""

fig, ax = plt.subplots()
plt.title('Number of features Selected, MCAR case')
boxplots = list()
lineplots = list()
j=0
i=0
#['MM','BI+MM','GAIN','BI+GAIN','MF','BI+MF','SOFT','BI+SOFT','PPCA','BI+PPCA']
res_Ft = res_Ft[['Best','MM','BI+MM']]
for x in res_Ft.columns:
    data =  list([np.array(res_Ft.loc[list_10][x].dropna(axis=0)),np.array(res_Ft.loc[list_25][x].dropna(axis=0)),np.array(res_Ft.loc[list_50][x].dropna(axis=0))])
    #print(data)
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[0.5+j,2+j,3.5+j],widths = 0.15,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_normal,showmeans=True,meanprops=meanprops_normal,meanline=True))
    lineplots.append(ax.plot([0.5+j,2+j,3.5+j], [np.mean(data[0]),np.mean(data[1]),np.mean(data[2])],color=colors[i],zorder=500))
    j+= 0.18
    i+=1

plt.xlim(0,5)
locs, labels = plt.xticks()
plt.xticks([0.7, 2.2, 3.7],['10%','25%','50%'])
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_Ft.columns], loc='lower right')
# Multiple box plots on one Axes
plt.ylabel('Number of Features Selected')
plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()



#With feature selection.
feature_selection_res = pd.DataFrame(res_mar['Analysis'])
feature_selection_res = pd.concat([feature_selection_res,res_mar['fs']],axis=1)
for i in res_mar.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res = pd.concat([feature_selection_res,res_mar[i]],axis=1)
res_Ft=pd.DataFrame(feature_selection_res)
res_Ft.columns=['Analysis','Best','GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
res_Ft.dropna(axis=1,how='all',inplace=True)

row_id=list(res_Ft['Analysis'])
list_10 =  [x for x in [row_id.index(i) if '10' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_25 =  [x for x in [row_id.index(i) if '25' in i else np.nan for i in  row_id] if ~np.isnan(x)]
list_50 =  [x for x in [row_id.index(i) if '50' in i else np.nan for i in  row_id] if ~np.isnan(x)]

"""
fig, ax = plt.subplots()
plt.title('Features Selected, MAR case')
boxplots = list()
j=0
i=0
#['MM','BI+MM','GAIN','BI+GAIN','MF','BI+MF','SOFT','BI+SOFT','PPCA','BI+PPCA']
res_Ft = res_Ft[['Best','MM','BI+MM','MF','BI+MF','GAIN','BI+GAIN','SOFT','BI+SOFT','PPCA','BI+PPCA']]
for x in res_Ft.columns:
    data =  list([np.array(res_Ft.loc[list_10][x].dropna(axis=0)),np.array(res_Ft.loc[list_25][x].dropna(axis=0)),np.array(res_Ft.loc[list_50][x].dropna(axis=0))])
    #print(data)
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[0.5+j,2.7+j,4.9+j],widths = 0.15,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_normal))
    j+= 0.18
    i+=1

plt.xlim(0,9)
locs, labels = plt.xticks()
plt.xticks([1.5, 3.7, 5.9],['10%','25%','50%'])
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_Ft.columns], loc='lower right')
# Multiple box plots on one Axes
plt.ylabel('Features Selected')
plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()
"""

fig, ax = plt.subplots()
plt.title('Number of features Selected, MAR case')
boxplots = list()
lineplots = list()
j=0
i=0
#['MM','BI+MM','GAIN','BI+GAIN','MF','BI+MF','SOFT','BI+SOFT','PPCA','BI+PPCA']
res_Ft = res_Ft[['Best','MM','BI+MM']]
for x in res_Ft.columns:
    data =  list([np.array(res_Ft.loc[list_10][x].dropna(axis=0)),np.array(res_Ft.loc[list_25][x].dropna(axis=0)),np.array(res_Ft.loc[list_50][x].dropna(axis=0))])
    
    #print(data)
    boxplots.append(ax.boxplot(data,patch_artist=True,positions=[0.5+j,2+j,3.5+j],widths = 0.15,boxprops=dict(facecolor=colors[i]), medianprops=medianprops_normal,showmeans=True,meanprops=meanprops_normal,meanline=True))
    lineplots.append(ax.plot([0.5+j,2+j,3.5+j], [np.mean(data[0]),np.mean(data[1]),np.mean(data[2])],color=colors[i],zorder=500))
    j+= 0.18
    i+=1
plt.xlim(0,5)
locs, labels = plt.xticks()
plt.xticks([0.7, 2.2, 3.7],['10%','25%','50%'])
ax.legend([boxplots[i]['boxes'][0] for i in range(len(boxplots))],[i for i in res_Ft.columns], loc='lower right')
# Multiple box plots on one Axes
plt.ylabel('Number of Features Selected')
plt.tight_layout()
plt.rcParams["savefig.bbox"] = "tight"
plt.show()
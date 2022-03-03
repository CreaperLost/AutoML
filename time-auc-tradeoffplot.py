import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table
from scipy import stats
import numpy as np

time_df = pd.read_csv('Real-World-Results-Time2.csv',sep=';',engine='python',na_values='?')
Auc_df = pd.read_csv('Real-World-Results.csv',sep=';',engine='python',na_values='?')

time_df.drop(['Analysis','DatasetID'],axis=1,inplace=True)
Auc_df.drop(['Analysis','DatasetID'],axis=1,inplace=True)

print(time_df.info())
print(Auc_df.info())


#Without feature selection.
no_feature_selection_res_time = pd.DataFrame()
no_feature_selection_auc = pd.DataFrame()

for i in time_df.columns:
    if 'no_indicators_and_No_Feature_Selection' == i or 'Indicator_Variables_and_No_Feature_Selection' == i:
        continue
    if 'No_Feature_Selection' in i:
        no_feature_selection_res_time = pd.concat([no_feature_selection_res_time,time_df[i]],axis=1)

for i in Auc_df.columns:
    if 'no_indicators_and_No_Feature_Selection' == i or 'Indicator_Variables_and_No_Feature_Selection' == i:
        continue
    if 'No_Feature_Selection' in i:
        no_feature_selection_auc = pd.concat([no_feature_selection_auc,Auc_df[i]],axis=1)

res_no_Ft_time=pd.DataFrame(no_feature_selection_res_time)
res_no_Ft_time.columns=['GAIN','MF','MM','PPCA','SOFT','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT']
res_no_Ft_time.dropna(axis=1,how='all',inplace=True)
res_no_Ft_time_ready=res_no_Ft_time.mul(-1,axis=0).add(res_no_Ft_time['MM'],axis=0) 
res_no_Ft_time_ready=res_no_Ft_time_ready.divide(res_no_Ft_time['MM'],axis=0)
res_no_Ft_time_ready.drop('MM',axis=1,inplace=True)
print(res_no_Ft_time_ready)

res_no_Ft_auc=pd.DataFrame(no_feature_selection_auc)
res_no_Ft_auc.columns=['GAIN','MF','MM','PPCA','SOFT','BI+MM','BI+MF','BI+PPCA','BI+GAIN','BI+SOFT']
res_no_Ft_auc.dropna(axis=1,how='all',inplace=True)
res_no_Ft_auc_ready=res_no_Ft_auc.subtract(res_no_Ft_auc['MM'],axis=0)
res_no_Ft_auc_ready=res_no_Ft_auc_ready.divide(res_no_Ft_auc['MM'],axis=0)
res_no_Ft_auc_ready.drop('MM',axis=1,inplace=True)
print(res_no_Ft_auc_ready)

total = 0
better_slower = 0
worse_slower = 0
better_faster = 0
worse_faster =0
for i in res_no_Ft_auc_ready:
    for key,values in res_no_Ft_auc_ready[i].items():
        if values== np.nan and res_no_Ft_time_ready[i].loc[key] == np.nan:
            continue
        if values>0  and res_no_Ft_time_ready[i].loc[key] >0:
            better_faster = better_faster + 1
        elif values>0  and res_no_Ft_time_ready[i].loc[key] <0:
            better_slower = better_slower + 1
        elif values<=0  and res_no_Ft_time_ready[i].loc[key] <0:
            worse_slower = worse_slower + 1
        elif values <= 0 and res_no_Ft_time_ready[i].loc[key] >0:
            worse_faster = worse_faster + 1
        total  = total + 1


print('Total Runs',total)
print('better_faster Runs',better_faster)
print('better_slower Runs',better_slower)
print('worse_faster Runs',worse_faster)
print('worse_slower Runs',worse_slower)


plt.figure()
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20
ULTRA_SIZE = 30

plt.rc('font', size=ULTRA_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=ULTRA_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


plt.xlim(-1.0, 0.4)
plt.ylim(-700000.0,100000.0)
plt.axvline(x=0,color = 'black', linestyle = '-')
plt.axhline(y=0, color = 'black', linestyle = '-')




plt.fill_between([-1, 0],-700000,0,alpha=0.3, color='#CC0000')  # blue
plt.fill_between([0, 0.4], 0, 100000, alpha=0.3, color='#00CC00')  # yellow
plt.fill_between([-1, 0], 0 ,100000, alpha=0.3, color='#C0C0C0')  # orange
plt.fill_between([0, 0.4], -700000,0, alpha=0.3, color='#C0C0C0')  # red


plt.text(-0.7,-350000, 'MM Domination Quadrant')
plt.text(-0.7,-390000,str(worse_slower)+' points')
plt.text(0.1,-350000, str(better_slower)+' points')
plt.text(0.1,30000, str(better_faster)+' points')
plt.text(-0.7,30000, str(worse_faster)+' points')

markers = ['o', 'v', '^', 's', 'D', 'X', '<', '>', 'p']
for i in res_no_Ft_auc_ready.columns:
    marker_sele=markers[list(res_no_Ft_auc_ready.columns).index(i)]
    plt.scatter(res_no_Ft_auc_ready[i],res_no_Ft_time_ready[i],marker= marker_sele,label=i)
plt.legend()
plt.yticks([-700000,-600000,-500000,-400000,-300000,-200000,-100000,0,100000],['-7x$10^5$','-6x$10^5$','-5x$10^5$','-4x$10^5$','-3x$10^5$','-2x$10^5$','-$10^5$','0','$10^5$'])
plt.ylabel('Relative efficiency = ( ( Time_MM - Time_Imputor)/ Time_MM )')
plt.xlabel('Relative effectiveness = ( ( AUC_Imputor - AUC_MM ) / AUC_MM )')
plt.title('Trade-off plot when feature selection is excluded')
plt.show()













#FEATURE SELECTION 




#CHANGE THIS.
feature_selection_res_time = pd.DataFrame()
feature_selection_auc = pd.DataFrame()

for i in time_df.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_res_time = pd.concat([feature_selection_res_time,time_df[i]],axis=1)

for i in Auc_df.columns:
    if 'no_indicators_and_Feature_Selection' == i or 'Indicator_Variables_and_Feature_Selection' == i:
        continue
    if 'and_Feature_Selection' in i:
        feature_selection_auc = pd.concat([feature_selection_auc,Auc_df[i]],axis=1)

res_Ft_time=pd.DataFrame(feature_selection_res_time)
res_Ft_time.columns=['GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
res_Ft_time.dropna(axis=1,how='all',inplace=True)
res_Ft_time_ready=res_Ft_time.mul(-1,axis=0).add(res_Ft_time['MM'],axis=0) 
res_Ft_time_ready=res_Ft_time_ready.divide(res_Ft_time['MM'],axis=0)
res_Ft_time_ready.drop('MM',axis=1,inplace=True)
print(res_Ft_time_ready)

res_Ft_auc=pd.DataFrame(feature_selection_auc)
res_Ft_auc.columns=['GAIN','MM','MF','SOFT','PPCA','BI+GAIN','BI+MM','BI+SOFT','BI+PPCA','BI+MF']
res_Ft_auc.dropna(axis=1,how='all',inplace=True)
res_Ft_auc_ready=res_Ft_auc.subtract(res_Ft_auc['MM'],axis=0)
res_Ft_auc_ready=res_Ft_auc_ready.divide(res_Ft_auc['MM'],axis=0)
res_Ft_auc_ready.drop('MM',axis=1,inplace=True)
print(res_Ft_auc_ready)

total = 0
better_slower = 0
worse_slower = 0
better_faster = 0
worse_faster =0
for i in res_Ft_auc_ready:
    for key,values in res_Ft_auc_ready[i].items():
        if values== np.nan and res_Ft_time_ready[i].loc[key] == np.nan:
            continue
        if values>0  and res_Ft_time_ready[i].loc[key] >0:
            better_faster = better_faster + 1
        elif values>0  and res_Ft_time_ready[i].loc[key] <0:
            better_slower = better_slower + 1
        elif values<=0  and res_Ft_time_ready[i].loc[key] <0:
            worse_slower = worse_slower + 1
        elif values <= 0 and res_Ft_time_ready[i].loc[key] >0:
            worse_faster = worse_faster + 1
        total  = total + 1


print('Total Runs',total)
print('better_faster Runs',better_faster)
print('better_slower Runs',better_slower)
print('worse_faster Runs',worse_faster)
print('worse_slower Runs',worse_slower)


plt.figure()


plt.rc('font', size=ULTRA_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=ULTRA_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


plt.xlim(-1.0, 0.4)
plt.ylim(-700000.0,100000.0)
plt.axvline(x=0,color = 'black', linestyle = '-')
plt.axhline(y=0, color = 'black', linestyle = '-')




plt.fill_between([-1, 0],-700000,0,alpha=0.3, color='#CC0000')  # blue
plt.fill_between([0, 0.4], 0, 100000, alpha=0.3, color='#00CC00')  # yellow
plt.fill_between([-1, 0], 0 ,100000, alpha=0.3, color='#C0C0C0')  # orange
plt.fill_between([0, 0.4], -700000,0, alpha=0.3, color='#C0C0C0')  # red


plt.text(-0.7,-350000, 'MM Domination Quadrant', style='italic')
plt.text(-0.7,-390000,str(worse_slower)+' points', style='italic')
plt.text(0.1,-350000, str(better_slower)+' points', style='italic')
plt.text(0.1,30000, str(better_faster)+' points', style='italic')
plt.text(-0.7,30000, str(worse_faster)+' points', style='italic')

markers = ['o', 'v', '^', 's', 'D', 'X', '<', '>', 'p']
for i in res_Ft_auc_ready.columns:
    marker_sele=markers[list(res_Ft_auc_ready.columns).index(i)]
    plt.scatter(res_Ft_auc_ready[i],res_Ft_time_ready[i],marker= marker_sele,label=i)
plt.legend()
plt.yticks([-700000,-600000,-500000,-400000,-300000,-200000,-100000,0,100000],['-7x$10^5$','-6x$10^5$','-5x$10^5$','-4x$10^5$','-3x$10^5$','-2x$10^5$','-$10^5$','0','$10^5$'])
plt.ylabel('Relative efficiency = ( (Time_MM - Time_Imputor)/ Time_MM )')
plt.xlabel('Relative effectiveness = ( ( AUC_Imputor - AUC_MM ) / AUC_MM )')
plt.title('Trade-off plot when feature selection is enforced')
plt.show()

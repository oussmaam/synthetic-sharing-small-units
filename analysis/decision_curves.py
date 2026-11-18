"""Supplementary Figure S1: summarise the decision-curve runs and draw the figure."""
import pandas as pd, numpy as np
T=list(range(2,41)); NB=[f'nb{t}' for t in T]
d=pd.read_csv('../results/results_dca_sim_cdc_diabetes.csv')
C=['gamma','share','rep']
pr=d.pivot_table(index=C,columns='arm',values='ex_kl')*1000
pn=(pr['oracle_D_ct_n']-pr['oracle_h_star']).groupby(level=['gamma','share']).transform('mean')
gain=(pn>=2).rename('gain').reset_index()
d=d.merge(gain,on=C,how='left')
MAP={'local':'M0','share_LR_ct':'M1','IW_donor':'IW','B_untargeted':'M2','C_target':'M3','BR_SDG':'Brown','treat_all':'All'}
d=d[d.arm.isin(MAP)].assign(A=lambda x:x.arm.map(MAP))
print('gain cells:',gain.groupby('gain').size().to_dict())
out={}
for g,sub in d.groupby('gain'):
    m=sub.groupby('A')[NB].mean()*1000; out['sim_'+('gain' if g else 'nogain')]=m
    dm=(m-m.loc['M1'])
    print('\nSIM gain=',g,' NB per 1000 (M1 row abs), others diff vs M1')
    print(pd.concat([m.loc[['M1','All']],dm.drop(['M1','All'])])[['nb5','nb10','nb15','nb20','nb30','nb40']].round(1).to_string())
    print('mean diff 2-40:',dm.drop(['M1']).mean(1).round(2).to_dict())
s=pd.read_csv('../results/results_dca_sites_brfss.csv'); s['PR']=s.receiver==72
for g,sub in s.groupby('PR'):
    m=sub.groupby('arm')[NB].mean()*1000; out['site_'+('PR' if g else 'other')]=m
    dm=m-m.loc['M1']
    print('\nSITES PR=',g,'pairs',sub.groupby(['donor','receiver','rep']).ngroups)
    print(pd.concat([m.loc[['M1','treat_all']],dm.drop(['M1','treat_all','shared_raw','teacher','R1_support'],errors='ignore')])[['nb5','nb10','nb15','nb20','nb30','nb40']].round(1).to_string())
    print('mean diff 2-40:',dm.drop(['M1']).mean(1).round(2).to_dict())



# ---- figure
c = out
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
T=np.arange(2,41)
S=[('M0','Local score','#2a78d6','-'),('IW','Reweighting','#eb6834','--'),('M2','Untargeted synthetic','#1baf7a','-.'),('M3','Targeted synthetic','#eda100',(0,(1,1))),('Brown','Synthetic, Brown et al.','#e87ba4',(0,(5,1,1,1)))]
P=[('sim_gain','A. Simulation: scenarios with room for gain\n(50 scenarios $\\times$ 4 draws)'),('sim_nogain','B. Simulation: other scenarios\n(23 scenarios $\\times$ 4 draws)'),('site_other','C. Real sites outside Puerto Rico\n(80 donor-receiver pairs, 1 draw)'),('site_PR','D. Puerto Rico (Kansas donor, 1 draw)')]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
f,ax=plt.subplots(2,2,figsize=(7.5,6.2),sharex=True)
for a,(k,t) in zip(ax.flat,P):
    m=c[k]; 
    if 'site' in k: m=m.rename(index={'R1':'Brown'}) if False else m
    a.axhline(0,color='#888',lw=1)
    for code,lab,col,ls in S:
        if code=='Brown' and 'site' in k: continue
        if code in m.index: a.plot(T,(m.loc[code]-m.loc['M1']).values,color=col,ls=ls,lw=2,label=lab)
    a.set_title(t,fontsize=9,loc='left'); a.grid(axis='y',color='#e5e5e5',lw=.6)
for a in ax[1]: a.set_xlabel('Risk threshold (%)')
for a in ax[:,0]: a.set_ylabel('Net benefit minus updated sharing\n(per 1000 patients)')
h,l=ax[0,0].get_legend_handles_labels(); f.legend(h,l,loc='lower center',ncol=3,frameon=False,fontsize=8,handlelength=3)
f.tight_layout(rect=(0,.08,1,1)); f.savefig('figS1_decision_curves.pdf'); f.savefig('figS1_decision_curves.png',dpi=200)

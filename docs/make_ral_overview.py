"""Figure 1: controlled intervention and measured downstream outcomes.
Road geometry is schematic. All event counts come from results_master.md.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/ral_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch,FancyBboxPatch
import numpy as np
OUT=Path(__file__).resolve().parent/'figures/ral';OUT.mkdir(parents=True,exist_ok=True)
ink='#233746';muted='#667580';red='#AC594F';blue='#50799A';teal='#267D78';gray='#D9E1E6'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'axes.labelcolor':ink,'text.color':ink})
f=plt.figure(figsize=(7.18,3.65));a=f.add_axes([.015,.08,.97,.87]);a.set(xlim=(0,100),ylim=(0,100));a.axis('off')
def text(x,y,s,sz=7,c=ink,ha='left',weight='normal'):
 a.text(x,y,s,fontsize=sz,color=c,ha=ha,va='center',fontweight=weight,linespacing=1.35)
def arrow(x,y,xx,yy,c=muted):a.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=8,color=c,lw=.8))
def box(x,y,w,h,s,c=ink,fc='white',sz=7):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.25,rounding_size=1',fc=fc,ec=c,lw=.7));text(x+w/2,y+h/2,s,sz,c,'center')
text(0,98,'a  Shared candidates',8,weight='bold');text(29,98,'b  Change only the winner',8,weight='bold');text(66,98,'c  Measure the fitted policy',8,weight='bold')
# A: landscape bird's-eye view, three illustrative alternatives.
a.add_patch(Rectangle((1,36),23,42,fc='#EEF1F3',ec='none'))
for y in [39,75]:a.plot([1,24],[y,y],color='#98A8B2',lw=.8)
a.plot([1,24],[57,57],color='white',lw=1.3,ls=(0,(3,3)))
a.add_patch(FancyBboxPatch((14,65),4.8,6,boxstyle='round,pad=0,rounding_size=.5',fc='#7C8C98',ec='white',lw=.7))
u=np.linspace(0,1,90)
for bend,length,c in [(12,19,red),(4,16,blue),(-2,11,teal)]:
 a.plot(4+length*u,49+bend*np.sin(np.pi*u/2)**2,color=c,lw=1.7)
a.add_patch(FancyBboxPatch((2.2,46.5),4.2,5,boxstyle='round,pad=0,rounding_size=.6',fc=ink,ec='white',lw=.7))
text(12.5,85,'Fixed SFT policy → 64 samples',6.4,ha='center')
text(12.5,31,'Illustrative candidate trajectories',6.1,muted,'center')
box(1,12,23,12,'Same NC / DAC / TTC gates',teal,'#EDF6F3',6.6)
arrow(12.5,28,12.5,24)
# B: winner choice contrasted, NOT fictitious geometry claims
for y,c,head,detail in [(72,red,'Score','Highest PDMS'),(51,blue,'Margin','Largest clearance'),(30,teal,'Nearest','Closest to reference')]:
 a.plot([30,30],[y-7,y+7],color=c,lw=2.4)
 text(32,y+3,head,7.5,c,weight='bold');text(32,y-4,detail,6.8,muted)
arrow(24.7,18,27,18);a.plot([27,27],[18,72],color=muted,lw=.7)
for y in [72,51,30]:arrow(27,y,29,y)
for y in [72,51,30]:
 a.plot([56,59],[y,y],color=gray,lw=.8)
a.plot([59,59],[19,72],color=gray,lw=.8);arrow(59,19,57,19)
box(31,8,26,12,'Same preference fitting',ink,'#F3F5F7',7)
text(44,1,'Frozen VLM · trainable action expert',5.9,muted,'center')
# C: measured run points, not pipeline rectangles
plot=f.add_axes([.675,.265,.295,.44])
vals=[[-125,-330,-208,-157],[237,171,279,207],[246,227,246,204]]
for yy,(vs,c) in enumerate(zip(vals,[red,blue,teal])):
 jitter=np.linspace(-.12,.12,len(vs));plot.scatter(vs,np.full(len(vs),2-yy)+jitter,s=22,color=c,edgecolors='white',linewidths=.5,zorder=3)
plot.axvspan(-380,0,color=red,alpha=.045);plot.axvspan(0,350,color=teal,alpha=.045);plot.axvline(0,color=muted,lw=.8)
plot.set(xlim=(-380,350),ylim=(-.5,2.5),xticks=[-300,0,300],yticks=[0,1,2],yticklabels=['Nearest','Margin','Score'])
plot.tick_params(axis='both',length=0,labelsize=6.5,pad=3)
for sp in plot.spines.values():sp.set_visible(False)
plot.grid(axis='x',color=gray,lw=.5);plot.set_axisbelow(True)
plot.set_xlabel('Repairs − new failures',fontsize=7,labelpad=5)
text(83,85,'12,146 scenes · 136 logs',6.7,muted,'center')
text(83,78,'Each dot: one training run',6.5,muted,'center')
text(83,3,'Summed NC / DAC / TTC events',6.1,muted,'center')
f.text(.02,.025,'Same passing-candidate requirement. Opposite downstream repair balances.',fontsize=8,color=ink,fontweight='bold')
for ext in ['pdf','png','svg']:f.savefig(OUT/f'fig1_framework.{ext}',dpi=350,bbox_inches='tight',pad_inches=.05,facecolor='white')
plt.close(f)

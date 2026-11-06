"""Reproducible RA-L framework and ledger. Counts transcribed from results_master.md.
No measured trajectory is synthesized: the road drawing is explicitly schematic.
"""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/ral_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle,FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'/'ral';OUT.mkdir(parents=True,exist_ok=True)
N='#263E53';B='#516F91';T='#287E77';R='#B46257';G='#667582';L='#DCE3E8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'text.color':N,'axes.labelcolor':N,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':L,'xtick.color':G,'ytick.color':N})
def save(fig,name):
 for ext in ['pdf','png','svg']:
  fig.savefig(OUT/(name+'.'+ext),dpi=300,bbox_inches='tight',pad_inches=.04,facecolor='white')
 plt.close(fig)
def box(ax,x,y,w,h,fc,ec='none'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.015',fc=fc,ec=ec,lw=.7))
def txt(ax,x,y,s,size=8,color=N,weight='normal',ha='left'):
 ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va='center',linespacing=1.4)
def arrow(ax,a,b,c=G):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=.9,color=c))
import runpy
runpy.run_path(str(ROOT/'make_ral_overview.py'), run_name='__main__')
labels=['PDMS / 20','PDMS / 21','PDMS / 22','PDMS / 23','Nearest / 20','Nearest / 21','Nearest / 22','Nearest / 23','Margin / 20','Margin / 21','Margin / 22','Margin / 23']
rows=[[(25,60),(147,158),(84,163)],[(27,79),(170,270),(90,268)],[(27,67),(143,181),(76,206)],[(20,61),(154,187),(76,159)],[(68,23),(160,106),(183,36)],[(54,23),(165,89),(163,43)],[(42,12),(139,22),(123,24)],[(52,21),(172,95),(143,47)],[(45,9),(126,40),(136,21)],[(64,34),(169,118),(155,65)],[(57,18),(157,57),(170,30)],[(66,29),(177,136),(170,41)]]
net=[sum(r-n for r,n in row) for row in rows]
fig,axs=plt.subplots(1,4,figsize=(10.4,4.5),sharey=True,gridspec_kw={'width_ratios':[1,1,1,1.12]})
fig.subplots_adjust(left=.14,right=.965,bottom=.14,top=.83,wspace=.28)
y=np.arange(12)[::-1]
for j,ax in enumerate(axs):
 ax.set_ylim(-.65,11.7);ax.set_yticks(y);ax.tick_params(axis='y',length=0);ax.grid(axis='x',color='#E6EBEF',lw=.6);ax.set_axisbelow(True)
 for i,yy in enumerate(y):
  if i%2==0:ax.axhspan(yy-.45,yy+.45,color='#F5F7F9',zorder=-1)
 ax.axhline(7.5,color=L,lw=.9);ax.axhline(3.5,color=L,lw=.9)
 if j<3:
  ax.set_title(['(a) NC','(b) DAC','(c) TTC'][j],fontsize=9,fontweight='bold');ax.set_xlim(-10,300);ax.set_xticks([0,100,200,300]);ax.set_xlabel('Gate events',fontsize=8)
  for yy,row in zip(y,rows):
   r,n=row[j];ax.plot([r,n],[yy,yy],color='#AAB7C0',lw=1.2);ax.scatter(r,yy,s=18,color=T,zorder=3);ax.scatter(n,yy,s=17,color=R,marker='s',zorder=3)
 else:
  ax.set_title('(d) Net gate events',fontsize=9,fontweight='bold');ax.set_xlim(-410,440);ax.set_xticks([-300,0,300]);ax.axvline(0,color=G,lw=.8);ax.set_xlabel('Repairs − new failures',fontsize=8)
  for yy,n in zip(y,net):
   ax.barh(yy,n,height=.50,color=T if n>0 else R,alpha=.87)
   ax.text(n+(12 if n>0 else -12),yy,f'{n:+d}',ha='left' if n>0 else 'right',va='center',fontsize=7.5,color=T if n>0 else R)
 axs[0].set_yticklabels(labels,fontsize=8)
fig.text(.14,.955,'A passing training target does not guarantee a favorable policy repair balance',fontsize=10,fontweight='bold')
fig.legend(handles=[Line2D([],[],marker='o',ls='',color=T,label='Repair'),Line2D([],[],marker='s',ls='',color=R,label='New failure')],loc='upper right',bbox_to_anchor=(.965,.94),frameon=False,ncol=2,fontsize=8)
fig.text(.14,.893,'Full navtest: 12,146 scenes / 136 logs; matched SFT baseline',fontsize=8,color=G)
save(fig,'fig2_repairs')
(OUT/'ledger_plot_data.json').write_text(json.dumps({'source':'results_master.md D2 and October 29 updates; recorded counts, not raw-score recomputation','labels':labels,'counts_nc_dac_ttc':rows,'net':net},indent=2)+'\n')
print('Saved framework and 12-run ledger (PDF + PNG).')

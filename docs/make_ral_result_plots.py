from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/ral_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import json
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'/'ral';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#B8C3CB','text.color':'#243746','axes.labelcolor':'#243746','xtick.color':'#526372','ytick.color':'#243746','savefig.facecolor':'white'})
TEAL='#287E77';BLUE='#516F91';RUST='#B46257';GRAY='#87949F'
DATA={
'paired':{'labels':['Nearest / 20','Margin / 20','Nearest / 21','Margin / 21','Nearest / 22','Margin / 22','Nearest / 23','Margin / 23'],
'PDMS':[[.340,.058,.651],[.646,.192,1.139],[1.598,1.034,2.193],[1.415,.938,1.938],[1.270,.747,1.872],[1.016,.584,1.512],[.968,.567,1.381],[.544,.223,.861]],
'NC':[[.626,.387,.888],[.543,.316,.794],[.671,.424,.945],[.671,.469,.910],[.552,.337,.785],[.634,.411,.886],[.572,.354,.818],[.630,.428,.853]],
'DAC':[[.535,.295,.791],[.799,.342,1.315],[1.449,.876,2.116],[1.243,.775,1.797],[1.276,.740,1.923],[1.136,.681,1.674],[.906,.507,1.354],[.609,.369,.857]],
'TTC':[[1.861,1.418,2.311],[1.597,1.184,2.010],[2.453,1.958,2.975],[2.206,1.687,2.738],[1.885,1.437,2.366],[2.223,1.728,2.764],[1.474,1.102,1.864],[1.745,1.317,2.196]],
'Progress':[[-1.942,-2.269,-1.589],[-1.264,-1.722,-.772],[-.735,-1.295,-.140],[-.708,-1.152,-.220],[-.721,-1.228,-.148],[-1.484,-1.913,-1.016],[-.506,-.902,-.105],[-1.381,-1.676,-1.077]]},
'sft_draws':[[-.002,-.238,.244],[.045,-.198,.277],[.047,-.215,.304]],
'default':[[.418,-.084,.912],[.72,.46,.98],[.078,-.473,.591]],
'L2':[[.422,-.117,.921],[.777,.445,1.096],[.275,-.341,.837]],
'larger':[[1.272,.913,1.628],[.774,.466,1.103]]}
def base(ax,limits):
 ax.set_xlim(*limits);ax.axvline(0,color='#6D7D89',lw=.85,zorder=0);ax.grid(axis='x',color='#E6EBEF',lw=.6);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0);ax.set_xlabel('Score-point difference')
def interval(ax,row,y,color,marker='o',open_=False):
 v,lo,hi=row
 ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt=marker,color=color,mfc='white' if open_ else color,mec=color,ms=4.5,lw=1.2,capsize=2,zorder=3)
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.06)
 fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.06)
 plt.close(fig)
fig,axs=plt.subplots(1,5,figsize=(10.4,3.9),sharey=True)
fig.subplots_adjust(left=.13,right=.99,bottom=.23,top=.77,wspace=.24)
ys=np.arange(8)[::-1]
for j,(ax,key) in enumerate(zip(axs,['PDMS','NC','DAC','TTC','Progress'])):
 limits=(-2.55,.35) if key=='Progress' else ((-.15,3.1) if key=='TTC' else (-.15,2.4 if key in ['PDMS','DAC'] else 1.1))
 base(ax,limits);ax.set_ylim(-.6,7.6);ax.set_title(f'({chr(97+j)}) {key}',fontweight='bold',pad=10)
 for i,y in enumerate(ys):
  ax.axhspan(y-.45,y+.45,color='#F5F7F9' if i%2==0 else 'white',zorder=-2)
  interval(ax,DATA['paired'][key][i],y,TEAL if i%2==0 else BLUE,'o' if i%2==0 else 's')
 ax.set_yticks(ys)
 if j==0:ax.set_yticklabels(DATA['paired']['labels'])
fig.text(.13,.94,'Geometric target minus PDMS target',fontsize=11,fontweight='bold')
fig.text(.13,.865,'Same training seed and evaluation scenes; 95% log-cluster intervals',fontsize=8,color='#526372')
fig.text(.13,.025,'12,146 scenes / 136 logs. Contrasts share models and logs; intervals condition on the trained policies.',fontsize=7,color='#526372')
save(fig,'fig3_target_contrasts')
fig,axs=plt.subplots(1,3,figsize=(10.4,3.5),gridspec_kw={'width_ratios':[1,1.2,1.2]})
fig.subplots_adjust(left=.10,right=.99,bottom=.25,top=.77,wspace=.62)
ax=axs[0];base(ax,(-.38,.38));ax.set_title('(a) SFT: different latent draws',fontweight='bold',pad=13)
for row,y in zip(DATA['sft_draws'],[2,1,0]):interval(ax,row,y,GRAY)
ax.set_yticks([2,1,0],['L2 − default','L3 − default','L3 − L2']);ax.set_ylim(-.6,2.6)
ax=axs[1];base(ax,(-.6,1.25));ax.set_title('(b) Same models, new draw',fontweight='bold',pad=13)
for i,y in enumerate([2,1,0]):
 interval(ax,DATA['default'][i],y+.13,GRAY,open_=True)
 interval(ax,DATA['L2'][i],y-.13,BLUE,'s')
ax.set_yticks([2,1,0],['Nearest / 20','Margin / 20','PDMS / 20']);ax.set_ylim(-.6,2.6)
ax.legend(handles=[Line2D([],[],color=GRAY,marker='o',mfc='white',label='Default'),Line2D([],[],color=BLUE,marker='s',label='L2')],loc='lower left',bbox_to_anchor=(0,-.31),frameon=False,ncol=2,fontsize=7,handlelength=1)
ax=axs[2];base(ax,(-.2,1.8));ax.set_title('(c) Larger training budget',fontweight='bold',pad=13)
for i,y in enumerate([1,0]):
 interval(ax,DATA['default'][i],y+.12,GRAY,open_=True)
 interval(ax,DATA['larger'][i],y-.12,TEAL,'s')
ax.set_yticks([1,0],['Nearest / 20','Margin / 20']);ax.set_ylim(-.65,1.65)
ax.legend(handles=[Line2D([],[],color=GRAY,marker='o',mfc='white',label='Standard'),Line2D([],[],color=TEAL,marker='s',label='Larger')],loc='lower left',bbox_to_anchor=(0,-.31),frameon=False,ncol=2,fontsize=7,handlelength=1)
fig.text(.10,.95,'Sensitivity to inference sampling and training budget',fontsize=11,fontweight='bold')
fig.text(.10,.865,'PDMS differences with 95% log-cluster intervals; each trained policy uses its matched SFT baseline',fontsize=8,color='#526372')
fig.text(.10,.005,'(a) Shared draws, not independent replicates. (c) About 2× scenes and updates; one seed per rule, not a scaling law.',fontsize=7,color='#526372')
save(fig,'fig4_sampling_budget')
(OUT/'result_plot_data.json').write_text(json.dumps({'source':'results_master.md D5/D6; reported summaries, not recomputed raw scores','data':DATA},indent=2)+'\n')
print('Saved two PNG/PDF figure pairs and the plotted numerical data.')

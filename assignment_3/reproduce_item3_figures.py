from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
E=Path(__file__).parent/'item3_evidence'
O=Path(__file__).parent/'reproduced'
(O/'figures').mkdir(parents=True,exist_ok=True)
df=pd.read_csv(E/'reports/validation/per_seed.csv');main=df[df.model.isin(['M0','M1','M2'])]
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','font.family':'DejaVu Sans'})
fig,ax=plt.subplots(figsize=(3.4,2.7))
colors={'M0':'#2166ac','M1':'#bf5818','M2':'#238b45'}
for model,g in main.groupby('model'):
 ax.errorbar(g.latency_ms,g.accuracy,xerr=[g.latency_ms-g.q1_ms,g.q3_ms-g.latency_ms],fmt='o',capsize=3,label=model,color=colors[model])
 for _,r in g.iterrows():ax.annotate(str(r.seed),(r.latency_ms,r.accuracy),xytext=(5,6),textcoords='offset points',fontsize=6.5)
ax.set(xlabel='CPU model-only latency (ms)\nMedian with Q1–Q3',ylabel='Validation accuracy (%)');ax.legend(ncol=3,fontsize=7,columnspacing=.8);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(O/'figures/accuracy_latency.png',dpi=220);plt.close(fig)
variants=['fp32','linear','embeddings','both'];fig,ax=plt.subplots(figsize=(3.4,2.6));g=df[df.kind=='M1'];x=np.arange(4)
for seed,col in zip([42,43,44],['#2166ac','#bf5818','#238b45']):
 a=g[g.seed==seed].set_index('variant').loc[variants];ax.plot(x,a.macro_f1,'o-',label=f'Seed {seed}',color=col,linewidth=1.5)
ax.set_xticks(x,['M1\n(FP32)','Linear\nonly','Embeddings\nonly','M2\n(both)']);ax.set_ylabel('Validation macro-F1 (%)');ax.set_ylim(20,51);ax.legend(ncol=3,loc='lower left',fontsize=6.5,columnspacing=.7);ax.grid(axis='y',alpha=.15);fig.tight_layout();fig.savefig(O/'figures/quantization_ablation.png',dpi=220);plt.close(fig)

summary=main.groupby('model')[['accuracy','macro_f1','latency_ms','size_MB','peak_process_MiB']].agg(['mean','std'])
summary.to_csv(O/'summary_mean_sample_SD.csv')
print('Reproduced two figures and summary from saved public values; no training or model evaluation.')

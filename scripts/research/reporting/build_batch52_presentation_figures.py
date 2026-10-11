"""Render scientific figures for the batch52 Njoint presentation."""
from pathlib import Path
import csv, json, html, zipfile
import numpy as np
import matplotlib.pyplot as plt
import build_v61_presentation_figures as g

ROOT = Path(__file__).resolve().parents[2]
g.OUT = ROOT / 'outputs' / '参赛展示配图_batch52_Njoint_20261004'
g.OUT.mkdir(parents=True, exist_ok=True)
g.FILES.clear()
g.TEXT_BOXES.clear()
OUT = g.OUT
txt, box, rect, arrow, canvas, save = g.txt, g.box, g.rect, g.arrow, g.canvas, g.save
BLUE, TEAL, PURPLE, INK, MUTED, GRID = g.BLUE, g.TEAL, g.PURPLE, g.INK, g.MUTED, g.GRID
METRICS = json.loads((ROOT/'reports/breakthrough_20261002/batch52/final_test_metrics.json').read_text())


def fig1():
    fig, ax = canvas(1,'','')
    box(ax,80,220,440,190,'DNA 片段','100–600 bp · FASTA',BLUE,'#F1F6FC',34)
    txt(ax,300,490,'A  C  G  T  N',34,BLUE,'bold')
    box(ax,670,220,360,190,'四通道编码','A / C / G / T',TEAL,'#EDF8F5',32)
    arrow(ax,[(520,315),(670,315)],BLUE,2.5,head=20)
    box(ax,1150,220,350,190,'卷积输入','4 × L',PURPLE,'#F5F1FA',34)
    arrow(ax,[(1030,315),(1150,315)],TEAL,2.5,head=20)
    # A compact one-hot illustration: columns are A, C, G, T and N.
    a=fig.add_axes([.415,.36,.25,.22]); a.set_axis_off()
    matrix=np.zeros((4,5)); matrix[np.arange(4),np.arange(4)]=1
    for r in range(4):
        a.text(-.45,r,'ACGT'[r],ha='center',va='center',fontsize=26,color=INK)
        for c in range(5):
            a.add_patch(plt.Rectangle((c-.42,r-.38),.84,.76,fc=TEAL if matrix[r,c] else '#EEF3F6',ec='white'))
            a.text(c,r,str(int(matrix[r,c])),ha='center',va='center',fontsize=25,color='white' if matrix[r,c] else MUTED)
    a.set(xlim=(-.9,4.7),ylim=(3.7,-1.1))
    for c,b in enumerate('ACGTN'):a.text(c,-.78,b,ha='center',va='center',fontsize=26,color=INK)
    box(ax,1110,520,430,135,'有效位置掩码','N 与补齐位设为 0',PURPLE,'white',28)
    box(ax,130,765,620,100,'GRCh37 / hg19 参考序列',color=BLUE,fs=28)
    box(ax,850,765,620,100,'HEK293 片段活性标签',color=TEAL,fs=28)
    save(fig,'01_输入与编码')


def fig2():
    fig, ax=canvas(2,'','')
    box(ax,65,210,285,150,'四通道输入','4 × L',BLUE,'#F2F7FC',31)
    box(ax,460,210,365,150,'初始卷积','64 通道 · 核宽 9',BLUE,'#F2F7FC',31)
    box(ax,940,185,590,200,'残差块 × 3','核宽 5 · 膨胀率 1 / 2 / 4',TEAL,'#EFF8F6',34)
    arrow(ax,[(350,285),(460,285)],BLUE,2.2,head=19)
    arrow(ax,[(825,285),(940,285)],TEAL,2.2,head=19)
    box(ax,995,510,495,150,'掩码均值 + 最大池化','64 + 64 = 128 维',TEAL,'#EFF8F6',29)
    arrow(ax,[(1240,385),(1240,510)],TEAL,2.2,head=19)
    box(ax,580,510,310,150,'特征拼接','130 维',PURPLE,'#F5F1FA',31)
    arrow(ax,[(995,585),(890,585)],TEAL,2.2,head=19)
    box(ax,90,500,375,170,'回归头','130 → 64 → 1',BLUE,'#F2F7FC',33)
    arrow(ax,[(580,585),(465,585)],PURPLE,2.2,head=19)
    box(ax,510,785,450,90,'log 长度 + GC 比例',color=PURPLE,fs=27)
    arrow(ax,[(735,785),(735,660)],PURPLE,2.2,head=19)
    box(ax,90,785,330,90,'预测活性',color=BLUE,fill='#EAF3FC',fs=33)
    arrow(ax,[(255,670),(255,785)],BLUE,2.2,head=19)
    save(fig,'02_残差卷积网络')


def fig3():
    fig,ax=canvas(3,'','')
    box(ax,75,200,385,140,'单条片段','序列 → 预测活性',BLUE,'#F2F7FC',31)
    box(ax,640,200,390,140,'活性损失','预测值 ↔ 实测值',BLUE,'#F2F7FC',31)
    arrow(ax,[(460,270),(640,270)],BLUE,2.2,head=19)
    box(ax,75,480,385,200,'嵌套片段对','短片段 / 长片段',TEAL,'#EFF8F6',31)
    box(ax,640,480,390,200,'差值损失','Δ = 短 − 长',TEAL,'#EFF8F6',31)
    arrow(ax,[(460,580),(640,580)],TEAL,2.2,head=19)
    box(ax,1180,305,340,245,'联合训练','活性损失\n+ 0.5 × 差值损失',PURPLE,'#F5F1FA',31)
    arrow(ax,[(1030,270),(1110,270),(1110,360),(1180,360)],BLUE,2.2,head=19)
    arrow(ax,[(1030,580),(1110,580),(1110,500),(1180,500)],TEAL,2.2,head=19)
    rect(ax,130,780,1340,100,'#F5F8FA',GRID,14)
    txt(ax,800,830,'共享 CNN    ·    分组加权 SmoothL1    ·    AdamW',28,INK,'bold')
    save(fig,'03_活性与配对联合训练')


def fig4():
    fig,ax=canvas(4,'','')
    for x,n,c,t in [(90,'531,860',BLUE,'训练片段'),(620,'112,640',TEAL,'开发片段'),(1150,'117,155',PURPLE,'测试片段')]:
        box(ax,x,200,360,180,n,t,c,'white',39)
    txt(ax,800,450,'按序列组划分数据',31,INK,'bold')
    box(ax,100,550,355,150,'训练 6 轮','随机种子 20261004',BLUE,'#F2F7FC',32)
    box(ax,620,550,365,150,'选择第 5 轮','开发集配对误差',TEAL,'#EFF8F6',32)
    box(ax,1150,550,350,150,'固定模型评估','TEST 指标',PURPLE,'#F5F1FA',32)
    arrow(ax,[(455,625),(620,625)],BLUE,2.2,head=20)
    arrow(ax,[(985,625),(1150,625)],TEAL,2.2,head=20)
    txt(ax,800,820,'GSE301246  ·  fold0  ·  HEK293',29,MUTED)
    save(fig,'04_数据划分与模型选择')


def fig5():
    fig,ax=canvas(5,'','')
    specs=[('activity','rho','活性排序相关',1.0,lambda v:f'{v:.3f}'),
           ('activity','top10_precision','前 10% 命中率',1.,lambda v:f'{v:.1%}'),
           ('activity','group_mae','活性误差 ↓',1.,lambda v:f'{v:.3f}')]
    cols=[MUTED,BLUE,TEAL]
    for i,(group,key,label,lim,fmt) in enumerate(specs):
        a=fig.add_axes([.07+i*.32,.30,.255,.45])
        vals=[METRICS['arms'][m][group][key] for m in ['R0','Nabs','Njoint']]
        a.bar(np.arange(3),vals,color=cols,width=.62)
        for j,v in enumerate(vals):a.text(j,v+.035,fmt(v),ha='center',fontsize=26,color=INK,weight='bold')
        a.set(ylim=(0,lim),xticks=[0,1,2],xticklabels=['R0','Nabs','Njoint'],yticks=[0,.5,1])
        a.tick_params(axis='both',labelsize=24,colors=INK,length=0,pad=12)
        a.spines[['top','right','left']].set_visible(False);a.spines['bottom'].set_color(GRID)
        a.grid(axis='y',color=GRID);a.set_axisbelow(True)
        txt(ax,315+i*512,185,label,30,INK,'bold')
    txt(ax,800,835,'同一 TEST 集  ·  117,155 条片段',28,MUTED)
    save(fig,'05_测试集表现对比')


def fig6():
    fig,ax=canvas(6,'','')
    rows=list(csv.DictReader((ROOT/'代码提交文件夹/demo_expected.tsv').open(encoding='utf-8-sig'),delimiter='\t'))
    rows=sorted(rows,key=lambda x:float(x['activity_pred_Log2RPM']),reverse=True)[:6]
    box(ax,70,190,425,150,'FASTA 输入','12 条示例片段',BLUE,'#F2F7FC',33)
    box(ax,70,435,425,150,'活性预测','Log2RPM',TEAL,'#EFF8F6',33)
    box(ax,70,680,425,150,'候选排序','按预测活性降序',PURPLE,'#F5F1FA',33)
    arrow(ax,[(282,340),(282,435)],BLUE,2.3,head=20)
    arrow(ax,[(282,585),(282,680)],TEAL,2.3,head=20)
    a=fig.add_axes([.47,.215,.455,.56])
    vals=[float(x['activity_pred_Log2RPM']) for x in rows]
    a.barh(np.arange(6),vals,color=[TEAL]+['#9EBFD5']*5,height=.58)
    a.invert_yaxis(); a.set_yticks(np.arange(6),[x['seq_id'] for x in rows])
    a.tick_params(labelsize=24,colors=INK,length=0,pad=12)
    a.set_xlim(min(0,min(vals)),max(vals)*1.25)
    for i,v in enumerate(vals):a.text(v+.035,i,f'{v:.3f}',va='center',fontsize=27,color=INK,weight='bold')
    a.spines[['top','right','left']].set_visible(False); a.spines['bottom'].set_color(GRID)
    a.grid(axis='x',color=GRID);a.set_axisbelow(True)
    txt(ax,1090,860,'预测活性 · Log2RPM',28,MUTED)
    save(fig,'06_实际运行与候选排序')


def fig7():
    fig,ax=canvas(7,'','')
    for i,(x,t,s) in enumerate([(80,'序列输入','100–600 bp'),(620,'活性预测','HEK293 · Log2RPM'),(1160,'候选筛选','按预测活性排序')]):
        box(ax,x,235,370,140,t,s,BLUE,'#F3F7FC',28)
        if i<2:arrow(ax,[(x+370,305),(x+540,305)],BLUE,2.1,head=19)
    box(ax,1160,560,370,145,'模型候选验证','同位点表征',TEAL,'white',28,True)
    box(ax,620,560,370,145,'配对数据','预测 + 实测',TEAL,'white',28,True)
    box(ax,80,560,370,145,'模型更新','下一轮训练',PURPLE,'white',28,True)
    arrow(ax,[(1345,375),(1345,560)],TEAL,2.1,True,19)
    arrow(ax,[(1160,632),(990,632)],TEAL,2.1,True,19)
    arrow(ax,[(620,632),(450,632)],PURPLE,2.1,True,19)
    arrow(ax,[(265,560),(265,465),(805,465),(805,375)],PURPLE,2.1,True,19)
    box(ax,1140,800,410,84,'Bxb1 同位点表征平台',color=TEAL,fill='#F0F8F5',fs=26)
    arrow(ax,[(1345,800),(1345,705)],TEAL,2.1,True,19)
    rect(ax,100,810,45,30,'white',BLUE,6,1.6)
    txt(ax,165,825,'已有模块',24,INK,ha='left')
    rect(ax,600,810,45,30,'white',TEAL,6,1.6,True)
    txt(ax,665,825,'后续衔接',24,INK,ha='left')
    save(fig,'07_实验平台衔接')


TEXTS=[
('输入与编码','模型读取 100–600 bp 的 DNA 片段，FASTA 中每条记录包含一个编号和一段序列。A、C、G、T 分别对应四个通道；N 与批次补齐位置编码为全零，并由有效位置掩码处理。L 是当前批次的序列长度，最高为 600。训练序列按 GRCh37（hg19）参考基因组提取，测量标签来自 HEK293。矩阵是 A/C/G/T/N 的编码示意。建议放在输入说明页，强调“序列进入网络，模型返回活性”。'),
('残差卷积网络','本版为序列 CNN。初始卷积从 4 通道映射至 64 通道，卷积核宽度为 9；随后通过 3 个残差块，各含两层核宽 5 的卷积，膨胀率依次为 1、2、4。有效位置的均值池化和最大池化各得到 64 维，再加入标准化后的 log 长度与 GC 比例，共 130 维。回归头按 130→64→1 输出标准化活性，随后还原到 Log2RPM 标签尺度。卷积使用 GELU，残差块与回归头的 Dropout 为 0.1。图按上排左到右、下排右到左阅读。'),
('活性与配对联合训练','Njoint 同时学习片段活性和嵌套片段的活性变化。第一项损失比较单条片段的预测值与实测值；第二项先分别计算短片段和长片段的活性，再用“短减长”得到差值，将预测差值与实测差值比较。两个任务共享同一个 CNN。损失函数为分组加权 SmoothL1，联合损失=活性损失+0.5×差值损失，训练在标准化标签尺度上计算。片段按组大小的倒数加权，配对按配对组大小的倒数加权。Nabs 使用同一结构和初始参数，训练目标为第一项活性损失。'),
('数据划分与模型选择','数据来自 GSE301246 的 v2_MH4 fold0，目标列为 Log2RPM_HEK293。按序列组划分后，训练集有 531,860 条，开发集 112,640 条，测试集 117,155 条，合计 761,655 条。组数分别为 3,724、814、794；组的归属在三个集合之间保持独立。训练共 6 轮，依据开发集配对误差选择第 5 轮，checkpoint 内的 epoch 从 0 计数，因此记录为 4。随机种子 20261004，AdamW 学习率 0.001，batch size 32，配对 microbatch 16。'),
('测试集表现对比','这张图读取同一 TEST 集的归档结果。R0 为 342 维序列组成特征的岭回归基线；Nabs 为活性回归 CNN；Njoint 为活性与配对差值联合训练 CNN。活性排序 Spearman 依次为 0.5026、0.5749、0.5799；前 10% 候选命中率依次为 39.55%、53.89%、55.22%；分组平均绝对误差依次为 0.8628、0.7796、0.7806。前 10% 命中率是预测排名前 10% 的片段中，实测也位于前 10% 的比例。排序相关和命中率越高越好，误差越低越好；三个纵轴均从零开始。选用 Njoint 的主要依据是高活性候选筛选；Nabs 的数值误差略低。两者配对 MAE 为 0.8398 与 0.8388，配对 MAE 差值的分组自助法 95% 区间为 [-0.00233, 0.00442]，两者配对表现接近。'),
('实际运行与候选排序','这张图展示本次提交包 demo.py 的实际输入与输出。示例包含来自 TEST 的 12 条片段，每组抽取一条，按文件次序选定；先用 Njoint 评分，再按预测活性降序排列，图中显示其中排名前六的片段。横轴为 activity_pred_Log2RPM，数值已经由训练标签标准化空间还原。DEMO 编号与原始坐标、实测值、序列组的对应关系保存在 demo_provenance.tsv，所有 12 条预测保存在 results.tsv。这个示例展示读取序列、批量推理、候选排序和结果导出的运行流程。新设计候选也可写入同样格式的 FASTA 后评分。'),
('实验平台衔接','上排从左到右是已有的计算流程：输入序列、预测活性、筛选候选。右侧向下进入模型候选的同位点实验验证；下排从右到左形成预测与实测的配对数据，并用于下一轮模型更新。模型更新沿中部折线向上返回“活性预测”。底部 Bxb1 平台通过独立向上箭头支持候选验证。实线边框表示已有模块，虚线边框及连线表示后续衔接。团队当前已有平台或天然启动子数据；模型候选的同位点验证为下一阶段工作。虚线连接均以实心三角形收尾，按箭头方向阅读。')
]


def handoff():
    intro='batch52 Njoint 展示配图使用说明\n版本：v6.38.0，2026-10-04\n\n7 张独立配图，各含 3840×2160 PNG 和文字可编辑 SVG。画幅 16:9，字体为微软雅黑；正文及刻度至少 24 pt。图片顶部的编号、整图标题与品牌名均已移除。详细解释保存在本文件，供 PPT 制作人与汇报人使用。PNG 可直接插入 PPT；SVG 适合在 Illustrator 中调整。\n\n建议顺序：02 网络 → 03 训练 → 04 数据 → 05 结果 → 06 运行；01 解释输入，07 说明实验衔接。\n\n'
    content=intro+'\n\n'.join(f'{i+1:02d}｜{title}\n文件：{g.FILES[i]}.png / .svg\n\n{body}' for i,(title,body) in enumerate(TEXTS))
    content+='\n\n依据：提交包 model_core.py、spec.py、losses.py、config.json、MODEL_MANIFEST.json、final_test_metrics.json、demo_expected.tsv、demo_provenance.tsv。\n'
    (OUT/'逐图说明_交给PPT制作人.txt').write_text(content,encoding='utf-8-sig')
    sections=''.join(f'<section><h2>{i+1:02d} {html.escape(t)}</h2><img src="{g.FILES[i]}.png"><p>{html.escape(s)}</p><a href="{g.FILES[i]}.svg">可编辑 SVG</a></section>' for i,(t,s) in enumerate(TEXTS))
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>batch52 配图与说明</title><style>body{font:18px/1.9 "Microsoft YaHei",sans-serif;background:#eff3f6;color:#17324d;margin:40px auto;max-width:1200px}section{padding:30px;background:white;margin:28px 0;border-radius:15px}img{width:100%}a{color:#287ab9}</style><h1>batch52 Njoint · 配图与说明</h1><p>7 张 PNG + 7 张可编辑 SVG；文字说明独立保存。</p>'+sections+'</html>'
    (OUT/'配图总览与说明.html').write_text(page,encoding='utf-8')
    g.contact_sheet()
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):z.write(p,p.name)
    print(OUT)


if __name__=='__main__':
    fig1();fig2();fig3();fig4();fig5();fig6();fig7();handoff()

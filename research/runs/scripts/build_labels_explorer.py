"""Build a standalone offline gene-label explorer using Python's standard library.

Canvas renders every original plotted gene; gzip-encoded inline row arrays keep
all database records searchable without network access or a JavaScript CDN.
"""
import base64
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/label_trace_20261001"
FIELDS = ["gene_id", "gene_symbol", "aliases", "Score_E", "Score_S", "Score_B", "E_raw", "S_raw", "B_raw", "Score_HKG",
          "tissue_count", "detectable_tissue_count", "gct_file_lines_1based", "label_source_line_1based",
          "promoter_first_source_line_1based", "promoter_record_count", "in_attached_density_plot", "reason_index",
          "in_gtex_label_database", "has_promoter_record", "gene_types_all", "gtex_gene_ids_versioned"]
NUMERIC = {"Score_E", "Score_S", "Score_B", "E_raw", "S_raw", "B_raw", "Score_HKG"}
INTEGER = {"tissue_count", "detectable_tissue_count", "label_source_line_1based", "promoter_first_source_line_1based",
           "promoter_record_count", "in_attached_density_plot", "in_gtex_label_database", "has_promoter_record"}

HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>全库基因标签浏览 · Promoter Atlas</title>
<style>
:root{--ink:#173542;--muted:#627681;--line:#dce7e9;--teal:#118f89;--amber:#dd8b22;--paper:#f4f8f8}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}button,input{font:inherit}button{cursor:pointer;border:1px solid var(--line);background:#fff;border-radius:8px;padding:7px 12px;color:var(--ink)}button:hover{border-color:var(--teal);background:#edf8f6}a{color:#176785;text-underline-offset:3px}header{max-width:1460px;margin:auto;padding:28px 30px 16px}h1{font-size:29px;line-height:1.2;margin:5px 0 12px;letter-spacing:-.5px}header p{margin:0;color:var(--muted)}.eyebrow{font-size:12px;letter-spacing:1.6px;color:var(--teal);font-weight:650}.counts{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}.count{background:white;border:1px solid var(--line);border-radius:9px;padding:8px 14px;font-size:13px}.count b{font-size:19px;margin-right:7px;font-variant-numeric:tabular-nums}main{max-width:1460px;margin:auto;padding:0 30px 28px;display:grid;grid-template-columns:minmax(500px,1fr) 355px;gap:18px;align-items:start}.chart-card,.side,.sources{border:1px solid var(--line);background:#fff;border-radius:14px}.chart-card{overflow:hidden}.chart-head{padding:17px 18px 5px;display:flex;gap:12px;align-items:flex-start;justify-content:space-between}.chart-head h2{font-size:17px;margin:0}.chart-head p{font-size:12px;color:var(--muted);margin:3px 0 0}.plot{position:relative;height:590px;touch-action:none}.plot canvas{position:absolute;inset:0;width:100%;height:100%;outline:none}.plot canvas:focus-visible{outline:2px solid var(--teal);outline-offset:-3px}.legend{display:flex;gap:20px;align-items:center;font-size:12px;padding:0 20px 17px;color:var(--muted)}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--teal);margin-right:6px}.ring{display:inline-block;width:11px;height:11px;border:2px solid var(--amber);border-radius:50%;margin-right:6px}.side{padding:17px}.searchlabel{font-weight:650;font-size:15px;display:block;margin-bottom:8px}.searchbar{display:flex;gap:7px}input{min-width:0;flex:1;border:1px solid #a9bdc2;border-radius:8px;padding:10px 11px;width:100%;color:var(--ink)}input:focus{outline:2px solid #77c7bc;outline-offset:1px}.quick{display:flex;gap:6px;margin:10px 0;flex-wrap:wrap}.quick button{font-size:12px;padding:4px 8px}.search-state{font-size:12px;color:var(--muted);min-height:22px}.results{max-height:170px;overflow-y:auto;margin:5px 0 12px}.result{display:flex;justify-content:space-between;align-items:center;width:100%;border:0;border-radius:6px;text-align:left;padding:7px 6px}.result small{color:var(--muted);font-size:10px}.result.active{background:#e9f5f3}.detail{border-top:1px solid var(--line);padding-top:14px}.detail h2{font-size:23px;line-height:1.2;margin:0 0 5px;overflow-wrap:anywhere}.gene-id{font-size:12px;color:var(--muted);font-family:ui-monospace,Consolas,monospace}.badge{display:inline-block;background:#eaf6f3;color:#13766d;padding:3px 8px;border-radius:20px;font-size:11px;margin:10px 0}.badge.absent{background:#fff2df;color:#955813}.scores{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-bottom:12px}.score{padding:9px;background:#f5f8f9;border-radius:8px}.score span{display:block;font-size:11px;color:var(--muted)}.score b{font-size:17px;font-weight:600;font-variant-numeric:tabular-nums}.reason{font-size:12px;background:#fff8ee;border-left:3px solid #e3a955;padding:8px 10px;margin-bottom:12px}dl{font-size:12px;margin:0}dl>div{display:grid;grid-template-columns:1fr 1.35fr;gap:9px;border-bottom:1px solid #edf1f2;padding:5px 0}dt{color:var(--muted)}dd{margin:0;text-align:right;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}.detail-links{display:flex;gap:11px;flex-wrap:wrap;font-size:12px;margin-top:13px}.export{font-size:12px;margin-top:12px;width:100%}.sources{grid-column:1/-1;padding:17px 20px;font-size:13px}.sources h2{font-size:15px;margin:0 0 8px}.source-links{display:flex;flex-wrap:wrap;gap:9px 20px}.sources p{font-size:12px;color:var(--muted);margin:9px 0 0}.tooltip{position:absolute;z-index:3;pointer-events:none;background:rgba(255,255,255,.98);border:1px solid #adc5c7;box-shadow:0 5px 20px #18343c22;border-radius:8px;padding:10px 12px;font-size:11px;max-width:290px;display:none}.tooltip b{font-size:13px}.tooltip .tip-id{font-size:10px;color:var(--muted)}.tooltip table{margin-top:5px;width:100%;border-collapse:collapse}.tooltip td:last-child{text-align:right;padding-left:13px;font-variant-numeric:tabular-nums}.loading{padding:18px;color:var(--muted)}.selection-note{font-size:12px;color:var(--muted);padding:0 19px 13px;min-height:31px}.tools{display:flex;gap:5px;flex-shrink:0}.tools button{font-size:12px;padding:5px 9px}footer{max-width:1460px;margin:auto;padding:0 30px 25px;font-size:11px;color:var(--muted)}@media(max-width:1020px){main{grid-template-columns:minmax(400px,1fr) 310px;gap:12px;padding:0 16px 24px}header{padding:22px 16px 16px}.plot{height:535px}.side{padding:13px}}@media(max-width:760px){main{grid-template-columns:1fr}.side{grid-row:1}.plot{height:440px}.results{max-height:120px}.detail dl{display:grid;grid-template-columns:1fr 1fr;gap:0 18px}.sources{grid-column:1}.count b{font-size:17px}h1{font-size:25px}header p{font-size:13px}.chart-head{padding:15px 12px 5px}.chart-head p{max-width:270px}.scores{max-width:360px}.detail{padding-top:12px}.quick{margin-bottom:5px}} 
</style></head><body>
<header><div class="eyebrow">PROMOTER ATLAS · CANDIDATE E</div><h1>全库基因标签浏览</h1><p>按基因符号或 ENSG 编号查找标签与原始数据位置。这是同一份数据的交互散点，<a href="report.html">原六边形图与重算报告</a>仍可查看。</p><div class="counts"><div class="count"><b id="total-count">79,852</b>全库可查询</div><div class="count"><b id="plot-count">18,947</b>原图入图基因</div><div class="count"><b id="missing-count">20,864</b>无 GTEx 表达标签</div></div></header>
<main><section class="chart-card"><div class="chart-head"><div><h2>表达强度 × 跨组织稳定度</h2><p>悬停查看标签 · 点击选择 · 滚轮缩放 · 拖动平移</p></div><div class="tools"><button id="zoom-in" title="放大">＋</button><button id="zoom-out" title="缩小">−</button><button id="reset">重置视图</button></div></div><div class="plot" id="plot"><canvas id="base"></canvas><canvas id="overlay" tabindex="0" aria-label="全部18947个入图基因的E与S散点；可使用右侧搜索查看任何基因"></canvas><div class="tooltip" id="tooltip"></div></div><div class="legend"><span><i class="dot"></i>原图入图基因</span><span><i class="ring"></i>当前选择</span></div><div id="selection-note" class="selection-note"></div></section>
<aside class="side"><label for="search" class="searchlabel">查找基因</label><div class="searchbar"><input id="search" placeholder="ACTB / GAPDH / ENSG…" autocomplete="off" spellcheck="false"><button id="clear" title="清空搜索">清空</button></div><div class="quick"><button data-query="ACTB">ACTB</button><button data-query="GAPDH">GAPDH</button><button data-query="TNMD">缺失 S 示例</button><button data-query="NME1-NME2">无 GTEx 示例</button></div><div id="search-state" class="search-state">正在读取本页内嵌数据…</div><div id="results" class="results"></div><div id="detail" class="detail"><div class="loading">正在加载标签；无需网络连接。</div></div></aside>
<section class="sources"><h2>数据与追溯入口</h2><div class="source-links"><a href="gene_labels_all_database.tsv">全库逐基因标签</a><a href="tissue_mean_tpm_used_for_labels.tsv">标签所用逐组织 TPM</a><a href="raw_gct_gene_rows.tsv">原始 GCT 基因行索引</a><a href="duplicate_resolution.tsv">重复行处理记录</a><a href="promoter_gene_label_mapping.tsv">启动子—基因映射</a><a href="source_manifest.tsv">原始文件路径与 SHA256</a><a href="provenance.json">来源与重算核验</a><a href="report.html">原图与完整报告</a></div><p>显示的是数据库来源标签，不是预测模型输出。Score_B 是分位得分；B_raw 是 TPM ≥ 1 的组织占比，两者不可混用。S 是跨组织稳定度，不代表培养时间稳定性。缺失标签保留为空，不填 0。</p><p>数值沿用原表的 6 位小数，点坐标与原图一致；六边形密度与散点叠加的显示方式不同。标签原文件行号、GCT 原文件行号和表达表记录范围均可追溯至同目录表格。</p></section></main>
<footer>本页数据与功能均已内嵌，可离线运行。需要支持 DecompressionStream 的现代 Chrome、Edge、Firefox 或 Safari；追溯表格需与本页保存在同一目录。</footer>
<script id="dataset" type="application/octet-stream">__PAYLOAD__</script>
<script>
(async function(){'use strict';
const el=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),fmt=x=>x===null?'—':Number(x).toFixed(6);
try{
const bytes=Uint8Array.from(atob(el('dataset').textContent.trim()),c=>c.charCodeAt(0));
const unpacked=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
const data=JSON.parse(new TextDecoder().decode(unpacked)),rows=data.rows,C=Object.fromEntries(data.fields.map((v,i)=>[v,i]));
const plotted=rows.map((r,i)=>r[C.in_attached_density_plot]===1?i:-1).filter(i=>i>=0),byID=new Map(rows.map((r,i)=>[r[C.gene_id],i]));
const norm=s=>String(s??'').toUpperCase().trim(),names=rows.map(r=>norm(r[C.gene_symbol])),aliases=rows.map(r=>norm(r[C.aliases]).split(';').filter(Boolean));
const reasonText={'B_raw_below_0.30':'可检出组织占比 B_raw 低于 0.30',detectable_tissues_below_3:'可检出组织少于 3 个',Score_S_undefined:'Score_S 未定义',no_GTEx_expression_label:'GTEx 来源中无该基因的表达标签'};
const why=r=>data.reasons[r[C.reason_index]].split('|').filter(Boolean).map(s=>reasonText[s]||s).join('；');
let selected=null,hovered=null,view={x:[...data.geometry.xlim],y:[...data.geometry.ylim]},W=0,H=0,dpr=1,bins=new Map(),screen=new Map(),drag=null,moved=false,scheduled=false;
const plot=el('plot'),base=el('base'),overlay=el('overlay'),ctx=base.getContext('2d'),over=overlay.getContext('2d'),margin={l:68,r:20,t:20,b:58};
const limits={x:[...data.geometry.xlim],y:[...data.geometry.ylim]};
function search(query){let q=norm(query);if(/^ENSG\d+\.\d+$/.test(q))q=q.replace(/\.\d+$/,'');if(!q)return[];if(byID.has(q))return[byID.get(q)];let exact=[],partial=[];for(let i=0;i<rows.length;i++){if(names[i]===q||aliases[i].includes(q))exact.push(i);else if(names[i].includes(q)||rows[i][C.gene_id].includes(q)||aliases[i].some(a=>a.includes(q)))partial.push(i);}return exact.length?exact:partial;}
function updateSearch(query,autoSelect=false){const hits=search(query);el('search-state').textContent=query.trim()?`找到 ${hits.length.toLocaleString()} 个基因${hits.length>60?'；先显示前 60 个，请用 ENSG 编号精确定位':''}`:'输入基因符号或 ENSG 编号；同名基因会全部列出。';el('results').innerHTML=hits.slice(0,60).map(i=>`<button class="result ${i===selected?'active':''}" data-index="${i}"><span>${esc(rows[i][C.gene_symbol]||'未命名')}</span><small>${esc(rows[i][C.gene_id])}</small></button>`).join('');if(autoSelect&&hits.length)select(hits[0]);return hits;}
function detailRow(label,value){return `<div><dt>${label}</dt><dd>${esc(value===null||value===''?'—':value)}</dd></div>`;}
function select(index){selected=index;const r=rows[index],inplot=r[C.in_attached_density_plot]===1;el('detail').innerHTML=`<h2>${esc(r[C.gene_symbol]||'未命名基因')}</h2><div class="gene-id">${esc(r[C.gene_id])}</div><div class="badge ${inplot?'':'absent'}">${inplot?'已进入原图':'未进入原图'}</div><div class="scores">${['E','S','B'].map(k=>`<div class="score"><span>Score_${k}</span><b>${fmt(r[C['Score_'+k]])}</b></div>`).join('')}</div>${inplot?'':`<div class="reason">${esc(why(r))}</div>`}<dl>${detailRow('E_raw',fmt(r[C.E_raw]))}${detailRow('S_raw',fmt(r[C.S_raw]))}${detailRow('B_raw',fmt(r[C.B_raw]))}${detailRow('Score_HKG',fmt(r[C.Score_HKG]))}${detailRow('可检出 / 总组织数',r[C.tissue_count]===null?'—':`${r[C.detectable_tissue_count]} / ${r[C.tissue_count]}`)}${detailRow('GCT 原文件行号',r[C.gct_file_lines_1based])}${detailRow('标签原文件行号',r[C.label_source_line_1based])}${detailRow('启动子记录数',r[C.promoter_record_count])}${detailRow('启动子原表首行',r[C.promoter_first_source_line_1based])}${detailRow('GTEx 基因版本号',r[C.gtex_gene_ids_versioned])}${detailRow('基因类型',r[C.gene_types_all])}</dl><div class="detail-links"><a href="raw_gct_gene_rows.tsv">GCT 行索引</a><a href="tissue_mean_tpm_used_for_labels.tsv">逐组织原值</a><a href="source_manifest.tsv">来源与哈希</a></div><button class="export" id="download-gene">下载此基因标签（TSV）</button>`;
el('selection-note').textContent=inplot?`${r[C.gene_symbol]} · E = ${fmt(r[C.Score_E])}，S = ${fmt(r[C.Score_S])}`:`${r[C.gene_symbol]} 未进入原图：${why(r)}。未创建替代坐标。`;
document.querySelectorAll('.result').forEach(b=>b.classList.toggle('active',Number(b.dataset.index)===index));el('download-gene').onclick=()=>{const fields=data.fields.filter(f=>f!=='reason_index'),values=fields.map(f=>r[C[f]]??'');fields.push('plot_exclusion_reason');values.push(data.reasons[r[C.reason_index]]);const clean=x=>String(x).replace(/[\t\r\n]/g,' '),tsv='\ufeff'+fields.join('\t')+'\n'+values.map(clean).join('\t')+'\n';const url=URL.createObjectURL(new Blob([tsv],{type:'text/tab-separated-values;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=r[C.gene_id]+'_labels.tsv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};paintOverlay();return getSelected();}
function xy(r){return [margin.l+(r[C.Score_E]-view.x[0])/(view.x[1]-view.x[0])*(W-margin.l-margin.r),H-margin.b-(r[C.Score_S]-view.y[0])/(view.y[1]-view.y[0])*(H-margin.t-margin.b)];}
function tickStep(span){const wanted=span/5,pow=10**Math.floor(Math.log10(wanted));return [1,2,5,10].map(x=>x*pow).find(x=>x>=wanted);}
function ticks(range){const step=tickStep(range[1]-range[0]),out=[];for(let t=Math.ceil(range[0]/step)*step;t<=range[1]+step*1e-7;t+=step)out.push(t);return[out,step];}
function render(){scheduled=false;const rect=plot.getBoundingClientRect();W=rect.width;H=rect.height;dpr=Math.min(window.devicePixelRatio||1,2);for(const canvas of[base,overlay]){canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr);}ctx.setTransform(dpr,0,0,dpr,0,0);over.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);ctx.font='12px system-ui,"Microsoft YaHei",sans-serif';ctx.lineWidth=1;ctx.strokeStyle='#e5edef';ctx.fillStyle='#647b85';const [xt,xs]=ticks(view.x),[yt,ys]=ticks(view.y);ctx.textAlign='center';ctx.textBaseline='top';for(const t of xt){const x=margin.l+(t-view.x[0])/(view.x[1]-view.x[0])*(W-margin.l-margin.r);ctx.beginPath();ctx.moveTo(x,margin.t);ctx.lineTo(x,H-margin.b);ctx.stroke();ctx.fillText(t.toFixed(Math.min(4,Math.max(2,-Math.floor(Math.log10(xs))))),x,H-margin.b+9);}ctx.textAlign='right';ctx.textBaseline='middle';for(const t of yt){const y=H-margin.b-(t-view.y[0])/(view.y[1]-view.y[0])*(H-margin.t-margin.b);ctx.beginPath();ctx.moveTo(margin.l,y);ctx.lineTo(W-margin.r,y);ctx.stroke();ctx.fillText(t.toFixed(Math.min(4,Math.max(2,-Math.floor(Math.log10(ys))))),margin.l-9,y);}ctx.strokeStyle='#34525f';ctx.lineWidth=1.3;ctx.beginPath();ctx.moveTo(margin.l,margin.t);ctx.lineTo(margin.l,H-margin.b);ctx.lineTo(W-margin.r,H-margin.b);ctx.stroke();ctx.fillStyle='#294956';ctx.font='14px system-ui,"Microsoft YaHei",sans-serif';ctx.textAlign='center';ctx.fillText('表达强度分位得分（Score_E）',(margin.l+W-margin.r)/2,H-17);ctx.save();ctx.translate(19,(margin.t+H-margin.b)/2);ctx.rotate(-Math.PI/2);ctx.fillText('跨组织稳定度分位得分（Score_S）',0,0);ctx.restore();bins=new Map();screen=new Map();ctx.save();ctx.beginPath();ctx.rect(margin.l,margin.t,W-margin.l-margin.r,H-margin.t-margin.b);ctx.clip();ctx.fillStyle='rgba(15,145,137,0.30)';for(const i of plotted){const r=rows[i];if(r[C.Score_E]<view.x[0]||r[C.Score_E]>view.x[1]||r[C.Score_S]<view.y[0]||r[C.Score_S]>view.y[1])continue;const [x,y]=xy(r);ctx.fillRect(x-1.05,y-1.05,2.1,2.1);screen.set(i,[x,y]);const k=Math.floor(x/16)+','+Math.floor(y/16);if(!bins.has(k))bins.set(k,[]);bins.get(k).push(i);}ctx.restore();paintOverlay();}
function schedule(){if(!scheduled){scheduled=true;requestAnimationFrame(render);}}
function paintOverlay(){over.clearRect(0,0,W,H);for(const[index,color,radius]of[[hovered,'#183d50',4],[selected,'#df8b22',7]]){if(index===null||!screen.has(index))continue;const[x,y]=screen.get(index);over.strokeStyle=color;over.lineWidth=2;over.beginPath();over.arc(x,y,radius,0,Math.PI*2);over.stroke();if(index===selected){const name=rows[index][C.gene_symbol];over.font='600 12px system-ui,"Microsoft YaHei",sans-serif';over.textAlign=x>W-110?'right':'left';const labelx=x>W-110?x-10:x+10;over.fillStyle='#ffffff';over.strokeStyle='#ffffff';over.lineWidth=4;over.strokeText(name,labelx,y-10);over.fillStyle='#80500b';over.fillText(name,labelx,y-10);}}}
function nearest(x,y){const bx=Math.floor(x/16),by=Math.floor(y/16);let best=null,distance=64;for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(const i of bins.get((bx+dx)+','+(by+dy))||[]){const[p,q]=screen.get(i),d=(p-x)**2+(q-y)**2;if(d<distance){best=i;distance=d;}}return best;}
function showTooltip(index,x,y){const tip=el('tooltip');if(index===null){tip.style.display='none';return;}const r=rows[index];tip.innerHTML=`<b>${esc(r[C.gene_symbol])}</b><div class="tip-id">${esc(r[C.gene_id])}</div><table>${['Score_E','Score_S','Score_B','E_raw','S_raw','B_raw'].map(k=>`<tr><td>${k}</td><td>${fmt(r[C[k]])}</td></tr>`).join('')}<tr><td>GCT 原行号</td><td>${esc(r[C.gct_file_lines_1based])}</td></tr><tr><td>入图状态</td><td>已进入原图</td></tr></table>`;tip.style.display='block';tip.style.left=Math.max(4,Math.min(x+14,W-tip.offsetWidth-7))+'px';tip.style.top=Math.max(3,Math.min(y+14,H-tip.offsetHeight-5))+'px';}
function bound(range,allowed){const span=Math.min(range[1]-range[0],allowed[1]-allowed[0]);let lo=Math.max(allowed[0],Math.min(range[0],allowed[1]-span));return[lo,lo+span];}
function zoom(factor,x=W/2,y=H/2){const fx=Math.max(0,Math.min(1,(x-margin.l)/(W-margin.l-margin.r))),fy=Math.max(0,Math.min(1,1-(y-margin.t)/(H-margin.t-margin.b)));for(const[k,f]of[['x',fx],['y',fy]]){const anchor=view[k][0]+f*(view[k][1]-view[k][0]),span=Math.max(.001,(view[k][1]-view[k][0])*factor);view[k]=bound([anchor-f*span,anchor+(1-f)*span],limits[k]);}schedule();}
function eventXY(event){const r=plot.getBoundingClientRect();return[event.clientX-r.left,event.clientY-r.top];}
overlay.onpointermove=event=>{const[x,y]=eventXY(event);if(drag){const dx=x-drag.x,dy=y-drag.y;if(Math.abs(dx)+Math.abs(dy)>3)moved=true;const sx=-dx/(W-margin.l-margin.r)*(drag.view.x[1]-drag.view.x[0]),sy=dy/(H-margin.t-margin.b)*(drag.view.y[1]-drag.view.y[0]);view.x=bound(drag.view.x.map(v=>v+sx),limits.x);view.y=bound(drag.view.y.map(v=>v+sy),limits.y);el('tooltip').style.display='none';schedule();return;}hovered=nearest(x,y);overlay.style.cursor=hovered===null?'grab':'pointer';showTooltip(hovered,x,y);paintOverlay();};
overlay.onpointerdown=event=>{const[x,y]=eventXY(event);drag={x,y,view:{x:[...view.x],y:[...view.y]}};moved=false;overlay.setPointerCapture(event.pointerId);};overlay.onpointerup=event=>{if(!moved){const[x,y]=eventXY(event),index=nearest(x,y);if(index!==null){el('search').value=rows[index][C.gene_id];updateSearch(el('search').value);select(index);}}drag=null;};overlay.onpointercancel=()=>{drag=null;};overlay.onpointerleave=()=>{if(!drag){hovered=null;el('tooltip').style.display='none';paintOverlay();}};overlay.addEventListener('wheel',event=>{event.preventDefault();const[x,y]=eventXY(event);zoom(event.deltaY<0?.8:1.25,x,y);},{passive:false});
overlay.onkeydown=event=>{if(event.key==='+'||event.key==='='){zoom(.8);event.preventDefault();}if(event.key==='-'){zoom(1.25);event.preventDefault();}if(event.key==='Home'){reset();event.preventDefault();}};
function reset(){view={x:[...limits.x],y:[...limits.y]};render();}el('reset').onclick=reset;el('zoom-in').onclick=()=>zoom(.8);el('zoom-out').onclick=()=>zoom(1.25);let timer;el('search').oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>updateSearch(el('search').value,true),100);};el('search').onkeydown=e=>{if(e.key==='Enter')updateSearch(el('search').value,true);};el('clear').onclick=()=>{el('search').value='';updateSearch('');el('search').focus();};el('results').onclick=e=>{const button=e.target.closest('[data-index]');if(button)select(Number(button.dataset.index));};document.querySelectorAll('[data-query]').forEach(button=>button.onclick=()=>{el('search').value=button.dataset.query;updateSearch(button.dataset.query,true);});
function getSelected(){if(selected===null)return null;const r=rows[selected],result=Object.fromEntries(data.fields.map((f,i)=>[f,r[i]]));result.plot_exclusion_reason=data.reasons[r[C.reason_index]];return result;}
new ResizeObserver(schedule).observe(plot);el('total-count').textContent=rows.length.toLocaleString();el('plot-count').textContent=plotted.length.toLocaleString();el('missing-count').textContent=rows.filter(r=>!r[C.in_gtex_label_database]).length.toLocaleString();render();el('search').value='ACTB';updateSearch('ACTB',true);
window.LabelExplorer={ready:true,counts:{all:rows.length,plotted:plotted.length,noGTEx:rows.filter(r=>!r[C.in_gtex_label_database]).length},search:q=>search(q).map(i=>({index:i,gene_id:rows[i][C.gene_id],gene_symbol:rows[i][C.gene_symbol]})),selectByID:id=>select(byID.get(id)),getSelected,sourceSHA256:data.source_sha256,plottedCoordinates:()=>plotted.map(i=>[rows[i][C.gene_id],rows[i][C.Score_E],rows[i][C.Score_S]]),screenPoint:id=>screen.get(byID.get(id)),view:()=>view};
}catch(error){el('detail').textContent='数据加载失败：'+error.message+'。请使用现代浏览器，或直接打开下方全库 TSV 表。';el('search-state').textContent='加载失败';window.LabelExplorer={ready:false,error:String(error)};console.error(error);}
})();
</script></body></html>'''


def main():
    source = OUT / "gene_labels_all_database.tsv"
    original = list(csv.DictReader(source.open(encoding="utf-8-sig", newline=""), delimiter="\t"))
    geometry = json.loads((OUT / "plot_geometry.json").read_text(encoding="utf-8"))
    reasons = sorted({row["plot_exclusion_reason"] for row in original})
    rows = []
    for raw in original:
        alias = sorted(set((raw["promoter_gene_symbols_all"] + ";" + raw["gtex_descriptions_all"]).split(";")) - {"", raw["gene_symbol"]})
        row = []
        for name in FIELDS:
            if name == "aliases":
                value = ";".join(alias)
            elif name == "reason_index":
                value = reasons.index(raw["plot_exclusion_reason"])
            elif name in NUMERIC:
                value = float(raw[name]) if raw[name] else None
            elif name in INTEGER:
                value = int(raw[name]) if raw[name] else None
            else:
                value = raw[name]
            row.append(value)
        rows.append(row)
    data = {"fields": FIELDS, "rows": rows, "reasons": reasons, "geometry": geometry,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    html = HTML.replace("__PAYLOAD__", encoded)
    path = OUT / "labels_explorer.html"
    path.write_text(html, encoding="utf-8")
    C = {name: i for i, name in enumerate(FIELDS)}
    plotted = [row for row in rows if row[C["in_attached_density_plot"]] == 1]
    assert len(rows) == 79852 and len(plotted) == 18947
    assert len({row[C["gene_id"]] for row in rows}) == len(rows)
    assert sum(not row[C["in_gtex_label_database"]] for row in rows) == 20864
    for row, raw in zip(rows, original):
        for field in NUMERIC:
            assert row[C[field]] == (float(raw[field]) if raw[field] else None)
        if row[C["in_attached_density_plot"]]:
            assert row[C["Score_E"]] is not None and row[C["Score_S"]] is not None
    assert json.loads(gzip.decompress(base64.b64decode(encoded))) == data
    examples = {raw["gene_symbol"]: {name: raw[name] for name in ["gene_id", "Score_E", "Score_S", "Score_B", "E_raw", "S_raw", "B_raw", "gct_file_lines_1based", "plot_exclusion_reason"]}
                for raw in original if raw["gene_symbol"] in {"ACTB", "GAPDH", "TNMD", "NME1-NME2"}}
    result = {"status": "structural_checks_passed_browser_check_pending", "source_path": str(source),
              "source_sha256": data["source_sha256"], "html_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "schema": FIELDS, "all_gene_count": len(rows), "plot_point_count": len(plotted),
              "all_original_numeric_values_match": True, "point_coordinates_use_original_values": True,
              "plotted_coordinate_sha256": hashlib.sha256('\n'.join(f'{row[C["gene_id"]]}\t{row[C["Score_E"]]:.6f}\t{row[C["Score_S"]]:.6f}' for row in plotted).encode()).hexdigest(),
              "nulls_preserved_not_imputed": True, "source_tables_modified": False,
              "exclusion_reasons": dict(Counter(raw["plot_exclusion_reason"] for raw in original)),
              "example_expectations": examples, "payload_json_bytes": len(payload), "compressed_bytes": len(compressed),
              "html_bytes": path.stat().st_size, "network_dependencies": [],
              "runtime": "HTML Canvas 2D + browser DecompressionStream; no external JavaScript libraries",
              "builder": str(Path(__file__).resolve()), "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "labels_explorer_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["all_gene_count", "plot_point_count", "payload_json_bytes", "compressed_bytes", "html_bytes"]}, indent=2))


if __name__ == "__main__":
    main()

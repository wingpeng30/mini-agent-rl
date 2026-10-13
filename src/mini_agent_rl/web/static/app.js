const $ = id => document.getElementById(id);
let runtime = null, examples = [], activeJob = null, currentResult = null, rendering = false;
const modelNames = {"goal-z":"Goal Z · 已发布权重", "am-research":"AM · 未发布研究候选"};
const retrievalNames={"bm25-v2":"BM25 关键词检索","dense-v1":"MiniLM 向量检索","hybrid-rrf-v1":"混合检索 · BM25 + 向量 / RRF"};
const caseNames = {"musique:2hop__796221_18803":"01  Rapana · 后续取证", "2wiki:f299dd600bdd11eba7f7acde48001122":"02  Medinaceli · 及时回答", "musique:2hop__96194_78303":"03  Hero / Walker · 绑定失败"};
function node(tag, text, cls) { const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(cls)n.className=cls; return n; }
function error(message) { $("error").textContent=message||""; $("error").hidden=!message; }
async function api(path, body) {
  const response = await fetch(path, body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const data=await response.json(); if(!response.ok)throw new Error(typeof data.detail==="string"?data.detail:JSON.stringify(data.detail)); return data;
}
function fillCorpora(corpora) { const previous=$("corpus").value; $("corpus").replaceChildren(); corpora.forEach(c=>{const option=node("option",c.label);option.value=c.id;$("corpus").append(option)}); if(corpora.some(c=>c.id===previous))$("corpus").value=previous; }
async function refresh() {
  try {runtime=await api("/api/status"); const names={idle:"尚未加载",loading:"正在加载模型…",ready:"模型已就绪",error:"模型加载失败"};$("status-text").textContent=names[runtime.state];$("status-dot").className="dot "+runtime.state;$("runtime-detail").textContent=runtime.model_id?modelNames[runtime.model_id]:"本地 CUDA / BF16";$("run").disabled=runtime.busy||runtime.state!=="ready"||runtime.model_id!==$("model").value;$("model").disabled=runtime.busy;$("retriever").disabled=!!activeJob;$("corpus").disabled=!!activeJob;fillCorpora(runtime.corpora);$("retrieval-status").textContent=runtime.retrieval?.ready?"向量索引就绪 · top-3":"向量索引未就绪 · 可使用 BM25";if(runtime.error)error(runtime.error);if(activeJob)await pollJob();}
  catch(e){error("连接服务失败："+e.message);$("status-text").textContent="服务未连接";$("run").disabled=true;}
}
function renderStep(step) {
  const container=node("div",undefined,"step"), head=node("div",undefined,"step-head");head.append(node("span",String(step.step+1).padStart(2,"0"),"step-number"),node("span",step.action==="search"?"搜索必要关系":step.action==="answer"?"生成答案":"原始动作","step-title"),node("span",`${step.input_tokens||0} 输入 / ${step.output_tokens||0} 输出 token`,"step-meta"));container.append(head);
  if(step.query)container.append(node("p",step.query,"query"));
  if(step.answer)container.append(node("p",step.answer,"query"));
  (step.documents||[]).forEach((d,i)=>{const detail=node("details",undefined,"doc");detail.append(node("summary",`${i+1}. ${d.title||"无标题"} · score ${Number(d.score||0).toFixed(3)}`),node("small",`DOCUMENT ${d.id}`),node("p",d.text));container.append(detail)});
  if(step.raw){const raw=node("details",undefined,"raw");raw.append(node("summary","原始JSON输出"),node("pre",step.raw));container.append(raw)}
  if(step.repeated)container.append(node("p","检测到重复查询；保留实际失败表现。","error"));
  if(step.parse_error)container.append(node("p",step.parse_error,"error"));return container;
}
function render(result, pending=false) {
  currentResult=result;const area=$("output");area.replaceChildren();const head=node("div",undefined,"result-head");head.append(node("span",result.mode==="replay"?"历史回放":"实时推理","pill "+(result.mode==="replay"?"replay":"")),node("span",modelNames[result.model_id],"small-label"));area.append(head,node("div",result.question,"result-question"));if(result.retrieval)area.append(node("p",retrievalNames[result.retrieval.retriever]||result.retrieval.retriever,"small-label"));
  if(!pending){const answer=node("div",undefined,"answer-box");answer.append(node("small","FINAL ANSWER"),node("p",result.answer??`未输出答案 · ${result.reason}`));area.append(answer);const stats=node("div",undefined,"stats");stats.append(node("span",`${result.model_calls} 次调用`),node("span",`${result.searches} 次搜索`),node("span",`${result.elapsed_seconds.toFixed(2)} 秒${result.historical_latency?" · 历史耗时":" · 实测"}`),node("span",`停止：${result.reason}`));area.append(stats);}
  result.steps.forEach(s=>area.append(renderStep(s)));
  if(pending){const progress=node("div",undefined,"running");progress.append(node("span",undefined,"spinner"),node("span","模型正在执行搜索与回答…"));area.append(progress);}
  if(result.adapter_sha256){area.append(node("div",`ADAPTER ${result.adapter_sha256}\nCORPUS ${result.corpus.sha256}`,"provenance"));}
}
async function pollJob() {
  const job=await api(`/api/runs/${activeJob}`);
  if(job.state==="complete"){activeJob=null;render(job.result);return;}
  if(job.state==="error"){activeJob=null;error(job.error);return;}
  const steps=[];for(const event of job.events){if(event.type==="action")steps.push({step:steps.length,action:event.query?"search":"answer",query:event.query,answer:event.answer,raw:event.raw,documents:[]});else if(steps.length)steps[steps.length-1].documents=event.documents;}
  render({mode:"live",model_id:$("model").value,question:$("question").value,steps,retrieval:{retriever:$("retriever").value}},true);
}
$("model").addEventListener("change",async()=>{error(null);try{await api("/api/model/load",{model_id:$("model").value});await refresh()}catch(e){error(e.message)}});
$("run").addEventListener("click",async()=>{error(null);try{const job=await api("/api/runs",{question:$("question").value,model_id:$("model").value,corpus_id:$("corpus").value,retriever:$("retriever").value});activeJob=job.id;$("run").disabled=true;await pollJob()}catch(e){error(e.message)}});
$("search-only").addEventListener("click",async()=>{error(null);$("search-only").disabled=true;try{const result=await api("/api/search",{query:$("question").value,corpus_id:$("corpus").value,retriever:$("retriever").value});const area=$("output");area.replaceChildren(node("h3",retrievalNames[result.retrieval.retriever]),node("p",result.query),node("p",`${result.elapsed_seconds.toFixed(3)} 秒 · 文档检索，无 Agent 推理`));area.append(renderStep({step:0,action:"search",documents:result.documents}));area.append(node("div",`CORPUS ${result.retrieval.corpus_sha256}\nINDEX ${result.retrieval.index_sha256||"BM25"}`,"provenance"));}catch(e){error(e.message)}finally{$("search-only").disabled=false}});
$("corpus").addEventListener("change",()=>{if($("corpus").value!=="goal-z-dev")$("retriever").value="bm25-v2"});
$("file").addEventListener("change",async e=>{const file=e.target.files[0];if(!file)return;error(null);try{const documents=JSON.parse(await file.text());if(!Array.isArray(documents))throw new Error("语料JSON必须是文档数组，每篇包含id、title、text字符串。");const value=await api("/api/corpora",{label:file.name,documents});await refresh();$("corpus").value=value.id;$("retriever").value="bm25-v2"}catch(e){error(e.message)}finally{$("file").value=""}});
async function init(){try{examples=await api("/api/examples");const seen=new Set();examples.forEach(example=>{if(seen.has(example.case_id))return;seen.add(example.case_id);const button=node("button",caseNames[example.case_id]||example.question);button.addEventListener("click",async()=>{error(null);try{const result=await api(`/api/replay/${example.case_id}?model_id=${$("model").value}`);$("question").value=result.question;render(result)}catch(e){error(e.message)}});$("examples").append(button)});await refresh();}catch(e){error(e.message)}setInterval(refresh,1200);}
init();

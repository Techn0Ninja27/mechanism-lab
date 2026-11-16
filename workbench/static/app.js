'use strict';
const S = {view:'overview', projects:[], project:null, docs:[], runs:[], jobs:[], studies:[], generated:[], operators:null, lastSourcePack:null, packSelected:new Set(), run:'proposals-v2', candidates:[], selected:null, offset:0, total:0};
const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let candidateRequest = 0;
let pdfURL;
async function api(path, options={}) {
  const response = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
  let body; try { body=await response.json(); } catch { throw Error('The server returned an unreadable response.'); }
  if (!response.ok) throw Error(body.error || body.reason || `Request failed (${response.status})`);
  return body;
}
const post = (path, body) => api(path, {method:'POST', body:JSON.stringify(body)});
function say(message, error=false) {
  const box=$('#notice'); box.textContent=message; box.classList.add('show');
  box.classList.toggle('is-error',error); box.setAttribute('role',error?'alert':'status');
  if (!error) setTimeout(()=>box.classList.remove('show'),5000);
}
function fail(error) { say(error.message || String(error), true); }
function markdown(text) {
  const blocks=[];
  const safe=esc(text).replace(/```[^\n]*\n([\s\S]*?)```/g,(_,code)=>{blocks.push(`<pre><code>${code}</code></pre>`);return `\u0000${blocks.length-1}\u0000`;});
  return safe.split(/\n\s*\n/).map(block=>{
    if (/^\u0000\d+\u0000$/.test(block)) return blocks[Number(block.slice(1,-1))];
    return block.split('\n').map(line=>{
      const heading=line.match(/^(#{1,6}) (.*)$/);if(heading)return `<h${Math.min(heading[1].length+1,6)}>${heading[2]}</h${Math.min(heading[1].length+1,6)}>`;
      return line.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/`([^`]+)`/g,'<code>$1</code>');
    }).join('<br>');
  }).map(block=>/^<(h\d|pre)/.test(block)?block:`<p>${block}</p>`).join('');
}
function modal(title, content) {
  let dialog=$('#detail-dialog');if(!dialog){dialog=document.createElement('dialog');dialog.id='detail-dialog';document.body.append(dialog);}
  dialog.innerHTML=`<form method="dialog"><button class="button secondary">Close</button></form><h2>${esc(title)}</h2>${content}`;
  dialog.showModal();return dialog;
}
function go(view) {
  S.view=view;document.querySelectorAll('.screen').forEach(x=>x.classList.toggle('hidden',x.id!==`screen-${view}`));
  document.querySelectorAll('.nav').forEach(x=>x.classList.toggle('active',x.dataset.view===view));
  $('#page-title').textContent={overview:'Overview',documents:'Documents',runs:'Engine runs',candidates:'Candidates',operators:'Operator library',studies:'Implemented studies',generated:'Generated research',reviews:'Reviews',setup:'Setup & folders'}[view];
  ({documents:documents,runs:refreshRuns,candidates:candidates,operators:operatorLibrary,studies:studies,generated:generatedResearch,reviews:reviews,setup:setup,overview:documents}[view] || (()=>{}))().catch(fail);
}
function ensureOperatorSurface() {
  if (!$('[data-view="operators"]')) {
    const setupNav=$('[data-view="setup"]'); const button=document.createElement('button');
    button.className='nav'; button.dataset.view='operators'; button.textContent='Operator library ';
    setupNav?.before(button);
  }
  if (!$('#screen-operators')) {
    const section=document.createElement('section'); section.className='screen hidden'; section.id='screen-operators';
    section.innerHTML='<div class="screen-head"><div><h1>Operator library</h1><p>Hand-curated, source-grounded operators and their typed chains. This registry is descriptive only: scientific novelty is not established here.</p></div></div><div class="validity"><span class="validity-icon">i</span><div><b>Read-only composition registry</b><p>No macro-atom promotion, execution, or novelty claim is inferred from these records.</p></div></div><article class="run-card"><h2>Local reversible-chain check</h2><p>Run a bounded lossless check on signed 64-bit integers. This stays local and does not establish a full study or novelty.</p><label>Operator IDs<input id="operator-check-ids" value="for,lzma" aria-label="Operator IDs"></label><label>Values <span class="muted">(JSON array)</span><textarea id="operator-check-values" rows="3" aria-label="Integer values">[1, 2, 3]</textarea></label><button class="button primary" data-action="operator-check">Check lossless chain (local)</button><div id="operator-check-result" class="muted" aria-live="polite"></div></article><div id="operator-library" class="stack"></div>';
    $('.content')?.append(section);
  }
}
function renderProjects() {
  $('#project-select').innerHTML=S.projects.length?S.projects.map(p=>`<option value="${esc(p.id)}">${esc(p.name)}</option>`).join(''):'<option value="">No projects yet</option>';
  if(S.project)$('#project-select').value=S.project.id;
  $('#project-label').textContent=S.project?.name || 'Create a project to begin';
  localStorage.setItem('alien-project',S.project?.id || '');
}
async function load() {
  try {
    ensureOperatorSurface();
    const [summary,projects,runs,studyData,generatedData,operatorData]=await Promise.all([api('/api/summary'),api('/api/projects'),api('/api/runs'),api('/api/studies'),api('/api/generated-research'),api('/api/operator-library')]);
    S.projects=projects.items;S.runs=runs.runs;S.studies=studyData.items;S.generated=generatedData.items;S.operators=operatorData;
    S.project=S.projects.find(p=>p.id===(S.project?.id || localStorage.getItem('alien-project'))) || S.projects[0] || null;
    $('#metrics').innerHTML=[['Selected candidates (v2)',summary.counts?.candidates],['Completed drafts (v2)',summary.counts?.proposals],['Repaired source atoms',summary.counts?.source_units_repaired],['Ranking validity',summary.provenance?.status]].map(([label,value])=>`<div class="metric"><small>${esc(label)}</small><b>${esc(value ?? '—')}</b></div>`).join('');
    renderProjects();populateRuns();await documents();if(S.view==='runs')await refreshRuns();
  } catch(error){fail(error);}
}
function sourceLink(url) {
  const text=esc(url||'not recorded');
  return typeof url==='string' && /^https:\/\//.test(url) ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${text}</a>` : text;
}
async function operatorLibrary() {
  const data=await api('/api/operator-library'); S.operators=data;
  const operators=data.operators||[], chains=data.chains||[], target=$('#operator-library'); if(!target)return;
  const byId=new Map(operators.map(item=>[item.id,item]));
  const opCards=operators.map(item=>`<article class="run-card"><h2>${esc(item.id||'Unnamed operator')} <small>${esc(item.content_id||'')}</small></h2><p><b>${esc(item.input_type||'unknown')}</b> → <b>${esc(item.output_type||'unknown')}</b> · contract: ${esc(item.contract_status||'not recorded')}</p><p>${(Array.isArray(item.assumptions)?item.assumptions:[]).map(x=>esc(x)).join(' · ')||'No assumptions recorded.'}</p><p class="muted">Source: ${sourceLink(item.source_url)}</p></article>`).join('');
  const chainRows=chains.map(chain=>`<article class="run-card"><h2>${esc(chain.id||'Unnamed chain')} <small>${esc(chain.molecule_id||'')}</small></h2><p class="pipeline"><span>${esc(chain.input_type||'unknown')}</span> ${(chain.operator_ids||[]).map((id,i)=>`${i?' → ':''}<b>${esc(byId.get(id)?.id||id)}</b>`).join('')} → <span>${esc(chain.output_type||'unknown')}</span></p><p class="muted">${(chain.operator_ids||[]).length} operator${(chain.operator_ids||[]).length===1?'':'s'} · typed boundaries shown from catalog metadata</p></article>`).join('');
  target.innerHTML=`<p class="library-boundary">${esc(data.scope||'Catalog scope not recorded')} · ${operators.length} operators · ${chains.length} chains · novelty established: no</p>${opCards?`<h2>Operators</h2>${opCards}`:'<p class="muted">No operators are cataloged yet.</p>'}${chainRows?`<h2>Chains</h2>${chainRows}`:''}${data.study?.report_path?`<p class="muted">Study report: <code>${esc(data.study.report_path)}</code> · ${esc(data.study.status||'status not recorded')}</p>`:''}`;
}
async function studies() {
  const data=await api('/api/studies'); S.studies=data.items;
  $('#study-list').innerHTML=(data.items||[]).map(study=>{
    const scalar=(k,v)=>typeof v==='number'?`${k}: ${v.toFixed(4)}`:typeof v==='string'?`${k}: ${v}`:'';
    const p=study.primary||{}; const primary=Object.entries(p).map(([k,v])=>scalar(k,v)).filter(Boolean).slice(0,4).map(x=>`<span>${esc(x)}</span>`).join(' · ');
    const s=study.secondary||{}; const secondary=Object.entries(s).filter(([k,v])=>typeof v==='number' || (v&&typeof v==='object'&&typeof v.candidate==='number'&&typeof v.control==='number')).map(([k,v])=>typeof v==='number'?scalar(k,v):`${k}: candidate ${v.candidate.toFixed(4)}, control ${v.control.toFixed(4)}`).slice(0,3).map(x=>`<span>${esc(x)}</span>`).join(' · ');
    return `<article class="run-card"><h2>${esc(study.proposal_title||study.id)} <small>${esc(study.id)} · ${esc(study.kind)}</small></h2><p><b>${esc(study.raw_verdict)}</b> · ${esc(study.audit_qualification)}</p><p class="muted">Primary: ${primary||'not recorded'}</p><p class="muted">Secondary: ${secondary||'not recorded'}</p><p>${esc(study.prior_art_status||'Prior-art status not recorded.')}</p><details><summary>Limitations and artifacts</summary><ul>${(study.limitations||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul><p>${(study.artifacts||[]).map(x=>`<code>${esc(x)}</code>`).join(' ')}</p><pre class="run-log">${esc(JSON.stringify(study.secondary||{},null,2))}</pre></details></article>`;
  }).join('') || '<p class="muted">No committed studies found.</p>';
}
async function generatedResearch() {
  const data=await api('/api/generated-research'); S.generated=data.items;
  $('#generated-list').innerHTML=(data.items||[]).map(run=>{
    const p=run.proposal||{},r=run.review||{},b=run.budget||{};
    const evidence=(p.evidence||[]).map(e=>`<li><code>${esc(e.chunk_id)}</code> — ${esc(e.supports)}<blockquote>${esc(e.quote)}</blockquote></li>`).join('');
    const op=run.operator_evaluation||{};
    const coverage=(label,value)=>value==null?'':`<span><b>${label}:</b> ${esc(typeof value==='object'?JSON.stringify(value):value)}</span>`;
    const coverageText=run.retrieved_source_count==null?'':`${esc(run.retrieved_source_count)} retrieved sources; ${esc((run.retrieval_failures||[]).length)} retrieval errors. ${(run.lineage_warnings||[]).map(esc).join('; ')}`;
    return `<article class="run-card"><h2>${esc(p.title||run.id)} <small>${esc(run.phase)}/${esc(run.id)}</small></h2><p><b>${esc(run.status)}</b> · machine review only · empirical result: no</p><p><b>Hypothesis:</b> ${esc(p.hypothesis||'not recorded')}</p><p><b>Operator delta:</b> ${esc(p.operator_delta||'not recorded')}</p><p><b>Reviewer:</b> ${esc(r.decision||'not recorded')} · ${esc(r.failure_type||'')}</p><p>${esc(r.reason||'')}</p>${coverageText?`<p class="muted">${coverageText}<br>Coverage is retrieval/lineage accounting only: partial recall, not a novelty determination.</p>`:''}<p><b>Executable operator check:</b> ${esc(op.status||'unavailable')} · ${op.verified?'result hash verified':'not a verified result'}<br><span class="muted">${esc(op.summary||'Numerical diagnostic only; not a predictive benchmark.')}</span></p><details><summary>Baseline, control, evidence, and budget</summary><p><b>Baseline:</b> ${esc(p.baseline||'')}</p><p><b>Matched control:</b> ${esc(p.matched_control||'')}</p><ul>${evidence}</ul><p>Budget actual ${esc(b.actual_usd)} USD; active reservation ${esc(b.active_reserved_usd)} USD; cap ${esc(b.cap_usd)} USD.</p><p><code>${esc(run.artifact)}</code></p>${run.retrieval_coverage?`<pre class="run-log">${esc(JSON.stringify({retrieval:run.retrieval_coverage,lineage:run.lineage_coverage},null,2))}</pre>`:""}</details></article>`;
  }).join('') || '<p class="muted">No completed grounded research runs found.</p>';
}
function populateRuns() {
  $('#run-select').innerHTML=S.runs.map(r=>`<option value="${esc(r.id)}">${esc(r.id)}</option>`).join('');
  if(!S.runs.some(r=>r.id===S.run))S.run=S.runs.find(r=>r.has_selection)?.id || S.runs[0]?.id || '';
  $('#run-select').value=S.run;
}
async function documents() {
  if(!S.project){S.docs=[];renderDocs();renderFolders([]);return;}
  const id=S.project.id;
  const [docs,folders]=await Promise.all([api(`/api/projects/${id}/documents`),api(`/api/projects/${id}/folders`)]);
  if(S.project?.id!==id)return;
  S.docs=docs.items;renderDocs();renderFolders(folders.items);
}
function renderDocs() {
  const query=$('#document-search').value.toLowerCase();
  const docs=S.docs.filter(d=>(d.name+' '+d.relative_path).toLowerCase().includes(query));
  $('#document-items').innerHTML=docs.map(d=>`<div class="doc-pick"><label><input type="checkbox" aria-label="Include ${esc(d.name)} in source pack" data-packdoc="${esc(d.id)}" ${S.packSelected.has(d.id)?'checked':''} ${d.kind==='text'&&/\.(md|txt|json|py)$/i.test(d.name)?'':'disabled'}> pack</label><button class="doc-item ${S.doc?.id===d.id?'selected':''}" data-doc="${esc(d.id)}"><b>${esc(d.name)}</b><small>${esc(d.relative_path)}<br>${Number(d.size).toLocaleString()} bytes</small></button></div>`).join('') || `<p class="muted">${S.project?'No documents found. Import files or write a note.':'Create a project to import documents.'}</p>`;
}
function renderFolders(folders) {
  const render=folders.map(f=>`<div class="folder-row"><span><b>${esc(f.name)}/</b><small>${esc(f.description)}</small><code class="folder-path">${esc(f.path)}</code></span><span class="folder-count">${f.document_count} files</span></div>`).join('');
  $('#folder-preview').innerHTML=render || '<p class="muted">Create a project to get sources/, notes/, reviews/, runs/, and exports/ folders.</p>';
  $('#project-folder-map').innerHTML=`<article class="setup-card"><h2>${esc(S.project?.name || 'Project folders')}</h2>${render || '<p>Create a project first. Its exact paths will appear here.</p>'}</article>`;
}
async function openDocument(id) {
  const doc=await api(`/api/projects/${S.project.id}/documents/${id}`);S.doc=doc;
  if(pdfURL){URL.revokeObjectURL(pdfURL);pdfURL=null;}
  let content;
  if(doc.binary){
    const bytes=Uint8Array.from(atob(doc.content_base64),c=>c.charCodeAt(0));pdfURL=URL.createObjectURL(new Blob([bytes],{type:'application/pdf'}));
    content=`<p>Original PDF. No automated extraction has been performed.</p><iframe title="${esc(doc.name)}" src="${pdfURL}" style="width:100%;height:65vh;border:0"></iframe>`;
  } else content=/\.md$/i.test(doc.name)?`<div class="markdown">${markdown(doc.content)}</div>`:`<pre>${esc(doc.content)}</pre>`;
  $('#document-reader').innerHTML=`<h2>${esc(doc.name)}</h2><div class="reader-meta">${esc(doc.relative_path)}<details><summary>File identity</summary><code>${esc(doc.sha256)}</code></details><a href="${esc(doc.download_url)}">Download original</a>${/^(notes|reviews)\//.test(doc.relative_path)?' <button class="button secondary" data-action="revise-note">Save a revised copy</button>':''}</div>${content}`;
  renderDocs();
}
async function importFiles(input) {
  if(!S.project)throw Error('Create a project before importing.');
  const allowed=/\.(pdf|md|txt|json|csv)$/i;
  const all=Array.from(input), files=all.filter(f=>allowed.test(f.name)&&!(f.webkitRelativePath||f.name).split('/').some(p=>p.startsWith('.')));
  if(!files.length)throw Error('No supported files. Choose PDF, Markdown, text, JSON, or CSV.');
  if(files.length>100)throw Error('Import up to 100 files at once. Split this folder into smaller batches.');
  if(files.some(f=>f.size>10*1024*1024)||files.reduce((a,f)=>a+f.size,0)>40*1024*1024)throw Error('Import limit: 10 MiB per file, 40 MiB per batch.');
  say(`Importing ${files.length} files…`);
  const payload=[];
  for(const file of files){const bytes=new Uint8Array(await file.arrayBuffer());let binary='';for(let i=0;i<bytes.length;i+=16384)binary+=String.fromCharCode(...bytes.subarray(i,i+16384));payload.push({name:file.webkitRelativePath||file.name,content_base64:btoa(binary)});}
  const result=await post(`/api/projects/${S.project.id}/documents`,{files:payload});
  await documents();go('documents');
  const unique=new Set(result.items.map(x=>x.id)).size;
  say(`Indexed ${unique} documents. ${all.length-files.length} unsupported or hidden files skipped. Identical content is kept once.`);
  if(result.items[0])await openDocument(result.items[0].id);
}
async function droppedFiles(event) {
  event.preventDefault();event.currentTarget.classList.remove('dragging');
  const output=[];
  async function walk(entry,prefix=''){
    if(entry.isFile){const file=await new Promise((resolve,reject)=>entry.file(resolve,reject));Object.defineProperty(file,'webkitRelativePath',{value:prefix+file.name});output.push(file);}
    else if(entry.isDirectory){const reader=entry.createReader();let chunk;do{chunk=await new Promise((resolve,reject)=>reader.readEntries(resolve,reject));for(const child of chunk)await walk(child,prefix+entry.name+'/');}while(chunk.length);}
  }
  const entries=Array.from(event.dataTransfer.items||[]).map(x=>x.webkitGetAsEntry?.()).filter(Boolean);
  if(entries.length)for(const entry of entries)await walk(entry);else output.push(...event.dataTransfer.files);
  await importFiles(output);
}
async function refreshRuns() {
  const [jobs,runs]=await Promise.all([api('/api/jobs'),api('/api/runs')]);S.jobs=jobs.jobs;S.runs=runs.runs;populateRuns();
  $('#run-list').innerHTML=`<article class="run-card"><h2>Rerank a saved pool</h2><p>Choose a run in Candidates first. Current pool: <b>${esc(S.run || 'none')}</b>. Pool scores remain scientifically unvalidated.</p><button class="button secondary" data-action="rerank">Configure reranking</button><button class="button secondary" data-action="audit">Audit provenance</button></article><h2>Jobs</h2>`+
    (S.jobs.slice().reverse().map(j=>`<article class="run-card"><h2>${esc(j.action)} <small>${esc(j.id)}</small></h2><p><b>${esc(j.status)}</b>${j.reason?' · '+esc(j.reason):''}</p><button class="button secondary" data-log="${j.id}">Inspect log & parameters</button>${['running','queued','cancel_requested'].includes(j.status)?` <button class="button secondary" data-cancel="${j.id}">Cancel job</button>`:''}${j.output?` <button class="button secondary" data-run="${esc(j.output)}">Open outputs</button>`:''}</article>`).join('')||'<p>No jobs yet. Configure an offline sample.</p>')+
    '<h2>Saved results</h2>'+S.runs.map(r=>`<article class="run-card"><h2>${esc(r.id)}</h2><p>${esc(r.path)}</p><button class="button secondary" data-run="${esc(r.id)}">Open candidates</button><button class="button secondary" data-artifacts="${esc(r.id)}">Read output files</button></article>`).join('');
}
async function job(action,options={}) {
  const result=await post('/api/jobs',{action,options:{...options,project_id:S.project?.id}});
  say(`${action}: ${result.status}`);go('runs');return result;
}
async function buildSourcePack(question, priorArtQueries=[], allowArxiv=false) {
  if(!S.project)throw Error('Create a project before building a source pack.');
  const ids=[...S.packSelected]; if(!ids.length)throw Error('Check at least one text document first.');
  const result=await post(`/api/projects/${S.project.id}/source-pack`,{document_ids:ids,question,prior_art_queries:priorArtQueries,allow_arxiv:allowArxiv}); S.lastSourcePack=result;
  modal('Offline research plan created',`<p>${esc(result.source_count)} selected documents were packed locally.</p><p><code>${esc(result.source_pack)}</code></p><p><code>${esc(result.plan)}</code></p><p class="library-boundary">The plan is source-grounded and bounded. The next action starts one reviewed run capped at $2 and governed by the shared $10 phase budget.</p><p><button class="button primary" data-grounded-run="${esc(result.run_id)}">Generate and review — up to $2/run</button></p>`);
}
async function candidates() {
  const request=++candidateRequest;
  if(!S.run){$('#candidate-items').textContent='No saved candidate pool. Start an offline run first.';return;}
  const data=await api(`/api/candidates?run=${encodeURIComponent(S.run)}&limit=50&offset=${S.offset}&q=${encodeURIComponent($('#candidate-search').value)}`);
  if(request!==candidateRequest)return;
  S.candidates=data.items;S.total=data.total;
  $('#candidate-items').innerHTML=`<p class="muted">${data.total?S.offset+1:0}–${S.offset+data.items.length} of ${data.total.toLocaleString()} candidates</p>`+
    data.items.map(c=>`<button class="candidate-item ${S.selected?.id===c.id?'selected':''}" data-candidate="${esc(c.id)}"><b>${esc(c.id)}</b><small>${(c.atoms||[]).map(a=>esc(a.description)).join(' · ')}</small></button>`).join('')+
    `<div class="pager"><button class="button secondary" data-page="-1" ${S.offset===0?'disabled':''}>Previous</button> <button class="button secondary" data-page="1" ${S.offset+50>=data.total?'disabled':''}>Next</button></div>`;
  if(!S.selected&&data.items[0])await openCandidate(data.items[0].id);
}
async function openCandidate(id) {
  const candidate=S.candidates.find(c=>c.id===id);if(!candidate)return;
  S.selected=candidate;const run=S.run;
  $('#candidate-reader').innerHTML='<p>Loading source evidence…</p>';
  const evidence=await api(`/api/evidence/${encodeURIComponent(id)}?run=${encodeURIComponent(run)}`);
  if(S.run!==run||S.selected?.id!==id)return;
  const scores=['coherence','availability','unavailability','fusion_score'].filter(k=>candidate[k]!=null).map(k=>`<span>${esc(k)}: <b>${Number(candidate[k]).toFixed(3)}</b></span>`).join(' &nbsp; ');
  $('#candidate-reader').innerHTML=`<h2>${esc(id)}</h2><p class="muted">${esc(run)} · ${esc(candidate.status)} · ${esc(candidate.validity)}</p><p class="muted">${scores || 'No scores saved.'}</p><p class="library-boundary">Canonical descriptions are generated interpretations. Original source units do not establish every claim in them.</p>`+
    candidate.atoms.map(a=>{const atom=evidence.atoms?.[a.id];return `<details class="atom"><summary>Atom ${esc(a.id)}: ${esc(a.description.slice(0,135))}…</summary><p>${esc(a.description)}</p><h3>${atom?.source_units?.length||0} original source units</h3>${(atom?.source_units||[]).map(u=>`<details><summary>${esc(u.paper?.title||u.paper_id)} · assignment ${esc(u.assignment_index??'unknown')}</summary><p>${esc(u.text)}</p>${u.paper?.abstract?`<h4>Paper abstract</h4><p>${esc(u.paper.abstract)}</p>`:''}${/^https?:\/\//.test(u.paper?.pdf_url||'')?`<a href="${esc(u.paper.pdf_url)}" target="_blank" rel="noopener noreferrer">Open paper</a>`:''}</details>`).join('')}</details>`;}).join('')+
    `<p><button class="button secondary" data-artifacts="${esc(run)}">Read proposals & output files</button> <button class="button secondary" data-action="prompt">Export reconstruction prompt</button></p><section class="atom"><h3>Human decision</h3><label>Decision<select id="decision"><option>hold</option><option>investigate</option><option>revise</option><option>reject</option></select></label><label>Evidence and next check<textarea id="decision-notes" maxlength="5000" rows="4"></textarea></label><button class="button primary" data-action="save-decision">Save decision</button></section>`;
}
async function artifacts(run) {
  const files=await api('/api/artifacts?run='+encodeURIComponent(run));
  modal('Output files: '+run,files.files.map(f=>`<p><button class="button secondary" data-file-run="${esc(run)}" data-file="${esc(f.name)}">${esc(f.name)}</button> <small>${f.size.toLocaleString()} bytes</small></p>`).join('')||'<p>No readable output files yet.</p>');
}
async function openArtifact(run,name) {
  const path=run.split('/').map(encodeURIComponent).join('/')+'/'+name.split('/').map(encodeURIComponent).join('/');
  const file=await api('/api/artifacts/'+path);
  const dialog=$('#detail-dialog');if(dialog?.open)dialog.close();
  modal(name,/\.md$/i.test(name)?`<article class="markdown">${markdown(file.content)}</article>`:`<pre class="run-log">${esc(file.content)}</pre>`);
}
async function reviews() {
  const result=await api('/api/adjudications');
  const items=result.items.filter(x=>!S.project || !x.project_id || x.project_id===S.project.id);
  $('#review-items').innerHTML=items.slice().reverse().map(r=>`<article class="review-card"><h2>${esc(r.candidate_id)} · ${esc(r.decision)}</h2><p>${esc(r.run)} · ${new Date(r.created_at*1000).toLocaleString()}</p><p>${esc(r.notes)}</p><button class="button secondary" data-run="${esc(r.run)}">Open source run</button></article>`).join('')||'<p>No candidate decisions saved. Written project reviews appear in Documents under reviews/.</p>';
}
async function setup() {
  await documents();const result=await api('/api/setup');
  $('#setup-content').innerHTML=`<article class="setup-card"><h2>Inference engine: ${result.engine_ready?'assets found':'setup required'}</h2><p>Document projects work independently of these engine assets.</p>${result.checks.map(c=>`<p>${c.ready?'✓':'○'} ${esc(c.name)}<code>${esc(c.path)}</code></p>`).join('')}<p>Interpreter: <code>${esc(result.python)}</code></p><button class="button secondary" data-action="engine-guide">Read installation guide</button></article><article class="setup-card"><h2>What belongs in each folder</h2>${result.folder_guide.map(f=>`<p><b>${esc(f.folder)}</b><br>${esc(f.purpose)}</p>`).join('')}</article><article class="setup-card"><h2>Engine and app paths</h2>${Object.entries(result.paths).map(([name,path])=>`<p>${esc(name)}<code>${esc(path)}</code></p>`).join('')}</article>`;
}
function resetReader(){S.doc=null;$('#document-reader').innerHTML='<div class="reader-empty"><h2>Select a document</h2><p>Choose a source or a note from this project.</p></div>';}
async function click(event) {
  try {
    const target=event.target.closest('button, a.wordmark');if(!target)return;
    if(target.dataset.view){go(target.dataset.view);return;}
    if(target.classList.contains('wordmark')){event.preventDefault();go('overview');return;}
    if(target.dataset.doc){await openDocument(target.dataset.doc);return;}
    if(target.dataset.packdoc){target.checked ? S.packSelected.add(target.dataset.packdoc) : S.packSelected.delete(target.dataset.packdoc);return;}
    if(target.dataset.groundedRun){await job('grounded_execute',{sourcepack_run:target.dataset.groundedRun});return;}
    if(target.dataset.candidate){await openCandidate(target.dataset.candidate);return;}
    if(target.dataset.run){S.run=target.dataset.run;S.selected=null;S.offset=0;$('#candidate-search').value='';$('#run-select').value=S.run;go('candidates');return;}
    if(target.dataset.page){S.offset=Math.max(0,S.offset+Number(target.dataset.page)*50);S.selected=null;await candidates();return;}
    if(target.dataset.artifacts){await artifacts(target.dataset.artifacts);return;}
    if(target.dataset.file){await openArtifact(target.dataset.fileRun,target.dataset.file);return;}
    if(target.dataset.log){const record=await api('/api/jobs/'+target.dataset.log);modal(record.action+' · '+record.status,`<pre class="run-log">${esc(record.log_tail||'No process log.')}\n\n${esc(JSON.stringify(record,null,2))}</pre>`);return;}
    if(target.dataset.cancel){await api('/api/jobs/'+target.dataset.cancel,{method:'DELETE'});await refreshRuns();return;}
    switch(target.dataset.action){
      case 'refresh':await load();go(S.view);break;
      case 'new-project':$('#project-form').reset();$('#project-dialog').showModal();break;
      case 'new-note':if(!S.project)throw Error('Create a project before writing notes.');$('#note-form').reset();$('#note-dialog').showModal();break;
      case 'source-pack':if(!S.project)throw Error('Create a project before building a source pack.');$('#source-pack-form').reset();ensureSourcePackSearchFields();$('#source-pack-dialog').showModal();break;
      case 'revise-note':$('#note-form').reset();$('#note-form').elements.title.value=S.doc.name.replace(/\.md$/i,'')+' revision '+Date.now();$('#note-form').elements.content.value=S.doc.content;$('#note-form').elements.category.value=S.doc.relative_path.split('/')[0];$('#note-dialog').showModal();break;
      case 'new-run':$('#run-dialog').showModal();break;
      case 'open-import':if(!S.project)throw Error('Create a project before importing.');$('#file-input').click();break;
      case 'export-project':if(!S.project)throw Error('Create a project to export.');const bundle=await post(`/api/projects/${S.project.id}/export`,{});modal('Project export ready',`<p>${esc(bundle.path)}</p><a class="button primary" href="${esc(bundle.download_url)}">Download ZIP</a>`);await documents();break;
      case 'open-audit':await openArtifact('provenance-repair-2026-11-16','REPAIR-REPORT.md');break;
      case 'engine-guide':const guide=await api('/api/guide/ENGINE_SETUP.md');modal('Engine installation',`<article class="markdown">${markdown(guide.content)}</article>`);break;
      case 'operator-check': {
        const ids=($('#operator-check-ids').value||'').split(',').map(x=>x.trim()).filter(Boolean);
        let values; try { values=JSON.parse($('#operator-check-values').value); } catch { throw Error('Values must be valid JSON.'); }
        const result=await post('/api/operator-check',{operator_ids:ids,values});
        const payload=result.result||{};
        $('#operator-check-result').innerHTML=`<p><b>Check complete.</b> Payload bytes: ${esc(payload.payload_bytes??payload.input_bytes??'not reported')} · Roundtrip: ${esc(payload.roundtrip??payload.lossless??'not reported')} · Scope: local reversible-chain check only.</p><p><code>Artifact ${esc(result.id)}</code></p>`;
        break;
      }
      case 'audit':await job('audit_provenance');break;
      case 'prompt':await job('prompt_export',{run:S.run,candidate_id:S.selected.id});break;
      case 'save-decision':await post('/api/adjudications',{candidate_id:S.selected.id,run:S.run,project_id:S.project?.id,decision:$('#decision').value,notes:$('#decision-notes').value});say('Decision saved');go('reviews');break;
      case 'rerank':{const dialog=modal('Rerank '+S.run,`<form id="rerank-form"><label>Unavailability weight (beta)<input name="beta" type="number" min="0" max="1" step="0.05" value="0.6" required></label><label>Selected candidates<input name="limit" type="number" min="1" max="100" value="20" required></label><p class="muted">Uses the saved scored pool. No model or API calls.</p><button class="button primary">Rerank pool</button></form>`);$('#rerank-form').onsubmit=async e=>{e.preventDefault();try{const values=Object.fromEntries(new FormData(e.target));await job('rerank_offline',{run:S.run,beta:Number(values.beta),limit:Number(values.limit)});dialog.close();}catch(error){fail(error);}};break;}
    }
  }catch(error){fail(error);}
}
document.addEventListener('click',click);
$('#project-select').onchange=async event=>{S.project=S.projects.find(p=>p.id===event.target.value);resetReader();renderProjects();try{await documents();if(S.view==='setup')await setup();if(S.view==='reviews')await reviews();}catch(error){fail(error);}};
$('#document-search').oninput=renderDocs;
let searchTimer;$('#candidate-search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{S.offset=0;S.selected=null;candidates().catch(fail);},250);};
$('#run-select').onchange=event=>{S.run=event.target.value;S.selected=null;S.offset=0;$('#candidate-search').value='';candidates().catch(fail);};
for(const id of ['file-input','folder-input']){
  const input=$('#'+id);input.onchange=()=>importFiles(input.files).catch(fail).finally(()=>input.value='');
  const zone=input.closest('label');zone.tabIndex=0;zone.setAttribute('role','button');zone.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();input.click();}});zone.addEventListener('dragover',e=>{e.preventDefault();zone.classList.add('dragging');});zone.addEventListener('dragleave',()=>zone.classList.remove('dragging'));zone.addEventListener('drop',e=>droppedFiles(e).catch(fail));
}
for(const kind of ['project','note','run'])$('#'+kind+'-form').onsubmit=async event=>{
  if(event.submitter?.value==='cancel')return;
  event.preventDefault();const values=Object.fromEntries(new FormData(event.target));
  try{
    if(kind==='project'){const project=await post('/api/projects',values);S.projects.unshift(project);S.project=project;resetReader();renderProjects();await documents();go('documents');say('Project created');}
    else if(kind==='note'){const doc=await post(`/api/projects/${S.project.id}/notes`,values);await documents();go('documents');await openDocument(doc.id);say('Note saved');}
    else{values.betas=values.betas.split(',').map(Number);values.n_gen=Number(values.n_gen);values.seed=Number(values.seed);values.top_k=Number(values.top_k);await job('sample_offline',values);}
    $('#'+kind+'-dialog').close();
  }catch(error){fail(error);}
};
function ensureSourcePackSearchFields(){
  const form=$('#source-pack-form');if(form.elements.prior_art_queries)return;
  const question=form.elements.question?.closest('label');if(!question)return;
  const label=document.createElement('label');label.innerHTML='Prior-art queries <span class="muted">(optional, one per line; up to 3)</span><textarea name="prior_art_queries" rows="3" maxlength="600" placeholder="e.g. continual learning memory agents"></textarea>';
  const check=document.createElement('label');check.className='checkbox-label';check.innerHTML='<input name="allow_arxiv" type="checkbox" value="true"> Search arXiv metadata using these queries';
  const note=document.createElement('p');note.className='muted';note.textContent='Generation searches the local index using these queries. The checkbox also sends only these query strings to arXiv. Blank queries skip retrieval; building the source pack remains offline.';
  question.after(label,check,note);
}
$('#source-pack-form').onsubmit=async event=>{if(event.submitter?.value==='cancel')return;event.preventDefault();try{const form=new FormData(event.target);const queries=String(form.get('prior_art_queries')||'').split(/\r?\n/).map(x=>x.trim()).filter(Boolean);await buildSourcePack(form.get('question'),queries,form.get('allow_arxiv')==='true');$('#source-pack-dialog').close();}catch(error){fail(error);}};
setInterval(()=>{if(S.view==='runs'&&!document.querySelector('dialog[open]'))refreshRuns().catch(fail);},3000);
load();

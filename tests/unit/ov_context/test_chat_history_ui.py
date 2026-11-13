"""Exercise the real page's history loader with asynchronous HTTP responses."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_history_replay_empty_sessions_and_stale_responses() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the page JavaScript contract test")
    page = Path(__file__).parents[3] / "examples/deepagents_ov_context/chat/static/index.html"
    source = page.read_text().split("<script>", 1)[1].split("</script>", 1)[0]
    # Run the actual functions; replace only boot-time I/O and the browser DOM.
    source = source.removesuffix("\n  ").replace("}init();", "}")
    harness = r"""
const vm = require('node:vm');
const assert = require('node:assert/strict');
const fs = require('node:fs');
class Element {
  constructor(){this.children=[];this.textContent='';this.value='';this.dataset={};this.style={};this.classList={add(){},remove(){}};}
  append(...items){this.children.push(...items);}
  replaceChildren(...items){this.children=items;this.textContent='';}
}
const nodes = new Map();
const byId=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
const content=node=>node.textContent+node.children.map(content).join('');
const stats=run_id=>({run_id,status:'done',model_calls:1,usage_calls:1,input_tokens:3,output_tokens:2,total_tokens:5,seconds:1,first_token_seconds:0.1});
const message=(text)=>({type:'human',id:'u',content:text,additional_kwargs:{ov_context_injected:true}});
const trace=(id,text)=>({run_id:id,created_at:1,prompt:text,statistics:stats(id),events:[
  {ordinal:1,event:'stage',data:{name:'memory.recall',status:'success',count:1},elapsed_ms:1},
  {ordinal:2,event:'model_input',data:{messages:[message(text)],message_reprs:[`HumanMessage(content=${text})`]},elapsed_ms:2}
]});
const traces={a:trace('a','[参考记忆]\nALICE\n[用户原文]\nquery'),b:trace('b','[参考记忆]\nBOB\n[用户原文]\nquery')};
const routes=new Map();
const ctx={console,performance:{now:()=>10},setTimeout:()=>1,clearTimeout(){},Option:class extends Element{constructor(text,value){super();this.textContent=text;this.value=value;}},document:{getElementById:byId,createElement:()=>new Element(),querySelector:()=>({dataset:{prompt:'PPT'}}),querySelectorAll:()=>[]},fetch:async path=>({ok:true,json:async()=>{if(routes.has(path))return await routes.get(path);if(path.endsWith('/runs')){const sid=path.split('/')[3];return traces[sid]?[traces[sid]]:[];}return traces[path.split('/').at(-1)];}})};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(0,'utf8')+'\nglobalThis.hooks={loadDebug,select(sid){selected={session_id:sid};traceLoad++;},getEvents:()=>debugEvents,getRun:()=>displayedRun};',ctx);
(async()=>{
  ctx.hooks.select('a');await ctx.hooks.loadDebug('a');
  assert.match(content(byId('model-input')),/ALICE/);
  assert.match(byId('recall-content').textContent,/ALICE/);
  assert.equal(ctx.hooks.getEvents().length,2);
  assert.equal(ctx.hooks.getRun(),'a');
  assert.equal(byId('messages').children.length,0,'Replaying debug events must not append chat messages');
  ctx.hooks.select('empty');await ctx.hooks.loadDebug('empty');
  assert.equal(ctx.hooks.getEvents().length,0);
  assert.doesNotMatch(content(byId('model-input')),/ALICE/);
  assert.equal(byId('recall-details').hidden,true);
  assert.match(byId('trace-note').textContent,/没有已保存/);
  let resolveA;
  routes.set('/api/sessions/a/runs',new Promise(resolve=>resolveA=resolve));
  ctx.hooks.select('a');const slow=ctx.hooks.loadDebug('a');
  ctx.hooks.select('b');await ctx.hooks.loadDebug('b');resolveA([traces.a]);await slow;
  assert.match(content(byId('model-input')),/BOB/);
  assert.doesNotMatch(content(byId('events')),/ALICE/);
  assert.equal(ctx.hooks.getRun(),'b');
  // Two Run selections within one Session may also return out of order.
  routes.delete('/api/sessions/a/runs');
  let resolveOld;
  routes.set('/api/sessions/a/runs/a',new Promise(resolve=>resolveOld=resolve));
  ctx.hooks.select('a');const old=ctx.hooks.loadDebug('a','a');await new Promise(setImmediate);
  routes.set('/api/sessions/a/runs/new',trace('new','NEW RUN'));
  await ctx.hooks.loadDebug('a','new');resolveOld(traces.a);await old;
  assert.match(content(byId('model-input')),/NEW RUN/);
  assert.equal(ctx.hooks.getRun(),'new');
  routes.set('/api/sessions/a/runs',Promise.reject(new Error('HTTP failure')));
  await ctx.hooks.loadDebug('a');
  assert.equal(ctx.hooks.getEvents().length,0);
  assert.match(byId('trace-note').textContent,/读取失败/);
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    subprocess.run([node, "-e", harness], input=source, text=True, check=True, timeout=10)


def test_tool_calls_keep_structured_arguments_in_history_and_stream() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the page JavaScript contract test")
    page = Path(__file__).parents[3] / "examples/deepagents_ov_context/chat/static/index.html"
    source = page.read_text().split("<script>", 1)[1].split("</script>", 1)[0]
    source = source.replace("}init();", "}")
    harness = r"""
const vm=require('node:vm'),assert=require('node:assert/strict'),fs=require('node:fs');
class Element {
  constructor(tag='div'){
    this.tagName=tag;this.children=[];this._text='';this.className='';this.dataset={};this.style={};this.attrs={};this.value='';
    this.classList={add:name=>this.className+=' '+name,remove:name=>this.className=this.className.split(' ').filter(x=>x!==name).join(' '),toggle(){}};
  }
  get textContent(){return this._text+this.children.map(x=>x.textContent).join('');}
  set textContent(value){this._text=String(value);this.children=[];}
  append(...items){for(const item of items){item.parent=this;this.children.push(item);}}
  replaceChildren(...items){this.children=[];this._text='';this.append(...items);}
  setAttribute(key,value){this.attrs[key]=value;}
  getAttribute(key){return this.attrs[key]??null;}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(x=>x!==this);}
  querySelectorAll(selector){return this.children.flatMap(x=>[...(selector.startsWith('.')?x.className.split(' ').includes(selector.slice(1)):x.tagName===selector)?[x]:[],...x.querySelectorAll(selector)]);}
  querySelector(selector){return this.querySelectorAll(selector)[0]??null;}
}
const nodes=new Map(),byId=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
const long='print("<script>not HTML</script>")\n'.repeat(40);
const reasoning='Reasoning text <script>must remain text</script>\n'.repeat(40);
const message={type:'ai',content:'执行检查',additional_kwargs:{reasoning_content:reasoning},tool_calls:[
  {name:'execute',id:'call-command',args:{timeout:0,command:'python skills/pptx/scripts/office/validate.py output/demo.pptx'}},
  {name:'browser',id:'call-browser',args:{argv:['open','https://example.org/?a=1&b=2'],enabled:false,filter:null}},
  {name:'write_file',id:'call-script',args:{file_path:'/generate.py',content:long}}
]};
const original=JSON.stringify(message),session={session_id:'history',user_id:'u',agent_id:'a',account_id:'acct',title:'history',execution:{mode:'baseline'}};
const ctx={console,performance:{now:()=>10},setTimeout:()=>1,clearTimeout(){},localStorage:{setItem(){}},Option:Element,
 document:{getElementById:byId,createElement:tag=>new Element(tag),querySelector:()=>({dataset:{prompt:'PPT'}}),querySelectorAll:()=>[]},
 fetch:async path=>({ok:true,json:async()=>path.endsWith('/memory')?{status:'idle'}:path.endsWith('/artifacts')||path.endsWith('/runs')?[]:{session,context:{source:'checkpoint',messages:[message],cached_message_count:1}}})};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(0,'utf8')+'\nglobalThis.hooks={selectSession,accept,renderChatMessage};',ctx);
function verify(){
 const turns=byId('messages').querySelectorAll('.turn');assert.equal(turns.length,1,'AI tool calls belong to one assistant turn');
 assert.match(turns[0].className,/assistant/);const cards=turns[0].querySelectorAll('.tool-call');assert.equal(cards.length,3);
 assert.equal(cards[0].querySelector('strong').textContent,'execute');
 assert.equal(cards[0].querySelector('.tool-call-id').textContent,'call-command');
 assert.equal(cards[0].querySelector('dt').textContent,'命令 · command');
 assert.equal(cards[0].querySelector('.call-arg-value').textContent,message.tool_calls[0].args.command);
 assert.match(cards[1].querySelector('.call-arg-value').textContent,/https:\/\/example.org/);
 const values=cards[1].querySelectorAll('.call-arg-value').map(x=>x.textContent);assert.ok(values.includes('false'));assert.ok(values.includes('null'));
 const script=cards[2].querySelectorAll('.call-arg-value')[1],toggle=cards[2].querySelector('button');
 assert.ok(script.textContent.length<long.length);toggle.onclick();assert.equal(script.textContent,long);toggle.onclick();assert.ok(script.textContent.length<long.length);
 assert.equal(cards[2].querySelectorAll('script').length,0,'Arguments must render as text');
 const folded=turns[0].querySelector('.reasoning');assert.ok(folded);
 assert.equal(folded.tagName,'details');assert.notEqual(folded.open,true,'Reasoning starts folded');
 assert.equal(folded.querySelector('summary').textContent,'Reasoning');
 assert.equal(folded.querySelector('pre').textContent,reasoning,'Reasoning remains complete');
 assert.equal(folded.querySelectorAll('script').length,0,'Reasoning must render as text');
 assert.equal(turns[0].querySelector('.body').textContent,'执行检查','Reasoning stays out of ordinary content');
 assert.equal(JSON.stringify(message),original,'Display must preserve original arguments');
}
(async()=>{
 await ctx.hooks.selectSession(session);verify();
 byId('messages').replaceChildren();ctx.hooks.accept('token',{text:'执行检查'});ctx.hooks.accept('message',{message});verify();
 ctx.hooks.accept('message',{message:{type:'tool',name:'execute',tool_call_id:'call-command',content:'ok'}});
 assert.equal(byId('messages').querySelectorAll('.turn').length,2);
 assert.equal(byId('messages').querySelectorAll('.tool').length,1,'Tool results remain separate from calls');
 byId('messages').replaceChildren();assert.equal(ctx.hooks.renderChatMessage({type:'ai',content:'',tool_calls:[]}),null);
 assert.equal(byId('messages').children.length,0,'Empty assistant messages add no blank cards');
 ctx.hooks.renderChatMessage({type:'ai',content:'',tool_calls:[{name:'ping',id:'empty',args:{}}]});
 assert.equal(byId('messages').querySelector('.body').hidden,true);assert.equal(byId('messages').querySelector('.call-no-args').textContent,'无参数');
 byId('messages').replaceChildren();
 const only=ctx.hooks.renderChatMessage({type:'ai',content:'',additional_kwargs:{reasoning_content:reasoning}});
 assert.ok(only,'A reasoning-only response must not disappear');assert.equal(only.querySelector('.body').hidden,true);
 assert.equal(only.querySelector('.reasoning').querySelector('pre').textContent,reasoning);
 ctx.hooks.renderChatMessage({type:'ai',content:'updated',additional_kwargs:{reasoning_content:'new reasoning'}},only);
 assert.equal(only.querySelectorAll('.reasoning').length,1,'Re-rendering must not duplicate reasoning');
 byId('messages').replaceChildren();
 const blocks=ctx.hooks.renderChatMessage({type:'ai',content:[{type:'text',text:'visible answer'},{type:'reasoning',reasoning:'block reason'},{type:'thinking',thinking:'block thinking'}]});
 assert.equal(blocks.querySelector('.body').textContent,'visible answer');
 assert.equal(blocks.querySelector('.reasoning').querySelector('pre').textContent,'block reason\n\nblock thinking');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    subprocess.run([node, "-e", harness], input=source, text=True, check=True, timeout=10)

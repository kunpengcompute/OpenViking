'use strict';
const editor=document.querySelector('#editor');
const calendarData=JSON.parse(document.querySelector('#calendar-data').textContent);
let selectedDay=calendarData.selected;
let saving=false;
const form=editor.querySelector('form');
const saveButton=form.querySelector('button.primary');
const editorStatus=document.querySelector('#editor-status');
const availabilityPanel=document.querySelector('#availability-panel');
const field=key=>editor.querySelector('.datetime-field[data-key="'+key+'"]');
const input=key=>field(key).querySelector('[data-datetime-input]');
const keys=['start_date','start_time','end_date','end_time'];
const pad=n=>String(n).padStart(2,'0');
const dayString=date=>date.getUTCFullYear()+'-'+pad(date.getUTCMonth()+1)+'-'+pad(date.getUTCDate());
function valid(value,kind){
 if(kind==='time')return /^(?:[01]\d|2[0-3]):[0-5]\d$/.test(value);
 if(!/^20\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$/.test(value))return false;
 const date=new Date(value+'T00:00:00Z');return !Number.isNaN(date.getTime())&&dayString(date)===value;
}
function validDateTimeRange(){
 if(!keys.every(key=>valid(input(key).value,field(key).dataset.kind)))return false;
 const start=Date.parse(input('start_date').value+'T'+input('start_time').value+':00Z');
 const end=Date.parse(input('end_date').value+'T'+input('end_time').value+':00Z');
 return end>start&&end-start<=86400000;
}
function refreshSave(){
 const complete=keys.every(key=>valid(input(key).value,field(key).dataset.kind));
 const rangeValid=complete&&validDateTimeRange();
 for(const key of keys){
  const target=input(key);
  if(target.value&&!valid(target.value,field(key).dataset.kind))target.setAttribute('aria-invalid','true');else target.removeAttribute('aria-invalid');
 }
 saveButton.disabled=saving||!rangeValid||Boolean(document.querySelector('#resource-booking')&&!form.querySelector('input[name=room_id]:checked'));
 if(!saving)editorStatus.textContent=!complete?'请填写有效日期 YYYY-MM-DD 和时间 HH:mm。':!rangeValid?'结束时间须晚于开始时间，日程时长上限为24小时。':'时区：Asia/Shanghai。';
}
editor.addEventListener('input',event=>{
 if(!event.target.matches('[data-datetime-input]'))return;
 availabilityPanel.hidden=true;
 refreshSave();
});
function openEditor(day=selectedDay){
 form.reset();availabilityPanel.hidden=true;
 input('start_date').value=day;input('end_date').value=day;
 input('start_time').value='';input('end_time').value='';
 refreshSave();editor.showModal();
}
document.querySelector('#new-event').onclick=()=>openEditor();
document.querySelector('#cancel-event').onclick=()=>editor.close();
for(const [id,title,description] of [
 ['export','导出日历','导出用于查看已有安排；请先选择日期范围和日历。本页不自动导出。'],
 ['settings','日历设置','当前默认日历为“我的日历”，时区为Asia/Shanghai。']]){
 document.querySelector('#'+id).onclick=()=>{
  document.querySelector('#info-title').textContent=title;
  document.querySelector('#info-body').textContent=description;
  document.querySelector('#info').showModal();
 };
}
function showOccupancy(data,conflict=false){
 availabilityPanel.replaceChildren();availabilityPanel.hidden=false;
 const heading=document.createElement('h3');heading.textContent=(conflict?'上次保存冲突：':'整日占用：')+data.date+' · '+data.calendar_label;availabilityPanel.append(heading);
 const hint=document.createElement('p');hint.textContent=conflict?data.message:'以下为目标日期与日历的已有安排，请按所需完整时长选择空闲时段。';availabilityPanel.append(hint);
 for(const row of data.events){const line=document.createElement('p');line.textContent=row.start.replace('T',' ').replace('+08:00','')+' — '+row.end.replace('T',' ').replace('+08:00','')+' · '+row.title;availabilityPanel.append(line);}
 if(!data.events.length){const line=document.createElement('p');line.textContent='当天该日历无占用。';availabilityPanel.append(line);}
}
document.querySelector('#destination').addEventListener('change',()=>availabilityPanel.hidden=true);
document.querySelector('#check-availability').onclick=async()=>{
 if(!valid(input('start_date').value,'date')){editorStatus.textContent='请输入有效开始日期 YYYY-MM-DD，再查询整日占用。';return;}
 const params=new URLSearchParams({date:input('start_date').value,calendar:document.querySelector('#destination').value});
 const button=document.querySelector('#check-availability');button.disabled=true;
 try{const response=await fetch(calendarData.base+'/availability?'+params,{headers:{Accept:'application/json'}});if(!response.ok)throw new Error('占用查询失败，请重试。');showOccupancy(await response.json());}
 catch(error){editorStatus.textContent=error.message;}finally{button.disabled=false;}
};
form.addEventListener('submit',async event=>{
 event.preventDefault();refreshSave();if(saveButton.disabled||!form.reportValidity())return;
 saving=true;saveButton.disabled=true;saveButton.textContent='正在保存…';editor.setAttribute('aria-busy','true');
 try{
  const response=await fetch(form.action,{method:'POST',body:new URLSearchParams(new FormData(form)),headers:{Accept:'application/json'}});
  if(response.status===409){const conflict=await response.json();if(conflict.resource){document.querySelector('#room-results').textContent=conflict.message;}else{showOccupancy(conflict,true);}saving=false;editor.removeAttribute('aria-busy');saveButton.textContent='保存日程';refreshSave();editorStatus.textContent=conflict.message;return;}
  if(!response.ok)throw new Error(response.status===422?'日程校验未通过，请检查标题及日期时间。':'保存失败（HTTP '+response.status+'），可重试。');
  const saved=await response.json();editor.close();window.location.assign(saved.return_url);
 }catch(error){saving=false;editor.removeAttribute('aria-busy');saveButton.textContent='保存日程';refreshSave();editorStatus.textContent=error.message;}
});

function showDay(day){
 selectedDay=day;
 const panel=document.querySelector('#day-pane'),body=document.querySelector('#day-events');
 panel.hidden=false;
 document.querySelector('#day-heading').textContent=new Intl.DateTimeFormat('zh-CN',{month:'long',day:'numeric',weekday:'long'}).format(new Date(day+'T12:00:00'));
 body.replaceChildren();
 const events=calendarData.events.filter(e=>e.start.slice(0,10)<=day&&e.end.slice(0,10)>=day&&!(e.end.slice(0,10)===day&&e.end.slice(11,16)==='00:00'));
 for(const event of events){
  const link=document.createElement('a');link.className='day-event';link.href=calendarData.base+'/events/'+event.id;
  const title=document.createElement('strong');title.textContent=event.title;
  const time=document.createElement('small');time.textContent=event.start.slice(11,16)+' – '+event.end.slice(11,16)+' · '+({personal:'我的日历',department:'产品部共享日历',rooms:'会议室使用安排'}[event.calendar_id]||event.calendar_id);
  link.append(title,time);body.append(link);
 }
 if(!events.length){const p=document.createElement('p');p.className='muted';p.textContent='这一天没有日程';body.append(p);}
 document.querySelectorAll('.month-cell').forEach(cell=>cell.classList.toggle('is-selected',cell.dataset.day===day));
 document.querySelectorAll('[data-select-day]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.selectDay===day)));
}
document.querySelectorAll('[data-select-day]').forEach(button=>button.onclick=()=>showDay(button.dataset.selectDay));
document.querySelectorAll('.month-cell').forEach(cell=>cell.ondblclick=event=>{if(!event.target.closest('a,button'))openEditor(cell.dataset.day);});
document.querySelector('#close-day').onclick=()=>document.querySelector('#day-pane').hidden=true;
document.querySelector('#show-day').onclick=()=>showDay(selectedDay);
document.querySelector('#day-new').onclick=()=>openEditor(selectedDay);
document.querySelectorAll('[data-autosubmit]').forEach(input=>input.onchange=()=>input.form.requestSubmit());
document.querySelector('#toggle-calendars').onclick=event=>{
 const open=document.querySelector('.outlook-shell').classList.toggle('sidebar-open');
 event.currentTarget.setAttribute('aria-expanded',String(open));
};
document.querySelector('#close-calendars').onclick=()=>{
 document.querySelector('.outlook-shell').classList.remove('sidebar-open');
 document.querySelector('#toggle-calendars').setAttribute('aria-expanded','false');
};

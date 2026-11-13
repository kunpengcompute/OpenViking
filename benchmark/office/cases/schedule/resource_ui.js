'use strict';
(()=>{
 let generation=0;
 const panel=document.querySelector('#resource-booking');if(!panel)return;
 const result=document.querySelector('#room-results'),status=document.querySelector('#room-status');
 const editor=document.querySelector('#editor'),form=editor.querySelector('form');
 const value=key=>form.querySelector('[name="'+key+'"]').value;
 const clear=()=>{generation++;result.replaceChildren();status.textContent='条件已更新，请查询会议室与日历占用。';refreshSave();};
 const rowText=(tag,text,parent)=>{const el=document.createElement(tag);el.textContent=text;parent.append(el);return el;};
 const interval=r=>r.start.slice(11,16)+'–'+r.end.slice(11,16)+' '+r.title;
 form.addEventListener('reset',clear);
 form.addEventListener('input',event=>{if(event.target.matches('[data-datetime-input],#attendees'))clear();});
 form.addEventListener('change',event=>{if(event.target.matches('#destination,#meeting-kind,#room-equipment,#room-scope,#room-minimum'))clear();if(event.target.name==='room_id')refreshSave();});
 document.querySelector('#query-rooms').onclick=async()=>{
  if(!validDateTimeRange()){status.textContent='请填写有效的起止日期和时间，结束须晚于开始且不超过24小时。';return;}
  const params=new URLSearchParams({start:value('start_date')+'T'+value('start_time')+'+08:00',end:value('end_date')+'T'+value('end_time')+'+08:00',calendar:value('calendar_id'),minimum:document.querySelector('#room-minimum').value,equipment:document.querySelector('#room-equipment').value,scope:document.querySelector('#room-scope').value});
  const request=++generation;result.replaceChildren();status.textContent='正在查询…';refreshSave();
  try{
   const response=await fetch(calendarData.base+'/room-options?'+params,{headers:{Accept:'application/json'}});if(!response.ok)throw new Error('请检查日期时间和筛选条件。');
   const data=await response.json();if(request!==generation)return;rowText('h3','查询日期 '+data.date+'；当前预约 '+data.start.slice(11,16)+'–'+data.end.slice(11,16),result);
   rowText('p','目标日历整日占用：'+(data.calendar_busy.map(interval).join('；')||'无'),result);
   for(const room of data.rooms){
    const item=document.createElement('section');item.className='room-option';result.append(item);
    const label=document.createElement('label'),radio=document.createElement('input');radio.type='radio';radio.name='room_id';radio.value=room.id;radio.required=true;radio.disabled=!room.available;label.append(radio,document.createTextNode(room.name));item.append(label);
    rowText('p','容量 '+room.capacity+'人；设备 '+([room.projector?'投屏':'',room.video?'视频':''].filter(Boolean).join('、')||'基础设施')+'；归属 '+({public:'公共',product:'产品部',finance:'财务部'}[room.department])+'；优先级 '+room.priority,item);
    rowText('p','状态：'+(!room.active?'停用':!['public','product'].includes(room.department)?'本账号无预订权限':room.available?'当前完整时段可用':'当前完整时段有占用'),item);
    rowText('p','整日占用：'+(room.busy.map(interval).join('；')||'无'),item);
   }
   status.textContent=data.rooms.length?'已显示筛选结果，请按会议类型、人数和优先级选择可用会议室。':'没有符合筛选条件的会议室。';
  }catch(error){if(request===generation)status.textContent=error.message;}
 };
})();

/* Only loaded by the isolated page layer for registered v2 projects. */
(() => {
  'use strict';
  function enhance() {
    for (const field of document.querySelectorAll('#issue_start_date,#issue_due_date')) {
      if (field.dataset.officeDate) continue;
      // Repair the CLI/native segmented-date incompatibility equally for both arms.
      field.type = 'text';
      field.placeholder = 'YYYY-MM-DD';
      field.pattern = '[0-9]{4}-[0-9]{2}-[0-9]{2}';
      field.dataset.officeDate = 'true';
      field.setAttribute('aria-description', '日期格式为YYYY-MM-DD，提交后由系统校验');
    }
    for (const link of document.querySelectorAll('a[href]')) {
      const url = new URL(link.href, location.href);
      if (url.origin === location.origin && /\/issues$/.test(url.pathname) && !url.search) {
        url.searchParams.set('per_page', '25'); link.href = url.href;
      }
    }
    const select = document.querySelector('#issue_assigned_to_id');
    if (!select || select.dataset.officeSearch) return;
    select.dataset.officeSearch = 'true';
    const wrapper = document.createElement('div'); wrapper.className = 'office-person-picker';
    const input = document.createElement('input');
    input.id = 'office-assignee-search'; input.type = 'search'; input.autocomplete = 'off';
    input.placeholder = '搜索负责人，选择候选人';
    input.setAttribute('aria-label', '搜索并选择负责人');
    input.setAttribute('role', 'combobox'); input.setAttribute('aria-autocomplete', 'list');
    input.setAttribute('aria-controls', 'office-assignee-options');
    const list = document.createElement('div'); list.id = 'office-assignee-options';
    list.setAttribute('role', 'listbox'); list.setAttribute('aria-label', '负责人候选列表'); list.hidden = true;
    const hint = document.createElement('small'); hint.textContent = '输入姓名后选择候选；项目变化时候选范围随表单更新。';
    input.required = select.required; select.required = false;
    select.hidden = true; select.setAttribute('aria-hidden', 'true');
    select.after(wrapper); wrapper.append(input, list, hint);
    const label = document.querySelector('label[for="issue_assigned_to_id"]');
    if (label) label.htmlFor = input.id;
    input.value = select.value ? select.selectedOptions[0].text.trim() : '';
    function validate() { input.setCustomValidity(input.value && !select.value ? '请从候选列表选择负责人' : ''); }
    function choose(option) {
      select.value = option.value; input.value = option.text.trim();
      select.dispatchEvent(new Event('change', {bubbles: true}));
      list.hidden = true; input.setAttribute('aria-expanded', 'false'); validate();
    }
    function render() {
      list.replaceChildren();
      const needle = input.value.trim().toLowerCase();
      const options = Array.from(select.options).filter(o => o.value && o.text.toLowerCase().includes(needle));
      for (const option of options) {
        const button = document.createElement('button'); button.type = 'button';
        button.setAttribute('role', 'option'); button.textContent = option.text.trim();
        button.setAttribute('aria-selected', String(option.value === select.value));
        button.onclick = () => choose(option); list.append(button);
      }
      if (!options.length) list.textContent = '没有匹配人员';
      list.hidden = false; input.setAttribute('aria-expanded', 'true');
    }
    input.onfocus = render;
    input.oninput = () => {select.value = ''; validate(); render();};
    input.onkeydown = event => {
      if (event.key === 'Escape') {list.hidden = true; input.setAttribute('aria-expanded', 'false');}
      if (event.key === 'ArrowDown') {event.preventDefault(); list.querySelector('button')?.focus();}
    };
    list.onkeydown = event => {
      const options = Array.from(list.querySelectorAll('button'));
      const index = options.indexOf(document.activeElement);
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault(); options[(index + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length]?.focus();
      }
    };
    document.addEventListener('click', event => {if (!wrapper.contains(event.target)) {list.hidden = true; input.setAttribute('aria-expanded', 'false');}});
    validate();
  }
  const style = document.createElement('style');
  style.textContent = '.office-person-picker{display:inline-block;min-width:300px}.office-person-picker small{display:block;color:#666}.office-person-picker [role=listbox]{max-height:210px;overflow:auto;border:1px solid #bbb;background:white}.office-person-picker button{display:block;width:100%;text-align:left;padding:7px}';
  document.head.append(style);
  let scheduled = false;
  const observer = new MutationObserver(() => {
    if (!scheduled) {scheduled = true; queueMicrotask(() => {scheduled = false; enhance();});}
  });
  enhance(); observer.observe(document.body, {childList: true, subtree: true});
})();

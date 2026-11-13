/* Synthetic office login and subscribed-app navigation. */
(() => {
  'use strict';
  const input = document.querySelector('#employee-input');
  if (input) {
    const picker = document.querySelector('#employee-options');
    const candidate = picker.querySelector('[role=option]');
    const selected = document.querySelector('#employee');
    const login = document.querySelector('#login-submit');
    const number = candidate.dataset.number;
    const label = candidate.textContent.trim();
    const choose = () => {
      input.value = label; selected.value = number; picker.hidden = true;
      input.setAttribute('aria-expanded', 'false'); login.disabled = false;
    };
    input.addEventListener('input', () => {
      selected.value = ''; login.disabled = true;
      const match = input.value.trim() && (label.includes(input.value.trim()) || number.includes(input.value.trim()));
      picker.hidden = !match; input.setAttribute('aria-expanded', String(!!match));
      document.querySelector('#employee-empty').hidden = !input.value.trim() || !!match;
    });
    candidate.addEventListener('click', choose);
    document.querySelector('#employee-choose').addEventListener('click', () => {
      picker.hidden = false;
      input.setAttribute('aria-expanded', 'true');
      document.querySelector('#employee-empty').hidden = true;
      candidate.focus();
    });
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !picker.hidden) { e.preventDefault(); choose(); }
      if (e.key === 'ArrowDown' && !picker.hidden) { e.preventDefault(); candidate.focus(); }
      if (e.key === 'Escape') { picker.hidden = true; input.setAttribute('aria-expanded', 'false'); }
    });
  }
  const search = document.querySelector('#app-search');
  if (search) search.addEventListener('input', () => {
    const needle = search.value.trim().toLowerCase(); let count = 0;
    for (const card of document.querySelectorAll('#subscribed article')) {
      card.hidden = !card.textContent.toLowerCase().includes(needle) && !card.dataset.keywords.includes(needle);
      if (!card.hidden) count++;
    }
    document.querySelector('#app-empty').hidden = !!count;
  });
  for (const button of document.querySelectorAll('[data-detail]')) {
    button.addEventListener('click', () => {
      document.querySelector('#app-detail-title').textContent = button.dataset.detail;
      document.querySelector('#app-detail-body').textContent = button.dataset.description;
      document.querySelector('#app-detail').showModal();
    });
  }
  const help = {
    manage: ['我的应用管理', '当前已订阅应用显示在工作台上方。通过工作台进入已订阅应用，通过应用市场查看功能介绍和订阅状态。测试工作区由管理员配置订阅。'],
    create: ['创建测试账号', '本工作区由管理员准备合成账号。请使用工号 100001，或搜索姓名林澈；账号新增由工作区管理员维护。'],
    help: ['登录帮助', '选择模拟网络环境，输入工号或姓名，从下拉候选中选择账号，然后点击登录。可按 Enter 确认候选，也可通过“选择”按钮展开列表。']
  };
  for (const button of document.querySelectorAll('[data-info]')) {
    button.addEventListener('click', () => {
      const [title, text] = help[button.dataset.info];
      document.querySelector('#info-title').textContent = title;
      document.querySelector('#info-body').textContent = text;
      document.querySelector('#info-dialog').showModal();
    });
  }
  for (const button of document.querySelectorAll('[data-close]')) {
    button.addEventListener('click', () => button.closest('dialog').close());
  }
  const tabs = document.querySelectorAll('[data-work-tab]');
  for (const tab of tabs) {
    tab.addEventListener('click', () => {
      for (const other of tabs) {
        const selected = other === tab;
        other.setAttribute('aria-selected', String(selected));
        document.querySelector('#' + other.getAttribute('aria-controls')).hidden = !selected;
      }
    });
  }
})();

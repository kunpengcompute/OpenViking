// Use Roundcube's server-side Date-header sort across the whole result set.
// Sorting changes the user's list preference, never the messages themselves.
(function () {
  'use strict';
  const mail = window.rcmail;
  if (!mail) return;

  mail.addEventListener('init', function () {
    const list = document.getElementById('messagelist-content');
    if (!list || document.getElementById('office-mail-sort')) return;

    const bar = document.createElement('div');
    bar.style.cssText = 'display:flex;align-items:center;gap:8px;padding:8px 12px;flex-shrink:0;border-bottom:1px solid #ddd;';
    const label = document.createElement('label');
    label.htmlFor = 'office-mail-sort';
    label.textContent = '发送时间排序';
    label.style.cssText = 'margin:0;white-space:nowrap;';
    const select = document.createElement('select');
    select.id = 'office-mail-sort';
    select.className = 'form-control';
    select.style.width = 'auto';
    [ ['', '当前列表顺序'], ['DESC', '最新在前'], ['ASC', '最早在前'] ].forEach(function (item) {
      const option = document.createElement('option');
      option.value = item[0];
      option.textContent = item[1];
      option.disabled = !item[0];
      select.appendChild(option);
    });
    function sync() {
      select.value = mail.env.sort_col === 'date' ? mail.env.sort_order.toUpperCase() : '';
    }
    select.addEventListener('change', function () {
      if (mail.busy || !['ASC', 'DESC'].includes(select.value)) {
        sync();
        return;
      }
      const order = select.value;
      mail.set_list_sorting('date', order);
      // Native list_mailbox preserves the active mailbox and search request.
      // Reset to page 1 so latest-first is useful even when invoked on page 2.
      mail.list_mailbox('', 1, 'date_' + order);
    });
    bar.appendChild(label);
    bar.appendChild(select);
    list.parentNode.insertBefore(bar, list);
    mail.addEventListener('listupdate', sync);
    sync();
  });
}());

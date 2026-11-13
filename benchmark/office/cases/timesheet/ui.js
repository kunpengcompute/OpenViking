/* Scoped work-card actions; mutations use the normal HTML form and CSRF. */
(() => {
  'use strict';
  const editor = document.querySelector('#add-entry');
  if (!editor) return;
  document.querySelector('#add-work-entry').addEventListener('click', () => editor.showModal());
  editor.querySelector('[data-cancel-entry]').addEventListener('click', () => editor.close());
  editor.querySelector('form').addEventListener('submit', event => {
    if (!event.target.checkValidity()) return;
    if (event.submitter) { event.submitter.disabled = true; event.submitter.textContent = '正在保存…'; }
  });
})();

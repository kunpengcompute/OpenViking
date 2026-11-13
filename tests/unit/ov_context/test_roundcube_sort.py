"""The visible sort control must delegate to native global sorting."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(not shutil.which("node"), reason="Node is needed for the mail UI test")
def test_sort_control_uses_native_sort_and_resets_pagination():
    source = (
        Path(__file__).resolve().parents[3]
        / "examples/deepagents_ov_context/docker/roundcube_sort.js"
    )
    subprocess.run(
        [
            "node",
            "-e",
            r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const nodes = {};
function element() {
  return {style: {}, children: [], events: {},
    appendChild(child) { this.children.push(child); if (child.id) nodes[child.id] = child; },
    addEventListener(name, fn) { this.events[name] = fn; }};
}
const list = element();
list.parentNode = {insertBefore(bar, before) { assert.equal(before, list); this.bar = bar; }};
nodes['messagelist-content'] = list;
const handlers = {};
const requests = [];
const mail = {env: {sort_col: '', sort_order: 'DESC', current_page: 2, search_request: 'active-search'},
  addEventListener(name, fn) { handlers[name] = fn; },
  set_list_sorting(col, order) { this.env.sort_col = col; this.env.sort_order = order; },
  list_mailbox(...args) { requests.push(args); }};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
  window: {rcmail: mail}, document: {createElement: element, getElementById: id => nodes[id]}
});
handlers.init();
const select = nodes['office-mail-sort'];
assert.equal(select.value, ''); // Do not silently change existing preferences.
assert.equal(select.children[1].textContent, '最新在前');
select.value = 'DESC'; select.events.change();
assert.deepEqual(requests.pop(), ['', 1, 'date_DESC']);
assert.equal(mail.env.search_request, 'active-search');
select.value = 'ASC'; select.events.change();
assert.deepEqual(requests.pop(), ['', 1, 'date_ASC']);
mail.busy = true; select.value = 'DESC'; select.events.change();
assert.equal(requests.length, 0); assert.equal(select.value, 'ASC');
mail.env.sort_col = 'from'; handlers.listupdate(); assert.equal(select.value, '');
const bar = list.parentNode.bar; handlers.init(); assert.equal(list.parentNode.bar, bar);
""",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

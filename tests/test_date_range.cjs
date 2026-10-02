const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('js/app.js', 'utf8');
const start = source.indexOf('async function loadPapersByDateRange(');
const end = source.indexOf('\n// 清除所有激活的关键词', start);
async function check(first, last, expected) {
  const fetched = [];
  const elements = {};
  const context = {
    availableDates: ['2026-09-01', '2026-09-02', '2026-09-03'],
    alert: text => { throw new Error(text); },
    document: {getElementById: id => elements[id] ||= {}},
    formatDate: value => value,
    selectLanguageForDate: () => 'Chinese',
    DATA_CONFIG: {getDataUrl: value => value},
    fetch: async url => { fetched.push(url.match(/data\/(.*?)_AI/)[1]); return {text: async () => ''}; },
    parseJsonlData: () => ({}), getAllCategories: () => [],
    renderCategoryFilter: () => {}, renderPapers: () => {},
    urlCategoryParam: null, urlJsonParam: null, urlAuthorParam: null, urlKeywordsParam: null,
    console: {error: error => { throw error; }}
  };
  vm.createContext(context);
  vm.runInContext(source.slice(start, end), context);
  await context.loadPapersByDateRange(first, last);
  assert.deepEqual(fetched, expected);
  assert.equal(elements.currentDate.textContent, `${expected[0]} - ${expected.at(-1)}`);
}
(async () => {
  await check('2026-09-01', '2026-09-03', ['2026-09-01', '2026-09-02', '2026-09-03']);
  await check('2026-09-03', '2026-09-01', ['2026-09-01', '2026-09-02', '2026-09-03']);
  await check('2026-09-02', '2026-09-02', ['2026-09-02']);
  console.log('Date range: 3 scenarios passed');
})().catch(error => { console.error(error); process.exitCode = 1; });

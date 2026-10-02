const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('js/app.js', 'utf8');
const elements = {};
function element() {
  return {children: [], dataset: {}, style: {}, classList: {add() {}},
    appendChild(child) {this.children.push(child);}, addEventListener() {}};
}
const context = {
  document: {getElementById: id => elements[id] ||= element(), createElement: element, body: element()},
  console, activeKeywords: [], activeAuthors: [], textSearchQuery: '',
  currentCategory: 'all', currentView: 'list', currentPaperIndex: 0,
  currentFilteredPapers: [], formatDate: value => value,
  formatAuthorsForCard: value => value
};
vm.createContext(context);
const parserStart = source.indexOf('const REPORT_GROUP_LABELS');
const parserEnd = source.indexOf('\nfunction renderCategoryFilter', parserStart);
vm.runInContext(source.slice(parserStart, parserEnd), context);
const renderStart = source.indexOf('function renderPapers()');
const renderEnd = source.indexOf('\nfunction closeModal', renderStart);
vm.runInContext(source.slice(renderStart, renderEnd), context);
const papers = [
  {id: 'main-brief', report_level: 'brief', interest_tier: 'primary', AI: {primary_direction_id: 'world_model', tldr: 'Primary brief'}},
  {id: 'aux-brief', report_level: 'brief', interest_tier: 'secondary', relevance_status: 'uncertain', AI: {primary_direction_id: 'continuous_language_multimodal', importance_score: 99, tldr: '简讯', method: 'must not appear'}},
  {id: 'aux-detail', report_level: 'detail', interest_tier: 'secondary', AI: {importance_score: 98, method: 'aux-method'}},
  {id: 'main-low', interest_tier: 'primary', AI: {importance_score: 80, method: 'main-method'}},
  {id: 'main-high', interest_tier: 'primary', AI: {importance_score: 90}},
  {id: 'historical', AI: {importance_score: 85, method: 'historical-method'}}
].map(p => ({title: p.id, categories: ['cs.CV'], authors: ['Author'], summary: 'Abstract', ...p}));
context.paperData = context.parseJsonlData(papers.map(p => JSON.stringify(p)).join('\n'), '2026-10-01');
assert.equal(context.paperData['cs.CV'].find(p => p.id === 'historical').reportLevel, 'detail');
context.renderPapers();
assert.deepEqual(Array.from(context.currentFilteredPapers, p => p.id), ['main-high', 'historical', 'main-low', 'aux-detail', 'aux-brief', 'main-brief']);
const children = elements.paperContainer.children;
assert.deepEqual(children.filter(p => p.className === 'report-group-title').map(p => p.textContent),
  ['主要方向详细摘要 · 3 篇', '辅助方向精选摘要 · 1 篇', '相关论文简讯 · 2 篇']);
assert.match(children.at(-1).innerHTML, /简讯，未做详细分析/);
context.showPaperDetails(context.currentFilteredPapers.at(-1), 6);
assert.match(elements.modalBody.innerHTML, /主要方向.*简讯，未做详细分析/);
context.showPaperDetails(context.currentFilteredPapers.find(p => p.id === 'aux-brief'), 5);
assert.match(elements.modalBody.innerHTML, /相关性待进一步确认/);
assert.doesNotMatch(elements.modalBody.innerHTML, /paper-sections|must not appear/);
context.showPaperDetails(context.currentFilteredPapers[1], 2);
assert.match(elements.modalBody.innerHTML, /historical-method/);
console.log('Report groups, quota briefs and historical detail rendering passed');

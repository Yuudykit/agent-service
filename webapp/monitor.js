'use strict';

const element = id => document.getElementById(id);
const services = {
  faq: '售后问答', order: '订单查询', refund: '退款试算', handoff: '转人工交接', other: '其他问题',
};
const modes = { mock: '离线演示', real: '真实模型' };
const statuses = { completed: '正常返回', degraded: '异常降级', limited: '护栏拦截', disconnected: '连接中断' };
const outcomes = {
  daily_token_budget_exceeded: '已达到每日 token 预算',
  max_turns_per_session_exceeded: '已达到单会话轮次上限',
  service_error: '服务异常，已返回备用回复',
  handoff_requested: '调用了本地交接工具',
  fallback_reply: '命中兜底话术标记',
};
const ratings = { helpful: '有帮助', unhelpful: '没帮助' };
const reasons = { irrelevant: '答非所问', incorrect: '信息不准', unresolved: '问题没解决', slow: '等待太久' };
const tools = {
  search_kb: '检索售后知识库', query_info: '查询模拟订单', calculate_refund: '计算退款金额',
  transfer_human: '生成本地交接记录', unknown_tool: '未知工具',
};
const number = value => value.toLocaleString('zh-CN');
const percentage = (part, total) => total ? Math.round(part * 1000 / total) / 10 + '%' : '—';
const duration = value => value == null ? '—' : value < 1000 ? value + ' ms' : (value / 1000).toFixed(2) + ' s';
const dialog = element('request-detail');

let selectedId = new URLSearchParams(location.search).get('request');
let fetchVersion = 0;
let lastSnapshot = '';

function make(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function pair(label, value) {
  const entry = make('div');
  entry.append(make('dt', label), make('dd', value));
  return entry;
}

function openDetail(request) {
  element('detail-id').textContent = request.request_id;
  const vote = request.feedback;
  const feedback = vote ? ratings[vote.rating] + (vote.reason ? ' · ' + reasons[vote.reason] : '') : '未评价';
  element('detail-values').replaceChildren(
    pair('时间', request.ts),
    pair('服务', services[request.service]),
    pair('运行模式', modes[request.mode]),
    pair('回复状态', statuses[request.status]),
    pair('服务端总耗时', duration(request.latency_ms)),
    pair('等待执行锁', duration(request.queue_ms)),
    pair('首段文本耗时', duration(request.first_delta_ms)),
    pair('已上报 token', number(request.tokens)),
    pair('处理说明', outcomes[request.outcome] || '—'),
    pair('用户评价', feedback),
  );
  element('detail-tool-list').replaceChildren(...request.tools.map(name => make('li', tools[name])));
  element('detail-no-tools').hidden = request.tools.length > 0;
  if (!dialog.open) dialog.showModal();
}

function render(snapshot) {
  const { total, returned } = snapshot;
  const positive = snapshot.ratings.helpful || 0;
  const negative = snapshot.ratings.unhelpful || 0;
  const rated = positive + negative;
  element('metric-total').textContent = number(total);
  element('sample-note').textContent = total ? '当前筛选范围内' : '暂无请求样本';
  element('metric-returned').textContent = percentage(returned, total);
  element('returned-note').textContent = total ? returned + ' / ' + total + ' 次发送结束帧' : '按服务端结束帧计算';
  element('metric-latency').textContent = duration(snapshot.latency_p95_ms);
  element('latency-note').textContent = returned ? returned + ' 个已返回样本' : '暂无耗时样本';
  element('metric-helpful').textContent = percentage(positive, rated);
  element('rating-note').textContent = rated
    ? positive + ' / ' + rated + ' 条有帮助 · 反馈率 ' + percentage(rated, returned)
    : '暂无评价 · 反馈率 ' + percentage(0, returned);

  const bars = Object.entries(services).map(([key, label]) => {
    const count = snapshot.services[key] || 0;
    const row = make('div', undefined, 'bar-row');
    const track = make('span', undefined, 'bar-track');
    const bar = make('i');
    bar.style.width = percentage(count, total) === '—' ? '0%' : percentage(count, total);
    track.append(bar);
    row.append(make('span', label), track, make('strong', number(count)));
    return row;
  });
  element('service-bars').replaceChildren(...bars);
  element('token-total').textContent = number(snapshot.tokens) + ' tokens';
  element('outcome-counts').replaceChildren(
    ...Object.entries(statuses).map(([key, label]) => pair(label, number(snapshot.statuses[key] || 0))),
    pair('转人工工具调用', number(snapshot.outcomes.handoff_requested || 0)),
    pair('命中兜底标记', number(snapshot.outcomes.fallback_reply || 0)),
    pair('有帮助', number(positive)),
    pair('没帮助', number(negative)),
  );
  const reasonItems = Object.entries(reasons).filter(([key]) => snapshot.reasons[key])
    .map(([key, label]) => make('li', label + ' · ' + snapshot.reasons[key]));
  element('feedback-reasons').hidden = !reasonItems.length;
  element('reason-list').replaceChildren(...reasonItems);

  const rows = snapshot.requests.map(request => {
    const row = make('tr');
    const id = make('button', '#' + request.request_id.slice(0, 8), 'request-id');
    id.type = 'button';
    id.title = '查看请求 ' + request.request_id;
    id.addEventListener('click', () => openDetail(request));
    const idCell = make('td');
    idCell.append(id);
    const state = make('td');
    state.append(make('span', statuses[request.status], 'state-tag ' + request.status));
    row.append(idCell, make('td', request.ts), make('td', modes[request.mode]),
      make('td', services[request.service]), state, make('td', duration(request.latency_ms)),
      make('td', request.feedback ? ratings[request.feedback.rating] : '—'));
    return row;
  });
  element('request-rows').replaceChildren(...rows);
  element('request-table').hidden = !rows.length;
  element('empty-requests').hidden = Boolean(rows.length);
  element('window-note').textContent = '仅汇总带请求编号的本机运行记录，最多 ' + number(snapshot.window_limit)
    + ' 条。列表显示最新 ' + rows.length + ' 条。';
}

async function refresh() {
  const version = ++fetchVersion;
  const query = new URLSearchParams({ mode: element('mode-filter').value, service: element('service-filter').value });
  if (selectedId) query.set('request', selectedId);
  const notice = element('monitor-notice');
  try {
    const response = await fetch('/api/monitor?' + query);
    if (!response.ok) throw new Error('Snapshot unavailable');
    const snapshot = await response.json();
    if (version !== fetchVersion) return;
    const serialized = JSON.stringify(snapshot);
    if (serialized !== lastSnapshot) {
      render(snapshot);
      lastSnapshot = serialized;
    }
    element('connection-text').textContent = '本机在线';
    element('connection').classList.add('is-real');
    element('update-time').textContent = '更新于 ' + new Date().toLocaleTimeString('zh-CN', { hour12: false });
    notice.hidden = true;
    if (selectedId) {
      if (snapshot.selected) openDetail(snapshot.selected);
      else {
        notice.textContent = '当前监控窗口内找不到这条请求。';
        notice.hidden = false;
      }
      selectedId = null;
    }
  } catch {
    if (version !== fetchVersion) return;
    element('connection-text').textContent = '数据暂不可用';
    element('connection').classList.remove('is-real');
    notice.textContent = '无法读取最新监控数据，请检查服务后刷新。';
    notice.hidden = false;
  }
}

element('mode-filter').addEventListener('change', refresh);
element('service-filter').addEventListener('change', refresh);
element('refresh').addEventListener('click', refresh);
refresh();
setInterval(() => { if (!document.hidden && !dialog.open) refresh(); }, 8000);

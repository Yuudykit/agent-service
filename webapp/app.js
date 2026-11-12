'use strict';

const byId = id => document.getElementById(id);
const welcome = byId('welcome');
const messages = byId('messages');
const scrollArea = byId('conversation-scroll');
const messageInput = byId('message-input');
const sendButton = byId('send-message');
const newButton = byId('new-session');
const mockMode = byId('mode-mock');
const realMode = byId('mode-real');
const serviceForm = byId('service-form');
const shortcuts = byId('service-shortcuts');
const orderInput = byId('order-id');
const handoffInput = byId('handoff-reason');
const serviceButtons = [...document.querySelectorAll('[data-service]')];
const promptButtons = [...document.querySelectorAll('[data-prompt]')];

const services = {
  faq: { title: '售后问答' },
  order: { title: '订单查询', submit: '查询订单', note: '演示使用模拟订单，可填入示例订单查看结果。' },
  refund: { title: '退款试算', submit: '计算退款', note: '按演示规则估算金额，提交后不会发起退款。' },
  handoff: { title: '转人工交接', submit: '生成交接记录', note: '将诉求和本次会话写入本地记录，不会接通人工坐席。' },
};
const toolNames = {
  search_kb: '检索售后知识库',
  query_info: '查询模拟订单',
  calculate_refund: '计算退款金额',
  transfer_human: '生成本地交接记录',
};
const degradedReasons = {
  daily_token_budget_exceeded: '今日用量已达到上限，本轮未调用模型。',
  max_turns_per_session_exceeded: '当前会话已达到轮次上限，可以开启新会话继续。',
};
const demoOrder = 'JD20260815001';

let sessionId = null;
let sentCount = 0;
let busy = false;
let realAvailable = false;
let activeService = null;
let activeLauncher = null;

function syncControls() {
  messageInput.style.height = 'auto';
  messageInput.style.height = Math.min(messageInput.scrollHeight, 140) + 'px';
  sendButton.disabled = busy || !messageInput.value.trim();
  newButton.disabled = busy;
  mockMode.disabled = busy;
  realMode.disabled = busy || !realAvailable;
  for (const button of serviceButtons) button.disabled = busy;
}

function updateMode() {
  const label = realMode.checked ? '真实模型' : '离线演示';
  const badge = byId('mode-badge');
  badge.textContent = label;
  badge.parentElement.classList.toggle('is-real', realMode.checked);
}

function sessionStatus(text, state) {
  byId('session-status').textContent = text;
  byId('session-dot').dataset.state = state;
}

function scrollToBottom() {
  scrollArea.scrollTop = scrollArea.scrollHeight;
}

function closeService(restoreFocus = false) {
  serviceForm.hidden = true;
  activeService = null;
  for (const button of serviceButtons) button.setAttribute('aria-expanded', 'false');
  if (restoreFocus) activeLauncher?.focus();
  activeLauncher = null;
}

function openService(name, launcher) {
  if (activeService === name) {
    closeService(true);
    return;
  }
  const service = services[name];
  activeService = name;
  activeLauncher = launcher;
  serviceForm.hidden = false;
  byId('service-title').textContent = service.title;
  byId('submit-service').textContent = service.submit || '';
  byId('service-note').textContent = service.note || '';
  byId('service-note').hidden = !service.note;
  byId('faq-fields').hidden = name !== 'faq';
  byId('service-fields').hidden = name === 'faq';
  byId('order-field').hidden = name !== 'order' && name !== 'refund';
  byId('refund-field').hidden = name !== 'refund';
  byId('handoff-field').hidden = name !== 'handoff';
  orderInput.setCustomValidity('');
  for (const button of serviceButtons) {
    button.setAttribute('aria-expanded', String(button.dataset.service === name));
  }
  const focusTarget = name === 'faq' ? promptButtons[0] : name === 'handoff' ? handoffInput : orderInput;
  focusTarget.focus();
}

function createMessage(role, text) {
  const message = document.createElement('article');
  message.className = 'message ' + role;
  if (role === 'assistant') {
    const mark = document.createElement('span');
    mark.className = 'message-mark';
    mark.innerHTML = '<svg class="icon" aria-hidden="true"><use href="#icon-brand"/></svg>';
    message.append(mark);
  }
  const body = document.createElement('div');
  body.className = 'message-body';
  const label = document.createElement('p');
  label.className = 'message-label';
  label.textContent = role === 'assistant' ? '售后助手' : '你';
  const content = document.createElement('div');
  content.className = 'message-text';
  content.textContent = text;
  const note = document.createElement('p');
  note.className = 'message-note';
  note.hidden = true;
  body.append(label, content, note);
  message.append(body);
  messages.append(message);
  return { body, content, note };
}

function addFeedback(answer, requestId, replySessionId) {
  const controls = byId('feedback-template').content.firstElementChild.cloneNode(true);
  const buttons = [...controls.querySelectorAll('button')];
  const reasons = controls.querySelector('select');
  const status = controls.querySelector('[role="status"]');
  controls.querySelector('a').href = '/monitor?request=' + encodeURIComponent(requestId);
  let rating = '';
  let reason = '';

  async function save(nextRating, nextReason) {
    for (const button of buttons) button.disabled = true;
    reasons.disabled = true;
    try {
      const response = await fetch('/api/feedback', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ request_id: requestId, session_id: replySessionId, rating: nextRating, reason: nextReason }),
      });
      if (!response.ok) throw new Error('Feedback was rejected');
      rating = nextRating;
      reason = nextReason;
      status.textContent = '已记录';
    } catch {
      status.textContent = '保存失败，请重试';
    } finally {
      for (const button of buttons) {
        button.disabled = false;
        button.setAttribute('aria-pressed', String(button.dataset.rating === rating));
      }
      reasons.value = reason;
      reasons.hidden = rating !== 'unhelpful';
      reasons.disabled = false;
    }
  }

  for (const button of buttons) button.addEventListener('click', () => {
    const next = button.dataset.rating;
    if (next !== rating) save(next, '');
  });
  reasons.addEventListener('change', () => save('unhelpful', reasons.value));
  answer.body.append(controls);
}

function renderAnswer(container, answer) {
  const result = /^\[(查询结果|计算结果)\]\s*/.exec(answer);
  container.textContent = answer;
  if (!result) return;

  const heading = document.createElement('span');
  heading.className = 'result-heading';
  heading.textContent = result[1] === '查询结果' ? '订单详情' : '退款试算结果';
  const text = answer.slice(result[0].length);
  const amount = /应退\s*¥\d+(?:\.\d+)?/.exec(text);
  if (result[1] === '查询结果' || !amount) {
    container.replaceChildren(heading, text);
    return;
  }

  const highlight = document.createElement('strong');
  highlight.className = 'result-amount';
  highlight.textContent = amount[0];
  container.replaceChildren(heading, text.slice(0, amount.index), highlight, text.slice(amount.index + amount[0].length));
}

function resetActivity(text) {
  byId('tool-list').replaceChildren();
  byId('tool-list').hidden = true;
  byId('activity-empty').hidden = false;
  byId('activity-empty').querySelector('p').textContent = text;
}

function showTool(name) {
  const label = toolNames[name] || name;
  const item = document.createElement('li');
  item.innerHTML = '<svg class="icon" aria-hidden="true"><use href="#icon-activity"/></svg>';
  const text = document.createElement('span');
  text.textContent = label;
  item.append(text);
  const list = byId('tool-list');
  list.append(item);
  list.hidden = false;
  byId('activity-empty').hidden = true;
  byId('composer-status').textContent = label + '…';
}

async function readEvents(response, onEvent) {
  if (!response.ok || !response.body || !response.headers.get('Content-Type')?.includes('text/event-stream')) {
    throw new Error('Chat endpoint did not return an event stream');
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let boundary;
      while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, boundary.index);
        buffer = buffer.slice(boundary.index + boundary[0].length);
        const data = frame.split(/\r?\n/).filter(line => line.startsWith('data:'))
          .map(line => line.slice(5).replace(/^ /, '')).join('\n');
        if (data) onEvent(JSON.parse(data));
      }
      if (done) return;
    }
  } finally {
    reader.releaseLock();
  }
}

async function sendMessage(text, service = null) {
  const question = text.trim();
  if (!question || busy) return;
  const mock = mockMode.checked;
  busy = true;
  sentCount += 1;
  syncControls();
  closeService();
  welcome.hidden = true;
  messages.hidden = false;
  shortcuts.hidden = false;
  byId('session-count').textContent = sentCount + ' 条';
  byId('latency-row').hidden = true;
  byId('tokens-row').hidden = true;
  sessionStatus('处理中', 'busy');
  resetActivity('等待工具调用');
  byId('composer-status').textContent = '正在处理…';

  createMessage('user', question);
  const answer = createMessage('assistant', '正在处理…');
  answer.content.classList.add('is-pending');
  scrollToBottom();

  let streamed = '';
  let completed = false;
  let toolCalled = false;
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: question, session_id: sessionId, mock, service }),
    });
    await readEvents(response, event => {
      if (event.type === 'session') {
        sessionId = event.session_id;
      } else if (event.type === 'tool') {
        toolCalled = true;
        showTool(event.name);
      } else if (event.type === 'delta') {
        streamed += event.text;
        answer.content.textContent = streamed;
        answer.content.classList.remove('is-pending');
        scrollToBottom();
      } else if (event.type === 'done') {
        completed = true;
        sessionId = event.session_id || sessionId;
        renderAnswer(answer.content, event.answer || streamed || '本轮没有返回内容。');
        answer.content.classList.remove('is-pending');
        if (event.feedback_available && event.request_id) addFeedback(answer, event.request_id, event.session_id);
        sessionStatus(event.degraded ? '已返回备用回复' : '本轮完成', event.degraded ? 'error' : 'ready');
        if (event.degraded) {
          answer.note.textContent = degradedReasons[event.detail] || '服务暂时不可用，已返回备用回复。';
          answer.note.hidden = false;
        }
        if (!toolCalled) resetActivity('本轮未调用工具');
        if (typeof event.latency_ms === 'number') {
          byId('latency').textContent = event.latency_ms < 1000
            ? event.latency_ms + ' ms' : (event.latency_ms / 1000).toFixed(1) + ' s';
          byId('latency-row').hidden = false;
        }
        if (typeof event.tokens === 'number') {
          byId('tokens').textContent = event.tokens.toLocaleString('zh-CN') + ' tokens';
          byId('tokens-row').hidden = false;
        }
        scrollToBottom();
      }
    });
    if (!completed) throw new Error('Chat stream ended before completion');
  } catch {
    answer.content.classList.remove('is-pending');
    if (streamed) {
      answer.content.textContent = streamed;
      answer.note.textContent = '连接中断，上方是已收到的部分回复。';
      answer.note.hidden = false;
    } else {
      answer.content.textContent = '未收到回复。请确认服务仍在运行，再重新发送。';
      answer.content.classList.add('is-error');
      if (!messageInput.value) messageInput.value = question;
    }
    sessionStatus('连接中断', 'error');
    if (!toolCalled) resetActivity('未收到工具调用信息');
    scrollToBottom();
  } finally {
    busy = false;
    byId('composer-status').textContent = '';
    syncControls();
  }
}

for (const button of serviceButtons) {
  button.addEventListener('click', () => openService(button.dataset.service, button));
}
for (const button of promptButtons) {
  button.addEventListener('click', () => sendMessage(button.dataset.prompt, 'faq'));
}
byId('close-service').addEventListener('click', () => closeService(true));
orderInput.addEventListener('input', () => orderInput.setCustomValidity(''));
byId('sample-order').addEventListener('click', () => {
  orderInput.value = demoOrder;
  orderInput.setCustomValidity('');
  orderInput.focus();
});

serviceForm.addEventListener('submit', event => {
  event.preventDefault();
  if (busy || !activeService || activeService === 'faq') return;
  if (activeService === 'handoff') {
    const reason = handoffInput.value.trim();
    sendMessage(reason ? '我需要转人工客服。诉求：' + reason : '我需要转人工客服。', 'handoff');
    return;
  }

  const order = orderInput.value.trim().toUpperCase();
  if (!/^(?:JD\d{8,}|\d{10,})$/.test(order)) {
    orderInput.setCustomValidity('请输入示例格式的 JD 订单号或至少 10 位数字订单号。');
    orderInput.reportValidity();
    return;
  }
  if (activeService === 'order') {
    sendMessage('订单 ' + order + ' 到哪了？', 'order');
  } else {
    const reason = byId('refund-reason').value === 'quality' ? '商品有质量问题' : '个人原因';
    sendMessage('请计算订单 ' + order + ' 可以退多少钱。退款原因：' + reason + '。', 'refund');
  }
});

messageInput.addEventListener('input', syncControls);
messageInput.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    if (!sendButton.disabled) byId('message-form').requestSubmit();
  }
});
byId('message-form').addEventListener('submit', event => {
  event.preventDefault();
  if (sendButton.disabled) return;
  const text = messageInput.value;
  messageInput.value = '';
  sendMessage(text);
  messageInput.focus();
});

newButton.addEventListener('click', () => {
  sessionId = null;
  sentCount = 0;
  closeService();
  serviceForm.reset();
  orderInput.setCustomValidity('');
  messageInput.value = '';
  messages.replaceChildren();
  messages.hidden = true;
  shortcuts.hidden = true;
  welcome.hidden = false;
  byId('session-count').textContent = '0 条';
  byId('latency-row').hidden = true;
  byId('tokens-row').hidden = true;
  sessionStatus('待开始', '');
  resetActivity('发送消息后查看工具调用');
  syncControls();
  messageInput.focus();
});

const panelToggle = byId('panel-toggle');
const sessionPanel = byId('session-panel');
panelToggle.addEventListener('click', () => {
  const opened = sessionPanel.classList.toggle('is-open');
  panelToggle.setAttribute('aria-expanded', String(opened));
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && sessionPanel.classList.contains('is-open')) {
    sessionPanel.classList.remove('is-open');
    panelToggle.setAttribute('aria-expanded', 'false');
    panelToggle.focus();
  } else if (event.key === 'Escape' && activeService) {
    closeService(true);
  }
});

mockMode.addEventListener('change', updateMode);
realMode.addEventListener('change', updateMode);
fetch('/api/health')
  .then(response => {
    if (!response.ok) throw new Error('Health check failed');
    return response.json();
  })
  .then(health => {
    realAvailable = Boolean(health.has_api_key);
    byId('mode-note').textContent = realAvailable
      ? '已配置 ' + (health.model || '模型') + '。切换到真实模型会产生 API 用量。'
      : '离线演示可直接使用。配置 API Key 后可切换真实模型。';
  })
  .catch(() => {
    byId('mode-note').textContent = '未能读取模型配置，可以尝试离线演示。';
  })
  .finally(syncControls);

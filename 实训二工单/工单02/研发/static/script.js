/**
 * 日程提醒智能体 - 前端交互逻辑
 * 支持：添加日程 | 查询日程 | 取消日程 | 到时提醒
 */

// DOM 元素
const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const welcomeMessage = document.getElementById('welcomeMessage');
const reminderToast = document.getElementById('reminderToast');
const reminderText = document.getElementById('reminderText');

// 状态
let isProcessing = false;
const sessionId = 'session-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8);

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    messageInput.focus();
    setupEventListeners();
    startReminderPolling();
});

function setupEventListeners() {
    sendButton.addEventListener('click', handleSend);
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
    messageInput.addEventListener('input', autoResize);
}

function autoResize() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
}

// 发送示例消息
function sendExample(text) {
    messageInput.value = text;
    handleSend();
}

// 发送消息
async function handleSend() {
    const message = messageInput.value.trim();
    if (!message || isProcessing) return;

    messageInput.value = '';
    messageInput.style.height = 'auto';
    welcomeMessage.style.display = 'none';

    // 显示用户消息
    addMessage(message, 'user');

    // 显示输入中
    const typingId = showTyping();

    // 发送请求
    isProcessing = true;
    setStatus('loading', '处理中...');

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: sessionId }),
        });

        const data = await response.json();

        removeTyping(typingId);

        if (response.ok) {
            addMessage(data.response, 'assistant');
        } else {
            addMessage(data.response || '⚠️ 请求失败，请稍后再试。', 'assistant');
        }
    } catch (err) {
        removeTyping(typingId);
        addMessage('⚠️ 网络连接失败，请检查服务器是否运行。', 'assistant');
    }

    isProcessing = false;
    setStatus('online', '就绪');
    messageInput.focus();
}

// 添加消息到聊天区
function addMessage(text, role) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message ' + role;

    const now = new Date();
    const timeStr = now.getHours().toString().padStart(2, '0') + ':' +
                    now.getMinutes().toString().padStart(2, '0');

    let avatar = '📅';
    if (role === 'user') avatar = '👤';
    if (role === 'reminder') avatar = '⏰';

    msgDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-bubble">${escapeHtml(text)}</div>
            <div class="message-time">${timeStr}</div>
        </div>
    `;

    chatContainer.appendChild(msgDiv);
    scrollToBottom();
}

// 显示输入中动画
function showTyping() {
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.className = 'typing-indicator';
    div.id = id;
    div.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    chatContainer.appendChild(div);
    scrollToBottom();
    return id;
}

function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// 轮询检查提醒
function startReminderPolling() {
    setInterval(async () => {
        try {
            const response = await fetch('/api/check_reminders');
            const data = await response.json();

            if (data.reminders && data.reminders.length > 0) {
                data.reminders.forEach(r => {
                    addMessage(r.message, 'reminder');
                });
                showReminderToast(data.reminders[0].message);
            }
        } catch (err) {
            // 静默失败
        }
    }, 30000);
}

// 显示提醒弹窗
function showReminderToast(message) {
    reminderText.textContent = message;
    reminderToast.style.display = 'block';

    // 5秒后自动隐藏
    setTimeout(() => {
        reminderToast.style.display = 'none';
    }, 5000);

    // 点击关闭
    reminderToast.onclick = () => {
        reminderToast.style.display = 'none';
    };
}

// 设置连接状态
function setStatus(state, text) {
    statusDot.className = 'status-dot';
    if (state === 'online') statusDot.classList.add('');
    else if (state === 'loading') statusDot.classList.add('loading');
    else if (state === 'error') statusDot.classList.add('error');
    statusText.textContent = text;
}

// 滚动到底部
function scrollToBottom() {
    setTimeout(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 50);
}

// HTML 转义
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

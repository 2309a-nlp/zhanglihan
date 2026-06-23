/**
 * 智能记账本 - 前端交互逻辑
 * 支持：支出 | 收入 | 查询 | 删除
 */

// ============================================================
// DOM 元素引用
// ============================================================
const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const welcomeMessage = document.getElementById('welcomeMessage');
const featureTabs = document.querySelectorAll('.feature-tab');

// ============================================================
// 状态管理
// ============================================================
let isProcessing = false;
let currentFilter = 'all';
// 生成随机会话 ID，用于维护对话历史
const sessionId = 'session-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8);

// ============================================================
// 初始化
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    messageInput.focus();
    setupEventListeners();
});

function setupEventListeners() {
    // 发送按钮
    sendButton.addEventListener('click', handleSend);

    // 键盘事件
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // 自动调整高度
    messageInput.addEventListener('input', autoResize);

    // 功能标签切换
    featureTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            featureTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentFilter = tab.dataset.filter;
            filterChips(currentFilter);
        });
    });
}

// ============================================================
// 功能标签：过滤快捷按钮
// ============================================================
function filterChips(filter) {
    const chipTitles = welcomeMessage.querySelectorAll('.chip-title');
    const chips = welcomeMessage.querySelectorAll('.chip');

    if (filter === 'all') {
        chipTitles.forEach(t => t.style.display = '');
        chips.forEach(c => c.style.display = '');
        return;
    }

    // 隐藏所有标题和按钮
    chipTitles.forEach(t => t.style.display = 'none');
    chips.forEach(c => c.style.display = 'none');

    // 根据过滤条件显示
    switch (filter) {
        case 'expense':
            showChipsByClass('expense-chip');
            break;
        case 'income':
            showChipsByClass('income-chip');
            break;
        case 'query':
            showChipsByClass('query-chip');
            break;
        case 'delete':
            showChipsByClass('delete-chip');
            break;
    }
}

function showChipsByClass(className) {
    const chips = welcomeMessage.querySelectorAll(`.${className}`);
    chips.forEach(chip => {
        chip.style.display = '';
        // 向前查找对应的 chip-title 并显示
        let prev = chip.previousElementSibling;
        while (prev) {
            if (prev.classList.contains('chip-title')) {
                prev.style.display = '';
                break;
            }
            prev = prev.previousElementSibling;
        }
    });
}

// ============================================================
// 自动调整输入框高度
// ============================================================
function autoResize() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

// ============================================================
// 发送消息
// ============================================================
function sendExample(text) {
    messageInput.value = text;
    handleSend();
}

async function handleSend() {
    const message = messageInput.value.trim();
    if (!message || isProcessing) return;

    messageInput.value = '';
    messageInput.style.height = 'auto';

    // 隐藏欢迎消息
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }

    appendMessage('user', message);
    await sendToServer(message);
}

async function sendToServer(message) {
    setProcessing(true);
    const typingId = showTyping();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message, session_id: sessionId }),
        });

        const data = await response.json();
        removeTyping(typingId);

        if (data.error) {
            appendMessage('assistant', data.response || 'Error: ' + data.error);
            setStatus('error', '异常');
        } else {
            appendMessage('assistant', data.response);
            setStatus('online', '就绪');
        }
    } catch (error) {
        removeTyping(typingId);
        appendMessage('assistant', '⚠️ 网络连接失败，请检查服务是否已启动。');
        setStatus('error', '连接失败');
        console.error('请求错误:', error);
    }

    setProcessing(false);
}

// ============================================================
// 消息渲染
// ============================================================
function appendMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const now = new Date();
    const timeStr = now.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });

    if (role === 'user') {
        messageDiv.innerHTML = `
            <div class="message-avatar">👤</div>
            <div>
                <div class="message-bubble">${escapeHtml(content)}</div>
                <div class="message-time">${timeStr}</div>
            </div>
        `;
    } else {
        messageDiv.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div>
                <div class="message-bubble">${formatMessage(content)}</div>
                <div class="message-time">${timeStr}</div>
            </div>
        `;
    }

    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function formatMessage(text) {
    let formatted = escapeHtml(text);

    // Markdown 处理
    formatted = formatted.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/\n/g, '<br>');

    return formatted;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================
// 打字指示器
// ============================================================
function showTyping() {
    const id = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = id;
    typingDiv.innerHTML = `
        <div class="message-avatar" style="background: linear-gradient(135deg, #f0fdf4, #dcfce7);">🤖</div>
        <div class="typing-dots">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    `;
    chatContainer.appendChild(typingDiv);
    scrollToBottom();
    return id;
}

function removeTyping(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

// ============================================================
// 状态指示器
// ============================================================
function setStatus(status, text) {
    statusText.textContent = text;
    statusDot.className = 'status-dot';
    if (status === 'error') statusDot.classList.add('error');
    if (status === 'loading') statusDot.classList.add('loading');
}

function setProcessing(processing) {
    isProcessing = processing;
    sendButton.disabled = processing;
    if (processing) {
        setStatus('loading', '处理中...');
        messageInput.placeholder = '正在处理...';
        messageInput.disabled = true;
    } else {
        setStatus('online', '就绪');
        messageInput.placeholder = '输入记录或查询... 支出/收入/查询/删除';
        messageInput.disabled = false;
        messageInput.focus();
    }
}

// ============================================================
// 滚动
// ============================================================
function scrollToBottom() {
    setTimeout(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 100);
}

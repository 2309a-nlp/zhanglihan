import { useEffect, useMemo, useRef, useState } from 'react'
import { Avatar, Button, Input, Select, Space, Spin, Typography, message as antMessage } from 'antd'
import axios from 'axios'

const roles = ['Medical', 'Psychology', 'Education', 'Law', 'Finance']
const { Text, Title } = Typography
const STORAGE_KEY_PREFIX = 'expert_rag_history_'

const createSession = (defaultRole = roles[0]) => ({
  id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  title: '新对话',
  role: defaultRole,
  messages: [],
  createdAt: Date.now(),
  updatedAt: Date.now(),
})

function normalizeSession(raw) {
  if (!raw || typeof raw !== 'object') return createSession()
  return {
    id: typeof raw.id === 'string' ? raw.id : `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    title: typeof raw.title === 'string' ? raw.title : '新对话',
    role: raw.role && roles.includes(raw.role) ? raw.role : roles[0],
    messages: Array.isArray(raw.messages) ? raw.messages : [],
    createdAt: typeof raw.createdAt === 'number' ? raw.createdAt : Date.now(),
    updatedAt: typeof raw.updatedAt === 'number' ? raw.updatedAt : Date.now(),
  }
}

function loadSessions(storageKey) {
  try {
    const saved = localStorage.getItem(storageKey)
    if (saved) {
      const parsed = JSON.parse(saved)
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed.map(normalizeSession)
      }
    }
  } catch {
    /* ignore */
  }
  const s = createSession()
  return [s]
}

/** LLM 返回可能是字符串、或多段 [{ type, text }]，统一成可读文本 */
function extractAnswerText(raw) {
  if (raw == null) return ''
  if (typeof raw === 'string') return raw.trim()
  if (Array.isArray(raw)) {
    return raw
      .map((block) => {
        if (typeof block === 'string') return block
        if (block && typeof block === 'object') {
          const t = block.text ?? block.content
          return typeof t === 'string' ? t : extractAnswerText(t)
        }
        return String(block)
      })
      .filter(Boolean)
      .join('\n')
      .trim()
  }
  if (typeof raw === 'object') {
    const t = raw.text ?? raw.content ?? raw.answer ?? raw.data
    if (typeof t === 'string') return t.trim()
    if (t != null) return extractAnswerText(t)
    try {
      return JSON.stringify(raw)
    } catch {
      return String(raw)
    }
  }
  return String(raw).trim()
}

/** 兼容多种后端 JSON 形态，并识别业务错误 code */
function parseChatResponse(res) {
  const d = res?.data
  if (typeof d === 'string') {
    const t = extractAnswerText(d)
    return t ? { ok: true, text: t } : { ok: false, text: '服务器返回空内容' }
  }
  if (d == null || typeof d !== 'object') {
    return { ok: false, text: '无效响应' }
  }
  if (d.code != null && Number(d.code) !== 200) {
    return { ok: false, text: String(d.msg ?? d.message ?? `请求失败 (${d.code})`) }
  }
  const raw =
    d.answer ??
    d.data ??
    d.content ??
    (typeof d.result === 'string' ? d.result : undefined)
  const text = extractAnswerText(raw)
  if (text !== '') {
    const bestSim =
      d.best_similarity != null && Number.isFinite(Number(d.best_similarity))
        ? Number(d.best_similarity)
        : null
    const simThr =
      d.similarity_threshold != null && Number.isFinite(Number(d.similarity_threshold))
        ? Number(d.similarity_threshold)
        : null
    return {
      ok: true,
      text,
      kbHit: Boolean(d.kb_hit),
      kbAvailable: Boolean(d.kb_available),
      answerMode: typeof d.answer_mode === 'string' ? d.answer_mode : '',
      sourceLabel: typeof d.source_label === 'string' ? d.source_label : '',
      citations: Array.isArray(d.citations) ? d.citations : [],
      bestSimilarity: bestSim,
      similarityThreshold: simThr,
    }
  }
  return { ok: false, text: String(d.msg ?? '回复为空') }
}

export default function Chat({ user }) {
  const storageKey = `${STORAGE_KEY_PREFIX}${user}`
  const initialSessions = useMemo(() => loadSessions(storageKey), [storageKey])
  const bottomRef = useRef(null)

  const [sessions, setSessions] = useState(initialSessions)
  const [currentId, setCurrentId] = useState(initialSessions[0].id)
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    localStorage.setItem(storageKey, JSON.stringify(sessions))
  }, [sessions, storageKey])

  const currentSession = useMemo(
    () => sessions.find((s) => s.id === currentId) ?? sessions[0],
    [sessions, currentId]
  )

  const currentRole =
    currentSession?.role && roles.includes(currentSession.role) ? currentSession.role : roles[0]
  const messages = currentSession?.messages ?? []

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, loading])

  const patchSession = (sessionId, patch) => {
    setSessions((prev) =>
      prev.map((s) => (s.id === sessionId ? { ...s, ...patch, updatedAt: Date.now() } : s))
    )
  }

  const setSessionRole = (role) => {
    patchSession(currentId, { role })
  }

  const send = async () => {
    const question = text.trim()
    if (!question || loading) return

    const sessionId = currentId
    const history = messages.slice(-8).map((item) => ({
      question: item.q,
      answer: item.a,
    }))

    setLoading(true)
    setText('')

    const titleFromQ =
      question.length > 28 ? `${question.slice(0, 28)}…` : question
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id !== sessionId) return s
        return {
          ...s,
          messages: [...s.messages, { q: question, a: '' }],
          title: s.title === '新对话' ? titleFromQ : s.title,
          updatedAt: Date.now(),
        }
      })
    )

    const fillLastAssistant = (payload) => {
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== sessionId) return s
          const next = [...s.messages]
          const last = next[next.length - 1]
          if (last && last.q === question && last.a === '') {
            next[next.length - 1] = {
              ...last,
              a: payload.a,
              sourceLabel: payload.sourceLabel,
              citations: payload.citations,
              kbHit: payload.kbHit,
              bestSimilarity: payload.bestSimilarity,
              similarityThreshold: payload.similarityThreshold,
            }
          }
          return { ...s, messages: next, updatedAt: Date.now() }
        })
      )
    }

    try {
      const res = await axios.post('/api/chat', {
        username: user,
        role: currentRole,
        question,
        history,
      })

      const parsed = parseChatResponse(res)
      if (parsed.ok) {
        fillLastAssistant({
          a: parsed.text,
          sourceLabel: parsed.sourceLabel,
          citations: parsed.citations,
          kbHit: parsed.kbHit,
          bestSimilarity: parsed.bestSimilarity,
          similarityThreshold: parsed.similarityThreshold,
        })
      } else {
        fillLastAssistant({
          a: `错误：${parsed.text}`,
          sourceLabel: '请求未成功，无知识库状态',
          citations: [],
          kbHit: false,
          bestSimilarity: null,
          similarityThreshold: null,
        })
        antMessage.warning(parsed.text)
      }
    } catch (err) {
      console.error(err)
      antMessage.error('请求失败，请确认后端已启动（默认 http://127.0.0.1:8000）')
      fillLastAssistant({
        a: '（网络错误）请检查网络或后端服务。',
        sourceLabel: '网络错误',
        citations: [],
        kbHit: false,
        bestSimilarity: null,
        similarityThreshold: null,
      })
    } finally {
      setLoading(false)
    }
  }

  const addSession = () => {
    const newS = createSession(currentRole)
    setSessions((prev) => [newS, ...prev])
    setCurrentId(newS.id)
  }

  const delSession = (id, e) => {
    e.stopPropagation()
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id)
      if (next.length === 0) {
        const s = createSession(currentRole)
        setCurrentId(s.id)
        return [s]
      }
      if (id === currentId) setCurrentId(next[0].id)
      return next
    })
  }

  const rename = (id, e) => {
    e.stopPropagation()
    const name = window.prompt('请输入新名称')
    if (name?.trim()) {
      patchSession(id, { title: name.trim() })
    }
  }

  const clearCurrent = () => {
    patchSession(currentId, { messages: [], title: '新对话' })
  }

  return (
    <div className="chat-page">
      <div className="chat-layout">
        <aside className="history-panel">
          <div className="history-header">
            <Text strong>会话</Text>
            <Button type="primary" size="small" onClick={addSession}>
              新对话
            </Button>
          </div>
          <div className="history-list">
            {sessions.map((s) => (
              <div
                key={s.id}
                className={`history-item ${s.id === currentId ? 'active' : ''}`}
              >
                <button
                  type="button"
                  className="history-main"
                  onClick={() => setCurrentId(s.id)}
                >
                  <span>{s.title}</span>
                  <small>{new Date(s.updatedAt).toLocaleString()}</small>
                </button>
                <div className="history-actions">
                  <Space size={4}>
                    <Button type="link" size="small" onClick={(e) => rename(s.id, e)}>
                      重命名
                    </Button>
                    <Button type="link" danger size="small" onClick={(e) => delSession(s.id, e)}>
                      删除
                    </Button>
                  </Space>
                </div>
              </div>
            ))}
          </div>
        </aside>

        <section className="chat-shell">
          <header className="chat-header">
            <Text type="secondary">
              结合领域知识与检索回答问题；内容由模型生成，请注意甄别。
            </Text>
            <Space wrap className="role-select">
              <Select
                value={currentRole}
                onChange={setSessionRole}
                options={roles.map((r) => ({ value: r, label: `${r}专家` }))}
                style={{ minWidth: 140 }}
              />
              <Button size="small" onClick={clearCurrent}>
                清空当前对话
              </Button>
            </Space>
          </header>

          <div className="chat-body">
            {messages.length === 0 && (
              <div className="empty-chat">
                <div style={{ textAlign: 'center' }}>
                  <Title level={4} style={{ marginBottom: 8 }}>
                    欢迎使用专家 RAG 系统
                  </Title>
                  <Text type="secondary">新建或选择左侧会话，选择领域后开始提问</Text>
                </div>
              </div>
            )}

            {messages.map((item, idx) => {
              const isLast = idx === messages.length - 1
              const pending = item.a === '' && isLast && loading

              return (
                <div key={`${currentId}-${idx}-${item.q?.slice(0, 12)}`} className="message-pair">
                  <div className="message message-user">
                    <Avatar style={{ background: '#4f46e5', flexShrink: 0 }}>我</Avatar>
                    <div className="bubble bubble-user">{item.q}</div>
                  </div>
                  <div className="message">
                    <Avatar style={{ background: '#52c41a', flexShrink: 0 }}>AI</Avatar>
                    <div className="bubble bubble-ai">
                      {item.a === '' ? (
                        pending ? (
                          <Space>
                            <Spin size="small" />
                            <Text type="secondary">思考中…</Text>
                          </Space>
                        ) : (
                          <Text type="secondary">等待回复…</Text>
                        )
                      ) : (
                        <>
                          <div className="bubble-ai-body">{item.a}</div>
                          {item.sourceLabel ? (
                            <div className="answer-source-footer">
                              <Text type="secondary" style={{ fontSize: 12 }}>
                                {item.sourceLabel}
                                {item.citations?.length
                                  ? ` · 引用：${item.citations.join('、')}`
                                  : ''}
                                {item.bestSimilarity != null && item.similarityThreshold != null
                                  ? ` · Top1相关度 ${item.bestSimilarity.toFixed(2)} / 阈值 ${item.similarityThreshold.toFixed(2)}`
                                  : ''}
                              </Text>
                            </div>
                          ) : null}
                        </>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
            <div ref={bottomRef} />
          </div>

          <footer className="chat-footer">
            <Input.TextArea
              value={text}
              onChange={(e) => setText(e.target.value)}
              onPressEnter={(e) => {
                if (e.shiftKey) return
                e.preventDefault()
                send()
              }}
              placeholder="输入问题，Shift+Enter 换行，Enter 发送"
              disabled={loading}
              rows={3}
              autoSize={{ minRows: 2, maxRows: 8 }}
              name="chat-input"
              autoComplete="off"
            />
            <Button type="primary" onClick={send} loading={loading} disabled={!text.trim()}>
              发送
            </Button>
          </footer>
        </section>
      </div>
    </div>
  )
}

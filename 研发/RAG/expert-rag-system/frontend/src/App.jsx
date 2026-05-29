import { useState } from 'react'
import { Layout, Card, Segmented, Typography } from 'antd'
import Login from './Login'
import Register from './Register'
import Chat from './Chat'
import './App.css'

const { Content } = Layout
const { Title, Text } = Typography

export default function App() {
  const [page, setPage] = useState('login')
  const [user, setUser] = useState(null)

  if (user) return <Chat key={user} user={user} />

  return (
    <Layout className="auth-layout">
      <Content className="auth-content">
        <Card className="auth-card">
          <Title level={2} className="auth-title">Expert RAG System</Title>
          <Text type="secondary" className="auth-subtitle">支持多角色、可回答任意问题的智能助手</Text>
          <Segmented
            className="auth-switch"
            options={[
              { label: '登录', value: 'login' },
              { label: '注册', value: 'reg' },
            ]}
            value={page}
            onChange={setPage}
            block
          />
          {page === 'login' && <Login goRegister={() => setPage('reg')} onLogin={setUser} />}
          {page === 'reg' && <Register goLogin={() => setPage('login')} />}
        </Card>
      </Content>
    </Layout>
  )
}
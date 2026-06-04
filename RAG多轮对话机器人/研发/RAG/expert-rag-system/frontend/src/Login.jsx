import { useState } from 'react'
import { Button, Form, Input, message } from 'antd'
import axios from 'axios'

export default function Login({ goRegister, onLogin }) {
  const [loading, setL] = useState(false)

  const login = async (v) => {
    setL(true)
    try {
      await axios.post('/api/login', v)
      message.success('登录成功')
      onLogin(v.username)
    } catch (e) {
      message.error('登录失败，请检查账号或密码')
    } finally {
      setL(false)
    }
  }

  return (
    <div className="auth-form-wrap">
      <Form layout="vertical" onFinish={login}>
        <Form.Item label="用户名" name="username" rules={[{ required: true, message: '请输入用户名' }]}>
          <Input placeholder="请输入用户名" size="large" />
        </Form.Item>
        <Form.Item label="密码" name="password" rules={[{ required: true, message: '请输入密码' }]}>
          <Input.Password placeholder="请输入密码" size="large" />
        </Form.Item>
        <Button type="primary" htmlType="submit" loading={loading} size="large" block>
          登录
        </Button>
      </Form>
      <Button type="link" onClick={goRegister}>还没有账号？去注册</Button>
    </div>
  )
}
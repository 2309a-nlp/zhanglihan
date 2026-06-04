import { useState } from 'react'
import { Button, Form, Input, message } from 'antd'
import axios from 'axios'

export default function Register({ goLogin }) {
  const [l, setL] = useState(false)

  const register = async (v) => {
    setL(true)
    try {
      await axios.post('/api/register', v)
      message.success('注册成功，请登录')
      goLogin()
    } catch (e) {
      message.error('注册失败，请稍后重试')
    } finally {
      setL(false)
    }
  }

  return (
    <div className="auth-form-wrap">
      <Form layout="vertical" onFinish={register}>
        <Form.Item label="用户名" name="username" rules={[{ required: true, message: '请输入用户名' }]}>
          <Input placeholder="设置用户名" size="large" />
        </Form.Item>
        <Form.Item label="密码" name="password" rules={[{ required: true, message: '请输入密码' }]}>
          <Input.Password placeholder="设置密码" size="large" />
        </Form.Item>
        <Button type="primary" htmlType="submit" loading={l} size="large" block>
          注册
        </Button>
      </Form>
      <Button type="link" onClick={goLogin}>已有账号？去登录</Button>
    </div>
  )
}
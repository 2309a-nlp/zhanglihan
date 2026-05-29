# 简单的内存数据库（使用 Python 字典来模拟，Key是用户名，Value是密码）
# ⚠️ 注意：数据存储在内存中，服务重启后所有用户数据会丢失
users_db = {}


def get_user(username):
    """
    根据用户名获取对应的密码。
    主要用于登录时校验用户是否存在以及密码是否匹配。
    """
    # .get() 方法：如果用户名存在则返回密码，不存在则返回 None
    return users_db.get(username)


def create_user(username, password):
    """
    注册新用户。
    返回: True 表示注册成功，False 表示用户名已被占用。
    """
    # 先检查用户名是否已经存在于数据库中
    if username in users_db:
        return False  # 用户已存在，注册失败

    # 将新的用户名和密码存入字典（⚠️ 注意：这里密码是明文存储的）
    users_db[username] = password
    return True  # 注册成功
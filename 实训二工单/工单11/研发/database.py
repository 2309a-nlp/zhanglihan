"""
数据库模块 - 医疗挂号管理 Agent
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-挂号管理任务

功能：严格按照工单 PDF 中的 7 张表结构定义 DDL，并提供 CRUD 操作。
表结构参考：工单中去水印后的三张表格图片（表2-3-9 至 表2-3-15）
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "hospital.db")


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """获取数据库连接"""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ============================================================
# DDL 脚本 - 严格按照工单 PDF 表格定义
# ============================================================

DDL_SCRIPT = """
-- 表2-3-9 department (科室信息表)
CREATE TABLE IF NOT EXISTS department (
    dep_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    dep_Name VARCHAR(50),
    dep_Address VARCHAR(200)
);

-- 表2-3-10 doctor (医生信息表)
CREATE TABLE IF NOT EXISTS doctor (
    d_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    d_Name VARCHAR(50),
    d_Sex CHAR(1) DEFAULT '男',
    d_Profession VARCHAR(50),
    d_LoginName VARCHAR(50),
    d_LoginPSW VARCHAR(50),
    dep_ID INTEGER,
    FOREIGN KEY (dep_ID) REFERENCES department(dep_ID)
);

-- 表2-3-11 patientstatus (就诊状态表)
CREATE TABLE IF NOT EXISTS patientstatus (
    ps_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    ps_Name VARCHAR(20),
    ps_Remark VARCHAR(100)
);

-- 表2-3-12 patient (病人信息表)
CREATE TABLE IF NOT EXISTS patient (
    p_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    p_Name VARCHAR(50),
    p_Sex CHAR(1) DEFAULT '男',
    p_Address VARCHAR(50),
    p_Birth DATETIME,
    ps_ID INTEGER,
    FOREIGN KEY (ps_ID) REFERENCES patientstatus(ps_ID)
);

-- 表2-3-13 diagnosis (诊疗信息表)
CREATE TABLE IF NOT EXISTS diagnosis (
    dia_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    d_ID INTEGER,
    p_ID INTEGER,
    dia_Time DATETIME,
    dia_Symptom VARCHAR(1000),
    dia_Diagnosis VARCHAR(1000),
    dia_Dispense VARCHAR(1000),
    dia_Remark VARCHAR(1000),
    FOREIGN KEY (d_ID) REFERENCES doctor(d_ID),
    FOREIGN KEY (p_ID) REFERENCES patient(p_ID)
);

-- 表2-3-14 worker (挂号员信息表)
CREATE TABLE IF NOT EXISTS worker (
    w_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    w_Name VARCHAR(20),
    w_LoginName VARCHAR(50),
    w_LoginPSW VARCHAR(50)
);

-- 表2-3-15 register (挂号信息表)
CREATE TABLE IF NOT EXISTS register (
    reg_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    dep_ID INTEGER,
    p_ID INTEGER,
    w_ID INTEGER,
    reg_Time DATETIME,
    reg_Fee INTEGER,
    reg_Order INTEGER,
    reg_Status INTEGER,
    FOREIGN KEY (dep_ID) REFERENCES department(dep_ID),
    FOREIGN KEY (p_ID) REFERENCES patient(p_ID),
    FOREIGN KEY (w_ID) REFERENCES worker(w_ID)
);

-- 表2-3-16 schedule (医生坐诊时间表)
CREATE TABLE IF NOT EXISTS schedule (
    s_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    d_ID INTEGER,
    s_Date DATE,
    s_TimeSlot VARCHAR(20),
    s_MaxPatients INTEGER DEFAULT 20,
    s_Available INTEGER DEFAULT 1,
    FOREIGN KEY (d_ID) REFERENCES doctor(d_ID)
);
"""


def init_db(db_path: str = None):
    """初始化数据库，创建所有表"""
    conn = get_connection(db_path)
    try:
        conn.executescript(DDL_SCRIPT)
        conn.commit()
        print("数据库表初始化完成 (严格按工单7张表结构)")
    finally:
        conn.close()


# ============================================================
# CRUD 操作
# ============================================================

def get_all_departments(db_path: str = None) -> List[Dict]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM department")
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_doctors_by_dept(dep_id: int, profession: str = None, db_path: str = None) -> List[Dict]:
    conn = get_connection(db_path)
    try:
        query = "SELECT d.*, dep.dep_Name FROM doctor d LEFT JOIN department dep ON d.dep_ID = dep.dep_ID WHERE d.dep_ID = ?"
        params = [dep_id]
        if profession:
            query += " AND d.d_Profession = ?"
            params.append(profession)
        cur = conn.execute(query, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_doctor_by_name(name: str, db_path: str = None) -> Optional[Dict]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT d.*, dep.dep_Name FROM doctor d LEFT JOIN department dep ON d.dep_ID = dep.dep_ID WHERE d.d_Name = ?",
            (name,)
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_patient_by_name(name: str, db_path: str = None) -> Optional[Dict]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM patient WHERE p_Name = ?", (name,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_patient(name: str, sex: str = '男', address: str = '', birth: str = None, ps_id: int = 1, db_path: str = None) -> int:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO patient (p_Name, p_Sex, p_Address, p_Birth, ps_ID) VALUES (?, ?, ?, ?, ?)",
            (name, sex, address, birth, ps_id)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def create_register(dep_id: int, p_id: int, w_id: int = 1, reg_time: str = None, fee: int = 10, status: int = 1, db_path: str = None) -> int:
    conn = get_connection(db_path)
    try:
        # 获取当前科室该时间段的次序
        cur = conn.execute(
            "SELECT COALESCE(MAX(reg_Order), 0) + 1 as next_order FROM register WHERE dep_ID = ? AND reg_Time LIKE ?",
            (dep_id, reg_time[:10] + '%' if reg_time else '%')
        )
        order = cur.fetchone()["next_order"]

        cur = conn.execute(
            "INSERT INTO register (dep_ID, p_ID, w_ID, reg_Time, reg_Fee, reg_Order, reg_Status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (dep_id, p_id, w_id, reg_time, fee, order, status)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def cancel_register(reg_id: int, db_path: str = None) -> bool:
    conn = get_connection(db_path)
    try:
        # reg_Status: 0-取消, 1-正常, 2-已完成
        conn.execute("UPDATE register SET reg_Status = 0 WHERE reg_ID = ?", (reg_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def get_registers_by_patient(p_id: int, status: int = None, db_path: str = None) -> List[Dict]:
    conn = get_connection(db_path)
    try:
        query = """SELECT r.*, d.dep_Name, doc.d_Name as doctor_name 
               FROM register r 
               LEFT JOIN department d ON r.dep_ID = d.dep_ID 
               LEFT JOIN doctor doc ON r.dep_ID = doc.dep_ID
               WHERE r.p_ID = ?"""
        params = [p_id]
        if status is not None:
            query += " AND r.reg_Status = ?"
            params.append(status)
        query += " ORDER BY r.reg_Time DESC"
        cur = conn.execute(query, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_department_by_name(name: str, db_path: str = None) -> Optional[Dict]:
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM department WHERE dep_Name LIKE ?", (f"%{name}%",))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()



def get_schedule_by_doctor(d_id: int, db_path: str = None) -> List[Dict]:
    """查询某医生的坐诊时间"""
    conn = get_connection(db_path)
    try:
        cur = conn.execute("""
            SELECT s.*, d.d_Name, dep.dep_Name as dept_name
            FROM schedule s
            LEFT JOIN doctor d ON s.d_ID = d.d_ID
            LEFT JOIN department dep ON d.dep_ID = dep.dep_ID
            WHERE s.d_ID = ? AND s.s_Available = 1
            ORDER BY s.s_Date, s.s_TimeSlot
        """, (d_id,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_schedule_by_department(dep_id: int, db_path: str = None) -> List[Dict]:
    """查询某科室所有医生的坐诊时间"""
    conn = get_connection(db_path)
    try:
        cur = conn.execute("""
            SELECT s.*, d.d_Name, d.d_Profession, dep.dep_Name as dept_name
            FROM schedule s
            LEFT JOIN doctor d ON s.d_ID = d.d_ID
            LEFT JOIN department dep ON d.dep_ID = dep.dep_ID
            WHERE d.dep_ID = ? AND s.s_Available = 1
            ORDER BY s.s_Date, s.s_TimeSlot, d.d_Name
        """, (dep_id,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_available_slots(d_id: int, date: str = None, db_path: str = None) -> List[Dict]:
    """查询某医生在指定日期（或所有日期）的可挂号时段"""
    conn = get_connection(db_path)
    try:
        if date:
            cur = conn.execute("""
                SELECT s.*, d.d_Name, dep.dep_Name as dept_name
                FROM schedule s
                LEFT JOIN doctor d ON s.d_ID = d.d_ID
                LEFT JOIN department dep ON d.dep_ID = dep.dep_ID
                WHERE s.d_ID = ? AND s.s_Date = ? AND s.s_Available = 1
                ORDER BY s.s_TimeSlot
            """, (d_id, date))
        else:
            cur = conn.execute("""
                SELECT s.*, d.d_Name, dep.dep_Name as dept_name
                FROM schedule s
                LEFT JOIN doctor d ON s.d_ID = d.d_ID
                LEFT JOIN department dep ON d.dep_ID = dep.dep_ID
                WHERE s.d_ID = ? AND s.s_Available = 1
                ORDER BY s.s_Date, s.s_TimeSlot
            """, (d_id,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()



def seed_data(db_path: str = None):
    """插入基础测试数据"""
    conn = get_connection(db_path)
    try:
        # 插入科室
        depts = [
            ('儿科', '门诊楼2层'), ('牙科', '门诊楼1层'), ('眼科', '门诊楼3层'),
            ('皮肤科', '门诊楼2层'), ('消化内科', '门诊楼4层'), ('专家门诊', '特需楼1层')
        ]
        for name, addr in depts:
            conn.execute("INSERT OR IGNORE INTO department (dep_Name, dep_Address) VALUES (?, ?)", (name, addr))
        
        # 插入医生
        doctors = [
            ('李儿科', '男', '专家', 1), ('王牙科', '男', '专家', 2),
            ('张眼科', '男', '专家', 3), ('刘皮肤', '男', '普通', 4),
            ('陈消化', '男', '普通', 5), ('张建国', '男', '专家', 1)
        ]
        for name, sex, prof, dep_id in doctors:
            conn.execute("INSERT OR IGNORE INTO doctor (d_Name, d_Sex, d_Profession, dep_ID) VALUES (?, ?, ?, ?)",
                         (name, sex, prof, dep_id))
        
        # 插入状态
        conn.execute("INSERT OR IGNORE INTO patientstatus (ps_Name, ps_Remark) VALUES ('初诊', '首次就诊')")
        conn.execute("INSERT OR IGNORE INTO patientstatus (ps_Name, ps_Remark) VALUES ('复诊', '再次就诊')")
        
        # 插入挂号员
        conn.execute("INSERT OR IGNORE INTO worker (w_Name, w_LoginName, w_LoginPSW) VALUES ('系统Agent', 'agent', '123')")
        
        # 插入坐诊时间表 (未来7天)
        from datetime import datetime, timedelta
        today = datetime.now().date()
        time_slots = [("上午", "08:00-12:00"), ("下午", "14:00-17:30")]
        
        # 医生排班: (医生ID, 周几出诊)
        # 1-李儿科(周一三五), 2-王牙科(周二四), 3-张眼科(周一三五), 
        # 4-刘皮肤(周二四六), 5-陈消化(周一三五), 6-张建国(周二四六)
        doctor_schedules = {
            1: [0, 2, 4],  # 周一三五
            2: [1, 3],     # 周二四
            3: [0, 2, 4],  # 周一三五
            4: [1, 3, 5],  # 周二四六
            5: [0, 2, 4],  # 周一三五
            6: [1, 3, 5],  # 周二四六
        }
        
        for day_offset in range(7):
            date = today + timedelta(days=day_offset)
            weekday = date.weekday()  # 0=Monday
            
            for d_id, weekdays in doctor_schedules.items():
                if weekday in weekdays:
                    for slot_name, slot_time in time_slots:
                        conn.execute("""
                            INSERT OR IGNORE INTO schedule (d_ID, s_Date, s_TimeSlot, s_MaxPatients, s_Available)
                            VALUES (?, ?, ?, 20, 1)
                        """, (d_id, date.isoformat(), f"{slot_name} {slot_time}"))
        
        conn.commit()
        print("基础测试数据已插入")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    seed_data()
    print("数据库初始化完成！")
    print("Tables:", [r[0] for r in get_connection().execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()])


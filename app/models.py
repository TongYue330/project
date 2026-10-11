import sqlite3
import os
from .logger import logger


def get_db(app):
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn


def init_db(app):
    os.makedirs(os.path.dirname(app.config['DATABASE']), exist_ok=True)
    conn = get_db(app)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS question (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_type TEXT NOT NULL CHECK(input_type IN ('image', 'text')),
            content TEXT,
            image_path TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS answer (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            steps_json TEXT,
            confidence REAL DEFAULT 1.0,
            is_low_confidence BOOLEAN DEFAULT 0,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (question_id) REFERENCES question(id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()
    logger.info('数据库初始化完成')


def insert_question(app, input_type, content=None, image_path=None):
    conn = get_db(app)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO question (input_type, content, image_path) VALUES (?, ?, ?)',
        (input_type, content, image_path)
    )
    qid = cursor.lastrowid
    conn.commit()
    conn.close()
    return qid


def insert_answer(app, question_id, content, steps_json=None,
                  confidence=1.0, is_low_confidence=False):
    conn = get_db(app)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO answer (question_id, content, steps_json, confidence, is_low_confidence) '
        'VALUES (?, ?, ?, ?, ?)',
        (question_id, content, steps_json, confidence, 1 if is_low_confidence else 0)
    )
    aid = cursor.lastrowid
    conn.commit()
    conn.close()
    return aid


def list_questions_with_answers(app):
    try:
        conn = get_db(app)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT q.id, q.input_type, q.content, q.created_at,
                   a.content AS answer_content, a.is_low_confidence
            FROM question q
            LEFT JOIN answer a ON a.question_id = q.id
            ORDER BY q.created_at DESC
        ''')
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    except sqlite3.Error as e:
        logger.error(f'数据库读取失败：{e}')
        return []


def get_question_detail(app, question_id):
    try:
        conn = get_db(app)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM question WHERE id = ?', (question_id,))
        q = cursor.fetchone()
        if not q:
            conn.close()
            return None

        cursor.execute(
            'SELECT * FROM answer WHERE question_id = ? ORDER BY id DESC LIMIT 1',
            (question_id,)
        )
        a = cursor.fetchone()
        conn.close()

        return {
            'question': dict(q),
            'answer': dict(a) if a else None
        }
    except sqlite3.Error as e:
        logger.error(f'查询详情失败：question_id={question_id}, error={e}')
        return None
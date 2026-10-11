from app import create_app
from app.models import init_db, insert_question, insert_answer, get_db
import json

app = create_app()

with app.app_context():
    init_db(app)
    conn = get_db(app)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM answer')
    cursor.execute('DELETE FROM question')
    conn.commit()
    conn.close()

    qid1 = insert_question(app, 'text', content='请解释快速排序的时间复杂度')
    insert_answer(app, qid1,
                  content='快速排序的平均时间复杂度为O(n log n)，最坏情况为O(n²)。',
                  steps_json=json.dumps(['步骤1：选择基准元素', '步骤2：分区操作', '步骤3：递归排序'], ensure_ascii=False),
                  confidence=0.95)

    qid2 = insert_question(app, 'text', content='求 f(x) = x³ + 2x 的导数')
    insert_answer(app, qid2,
                  content="f'(x) = 3x² + 2",
                  steps_json=json.dumps(['步骤1：对x³求导得3x²', '步骤2：对2x求导得2', '步骤3：合并结果'], ensure_ascii=False),
                  confidence=0.92)

    qid3 = insert_question(app, 'text', content='有一道超纲的偏微分方程题，请帮我解答')
    insert_answer(app, qid3,
                  content='本题超出模型知识范围，建议咨询专业老师。',
                  steps_json=json.dumps(['无法确定答案'], ensure_ascii=False),
                  confidence=0.35,
                  is_low_confidence=True)

    print('样例数据已初始化：3 条题目记录')
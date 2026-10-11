from app import create_app
from app.models import get_db

app = create_app()

with app.app_context():
    conn = get_db(app)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM answer')
    cursor.execute('DELETE FROM question')
    conn.commit()
    conn.close()
    print('业务数据已清空，表结构保留')
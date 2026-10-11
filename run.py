from app import create_app
from app.models import init_db

app = create_app()

with app.app_context():
    init_db(app)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
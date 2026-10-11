from flask import Flask
import os


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'course-project-dev-key'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app.config['DATABASE'] = os.path.join(base_dir, 'data', 'app.db')
    app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'data', 'uploads')

    from . import routes
    app.register_blueprint(routes.bp)

    return app
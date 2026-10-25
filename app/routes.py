from .logger import logger
import json
from flask import Blueprint, render_template, request, redirect, url_for, current_app
from . import models
from .ai_mock import mock_answer
from .rag import answer_question

bp = Blueprint('main', __name__)


@bp.route('/')
def index():
    return render_template('index.html')


@bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if not text:
            return render_template('upload.html', error='请输入题目内容')

        if request.form.get('mode') == 'llm':
            raw = answer_question(text, use_rag=request.form.get('use_rag') == 'on')
            if not raw['success']:
                return render_template('upload.html', error=raw['error'])
            source_line = '\n\n检索来源：' + '、'.join(raw['sources']) if raw['sources'] else '\n\n检索来源：无（资料不足）'
            result = {'success': True, 'content': raw['content'] + source_line, 'steps': [x for x in raw['content'].splitlines() if x.strip()], 'confidence': 0.9 if raw['contexts'] and raw.get('citation_valid', True) else 0.7, 'is_low_confidence': not bool(raw['contexts']) or not raw.get('citation_valid', True)}
        else:
            result = mock_answer(text)
            if not result['success']:
                return render_template('upload.html', error=result['message'])

        qid = models.insert_question(current_app, 'text', content=text)
        models.insert_answer(
            current_app, qid,
            content=result['content'],
            steps_json=json.dumps(result['steps'], ensure_ascii=False),
            confidence=result['confidence'],
            is_low_confidence=result['is_low_confidence']
        )
        return redirect(url_for('main.detail', question_id=qid))

    return render_template('upload.html')


@bp.route('/history')
def history():
    records = models.list_questions_with_answers(current_app)
    return render_template('history.html', records=records)


@bp.route('/detail/<int:question_id>')
def detail(question_id):
    data = models.get_question_detail(current_app, question_id)
    if not data:
        return render_template('detail.html', error='记录不存在'), 404
    if data['answer'] and data['answer'].get('steps_json'):
        try:
            data['answer']['steps'] = json.loads(data['answer']['steps_json'])
        except json.JSONDecodeError:
            data['answer']['steps'] = []
    return render_template('detail.html', data=data)

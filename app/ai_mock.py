import random


def mock_answer(question_text):
    """模拟AI解答，本次实验不调用真实API"""
    if not question_text or len(question_text.strip()) < 3:
        return {
            'success': False,
            'message': '题目内容太短，请重新输入'
        }

    confidence = round(random.uniform(0.6, 0.99), 2)
    is_low = confidence < 0.7

    return {
        'success': True,
        'content': f'模拟解答：关于“{question_text[:20]}”的详细解答。',
        'steps': ['步骤1：识别题目类型', '步骤2：分析解题思路', '步骤3：得出最终答案'],
        'confidence': confidence,
        'is_low_confidence': is_low
    }
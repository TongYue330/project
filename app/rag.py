"""LLM and keyword-RAG integration for the course Q&A assistant."""
from __future__ import annotations
import json, os, re, time
from pathlib import Path
from typing import Any
from urllib import request as urlrequest, error as urlerror
DEFAULT_MODEL = "qwen-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
def _knowledge_dir(): return Path(__file__).resolve().parent.parent / "data" / "knowledge_base"
def load_knowledge_base():
    chunks=[]
    for path in sorted(_knowledge_dir().glob("*.md")):
        paragraphs=[p.strip() for p in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")) if p.strip()]
        for index, paragraph in enumerate(paragraphs, 1): chunks.append({"id":f"{path.stem}-{index}","source":path.name,"text":paragraph})
    return chunks
def _terms(text):
    # Keep Chinese phrases intact enough to avoid matching every common single character.
    words = re.findall(r"[A-Za-z0-9_+#.-]+|[\u4e00-\u9fff]{2,}", text.lower())
    terms = set(words)
    for word in words:
        if len(word) >= 4 and not word.isascii():
            terms.update(word[i:i+2] for i in range(len(word)-1))
    return {w for w in terms if w.strip()}
def retrieve(question, top_k=3):
    qt=_terms(question); ranked=[]
    for c in load_knowledge_base():
        overlap=qt & _terms(c["text"]); score=len(overlap)/max(len(qt),1)
        if score: ranked.append({**c,"score":round(score,4),"matched_terms":sorted(overlap)})
    ranked.sort(key=lambda x:(x["score"],len(x["matched_terms"])), reverse=True)
    return ranked[:top_k]
def build_prompt(question, contexts, history=None):
    sources="\n\n".join(f"[{c['id']}] 来源：{c['source']}\n{c['text']}" for c in contexts) or "（没有检索到相关资料）"
    messages=[{"role":"system","content":"你是课程作业AI答疑助手。只根据给定资料回答；资料不足时明确说资料不足，不要编造事实。回答按‘结论、解释、依据’组织，引用资料时只能使用实际提供的方括号来源编号。"}]
    if history: messages.extend(history[-4:])
    messages.append({"role":"user","content":f"检索资料：\n{sources}\n\n用户问题：{question}"})
    return messages
def call_llm(messages, timeout=20):
    key=os.getenv("DASHSCOPE_API_KEY","").strip(); started=time.perf_counter()
    if not key: return {"success":False,"error":"未配置 DASHSCOPE_API_KEY","elapsed_ms":0}
    payload=json.dumps({"model":os.getenv("DASHSCOPE_MODEL",DEFAULT_MODEL),"messages":messages,"temperature":0.2,"top_p":0.8,"stream":False}).encode()
    req=urlrequest.Request(os.getenv("DASHSCOPE_BASE_URL",DEFAULT_BASE_URL), data=payload, headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp: data=json.loads(resp.read().decode())
        elapsed=round((time.perf_counter()-started)*1000); content=data.get("choices",[{}])[0].get("message",{}).get("content","").strip()
        if not content: return {"success":False,"error":"模型返回内容为空","elapsed_ms":elapsed}
        return {"success":True,"content":content,"elapsed_ms":elapsed,"usage":data.get("usage",{}),"model":data.get("model",os.getenv("DASHSCOPE_MODEL",DEFAULT_MODEL))}
    except urlerror.HTTPError as e: return {"success":False,"error":f"接口返回 HTTP {e.code}","elapsed_ms":round((time.perf_counter()-started)*1000)}
    except (urlerror.URLError, TimeoutError): return {"success":False,"error":"模型连接失败或超时","elapsed_ms":round((time.perf_counter()-started)*1000)}
    except (ValueError, KeyError, IndexError, TypeError): return {"success":False,"error":"模型返回格式无法解析","elapsed_ms":round((time.perf_counter()-started)*1000)}
def answer_question(question, use_rag=True):
    contexts=retrieve(question) if use_rag else []; prompt=build_prompt(question,contexts); result=call_llm(prompt); result.update({"contexts":contexts,"prompt":prompt}); result["sources"]=[c["id"] for c in contexts] if result.get("success") else []
    if result.get("success"):
        cited=set(re.findall(r"\[([^\]]+)\]", result.get("content", "")))
        allowed=set(result["sources"])
        result["citation_valid"] = cited.issubset(allowed) if cited else not allowed
    return result
def serialize_result(result): return json.dumps(result,ensure_ascii=False)

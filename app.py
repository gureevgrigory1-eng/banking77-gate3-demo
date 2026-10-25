"""Local Gate 3 demo. Run: python app.py"""

from __future__ import annotations

import os
import time
from collections import deque
from threading import Lock

from flask import Flask, jsonify, render_template, request

from rag import RAGPrototype, normalize_history

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
prototype = RAGPrototype()
requests_by_client: dict[str, deque[float]] = {}
request_lock = Lock()
HOURLY_LIMIT = 30


def over_limit(client: str) -> bool:
    now = time.monotonic()
    with request_lock:
        recent = requests_by_client.setdefault(client, deque())
        while recent and now - recent[0] > 3600:
            recent.popleft()
        if len(recent) >= HOURLY_LIMIT:
            return True
        recent.append(now)
        if len(requests_by_client) > 1000:
            for address, timestamps in list(requests_by_client.items()):
                if not timestamps or now - timestamps[-1] > 3600:
                    del requests_by_client[address]
    return False


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok", "model": prototype.remote.model,
        "remote_enabled": prototype.remote.available,
        "remote_provider": prototype.remote.provider if prototype.remote.available else None,
        "rag": True, "article_count": len(prototype.retriever.articles),
        "semantic_index_ready": prototype.retriever.vectors is not None,
        "kb_version": prototype.retriever.articles[0]["version"],
        "knowledge_base": "fictional demo",
    })


@app.post("/api/ask")
def ask():
    if over_limit(request.remote_addr or "unknown"):
        return jsonify({"error": "Достигнут лимит демо: 30 вопросов в час. Попробуйте позже."}), 429
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Ожидается JSON-объект"}), 400
    raw_query = payload.get("query", "")
    raw_history = payload.get("history", [])
    if not isinstance(raw_query, str) or not isinstance(raw_history, list):
        return jsonify({"error": "Некорректный формат вопроса или истории"}), 400
    history = normalize_history(raw_history)
    try:
        answer = prototype.ask(raw_query, history)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(answer)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8765")), debug=False)

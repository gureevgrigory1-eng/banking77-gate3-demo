"""Conservative Banking77 intent router for a fictional bank demo."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib

from remote_router import RemoteRouter, RemoteRouterError

ROOT = Path(__file__).resolve().parent
SAFE_INTENTS = {
    "card_delivery_estimate",
    "transfer_timing",
    "exchange_rate",
    "card_acceptance",
    "visa_or_mastercard",
}
PERSONAL_INTENT = "card_arrival"
RISK_INTENTS = {
    "card_payment_not_recognised",
    "cash_withdrawal_not_recognised",
    "direct_debit_payment_not_recognised",
    "compromised_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "passcode_forgotten",
    "pin_blocked",
    "cancel_transfer",
    "request_refund",
    "transaction_charged_twice",
    "terminate_account",
}
RISK_PATTERNS = [
    r"\b(fraud|scam|stolen|compromised|unauthori[sz]ed|not recogni[sz]e|didn.t make|unknown charge|suspicious|cvv|password|passcode|pin blocked)\b",
    r"(мошенн|украл|похит|краж|ограб|неизвестн.{0,12}списан|не узна[юе].{0,18}списан|не совершал|не делал.{0,18}плат[её]ж|cvv|парол|пин.?код|заблокир)",
    r"\b(please\s+)?(send|transfer)\s+money\s+to\b|\bпереведи\s+деньги\b",
]


def is_russian(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text))


def hard_risk(text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in RISK_PATTERNS)


class Prototype:
    def __init__(self, root: Path = ROOT, remote=None):
        self.root = Path(root)
        artifact = joblib.load(self.root / "model.joblib")
        self.model = artifact["model"]
        self.threshold = artifact["threshold"]
        self.personal_threshold = artifact["personal_threshold"]
        ru_artifact = self.root / "ru_model.joblib"
        self.ru_model = joblib.load(ru_artifact) if ru_artifact.exists() else None
        self.kb = {row["intent"]: row for row in json.loads((self.root / "knowledge_base_legacy.json").read_text(encoding="utf-8"))}
        self.remote = remote if remote is not None else RemoteRouter(self.root)

    def classify(self, query: str):
        probs = self.model.predict_proba([query])[0]
        ranked = probs.argsort()[::-1]
        return [(str(self.model.classes_[i]), float(probs[i])) for i in ranked[:3]]

    def ask(self, query: str, history=None):
        query = query.strip()[:1000]
        history = history or []
        if not query:
            raise ValueError("Введите вопрос")
        if hard_risk(query):
            decision = None
            if self.remote.available:
                try:
                    decision = self.remote.classify(query, history)
                except RemoteRouterError:
                    pass
            return self._handoff(
                query, history, "Правило безопасности: возможен рискованный запрос", [],
                decision.get("operator_draft", "") if decision else "",
                self.remote.provider if decision else "",
            )
        if query.lower().startswith("демо:"):
            return self._demo_status()
        if self.remote.available:
            try:
                decision = self.remote.classify(query, history)
                return self._from_remote(query, history, decision)
            except RemoteRouterError as exc:
                remote_error = str(exc)
        else:
            remote_error = "Удалённая модель не настроена"
        if is_russian(query):
            if self.ru_model:
                return self._from_local_ru(query, history)
            return self._clarify("Уточните, пожалуйста, о какой операции или продукте речь?", None, f"Русскоязычная модель недоступна: {remote_error}")
        top = self.classify(query)
        intent, confidence = top[0]
        if intent in RISK_INTENTS and confidence >= 0.3:
            return self._handoff(query, history, "Сценарий требует участия сотрудника", top)
        if intent in SAFE_INTENTS and confidence >= self.threshold:
            return self._knowledge_answer(intent, round(confidence, 4), top, "Локальная модель Banking77")
        if intent == PERSONAL_INTENT and confidence >= self.personal_threshold:
            return self._clarify(
                "Я не вижу статус вашей реальной карты. Вы спрашиваете о доставке новой карты или о том, можно ли ею пользоваться?",
                intent,
                "Запрос о конкретной карте; банковского подключения нет",
                confidence,
            )
        return self._clarify(
            "Уточните, пожалуйста, ваш вопрос одним предложением: о какой операции или продукте речь?",
            intent,
            f"Недостаточно данных для проверенного ответа. {remote_error}",
            confidence,
        )

    def _from_remote(self, query, history, decision):
        intent = decision["intent"] if decision["intent"] != "none" else None
        kind = decision["kind"]
        reason = f"{self.remote.provider}: {str(decision['reason'])[:180]}"
        if kind in {"risk", "human_request"} or intent in RISK_INTENTS:
            return self._handoff(query, history, reason, [(intent, 0)] if intent else [], decision.get("operator_draft", ""), self.remote.provider)
        if kind == "greeting":
            return self._clarify("Здравствуйте! Чем могу помочь по карте, платежу или переводу?", None, reason)
        if kind == "general" and intent in SAFE_INTENTS and intent in self.kb:
            return self._knowledge_answer(intent, None, [(intent, 0)], reason)
        question = str(decision.get("clarifying_question", "")).strip()[:250]
        if kind == "personal":
            answer = "Я не вижу ваши реальные карты и операции. " + (question or "Уточните, о какой операции речь?")
        else:
            answer = question or "Уточните, пожалуйста, о какой операции или продукте речь?"
        if kind == "general" and intent not in self.kb:
            answer = "По этой теме в учебной базе пока нет проверенного ответа. " + answer
        choices = decision.get("clarifying_choices", []) if kind in {"personal", "unclear"} else []
        return self._clarify(answer, intent, reason, choices=choices)

    def _from_local_ru(self, query, history):
        model = self.ru_model["model"]
        probs = model.predict_proba([query])[0]
        ranked = probs.argsort()[::-1]
        label = str(model.classes_[ranked[0]])
        confidence = float(probs[ranked[0]])
        top = [(str(model.classes_[i]), round(float(probs[i]), 4)) for i in ranked[:3]]
        reason = "Локальная русскоязычная модель на синтетических учебных примерах"
        if label == "risk" and confidence >= 0.45:
            return self._handoff(query, history, "Русская модель: возможный риск; нужна проверка сотрудника", top)
        if label in SAFE_INTENTS and confidence >= self.ru_model["threshold"]:
            if re.search(r"\b(мой|моя|мою|моей|мои|моего|у меня|для меня)\b", query, re.I):
                return self._clarify(
                    "Я не вижу ваши реальные данные. Вы спрашиваете об общих условиях или о статусе вашего случая?",
                    label, "Упоминание личного случая: ответ требует уточнения", confidence,
                    ["Общие условия", "Мой конкретный случай"],
                )
            return self._knowledge_answer(label, round(confidence, 4), top, reason)
        personal = {
            "personal_payment": (
                "Я не вижу вашу операцию. Речь об оплате картой или о переводе?",
                ["Оплата картой", "Перевод"],
            ),
            "personal_card": (
                "Я не вижу статус вашей карты. Вы спрашиваете о доставке или о возможности ею пользоваться?",
                ["Доставка карты", "Работа карты"],
            ),
            "personal_fee": (
                "Я не вижу ваш тариф. О какой операции и комиссии вы спрашиваете?",
                ["Перевод", "Снятие наличных", "Обслуживание карты"],
            ),
        }
        if label in personal and confidence >= 0.35:
            question, choices = personal[label]
            return self._clarify(question, label, reason, confidence, choices)
        return self._clarify(
            "Уточните, пожалуйста, о каком банковском продукте или операции речь?",
            label, f"Недостаточная уверенность русскоязычной модели ({confidence:.2f}); безопасное уточнение",
            confidence,
        )

    def _knowledge_answer(self, intent, confidence, top, reason):
        article = self.kb[intent]
        return {
            "mode": "autopilot", "answer": article["answer"], "intent": intent,
            "confidence": confidence, "top_intents": top, "article": article,
            "reason": reason,
            "note": "Источник составлен для учебного Демо-банка; это не условия реального банка.",
        }

    @staticmethod
    def _clarify(answer, intent, reason, confidence=None, choices=None):
        return {
            "mode": "clarify", "answer": answer, "intent": intent,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "top_intents": [], "reason": reason,
            "clarifying_choices": [str(choice).strip()[:80] for choice in (choices or [])[:3] if str(choice).strip()],
            "note": "Ассистент уточняет вопрос. Передачи реальному сотруднику в прототипе нет.",
        }

    @staticmethod
    def _demo_status():
        return {
            "mode": "personal_demo",
            "answer": "По синтетической записи учебного клиента карта выпущена и ожидается 21 октября 2026 года. Это демонстрационный пример, не статус вашей карты.",
            "intent": "card_arrival", "confidence": None, "top_intents": [],
            "article": {"id": "MOCK-001", "title": "Тестовый заказ карты", "source": "Синтетическая запись", "valid_from": "2026-10-18"},
            "reason": "Пользователь явно запросил демо-данные",
            "note": "Только тестовые данные. Нет подключения к банковским системам.",
        }

    @staticmethod
    def _handoff(query, history, reason, top, draft="", provider=""):
        prior = [str(x.get("text", ""))[:200] for x in history[-3:] if isinstance(x, dict)]
        summary = " → ".join(prior + [query])
        draft = str(draft).strip()[:500]
        return {
            "mode": "copilot",
            "answer": "Здесь нужен сотрудник. Прототип подготовил для него краткий контекст, но реального подключения к поддержке пока нет.",
            "reason": reason,
            "intent": top[0][0] if top else None,
            "confidence": round(top[0][1], 4) if top and top[0][1] else None,
            "top_intents": top,
            "operator_summary": summary,
            "operator_draft": draft or "Проверю ваш вопрос по актуальным данным. Если потребуется, уточню детали, прежде чем дать ответ.",
            "draft_source": f"Черновик {provider} · требует проверки сотрудником" if draft else "Шаблонный черновик · требует проверки сотрудником",
            "note": "Передача сотруднику показана в интерфейсе; реальной очереди операторов нет.",
        }

"""Check the manually curated knowledge base against the product catalog."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
articles = json.loads((ROOT / "knowledge_base.json").read_text(encoding="utf-8"))
catalog = json.loads((ROOT / "product_catalog.json").read_text(encoding="utf-8"))
by_id = {article["id"]: article for article in articles}

assert len(articles) == 32 and len(by_id) == len(articles)
assert all(article["version"] == catalog["version"] for article in articles)
assert all(article["valid_from"] == catalog["valid_from"] for article in articles)
assert all(article["disclaimer"] == catalog["disclaimer"] for article in articles)
assert all(not re.search(r"учебн|gate\s*\d", article["content"], re.I) for article in articles)

regions = catalog["card_usage"]["regions"]
for article_id in ("D01", "C01", "J01"):
    assert all(region[:-1] in by_id[article_id]["content"] for region in regions)

transfers = catalog["transfers"]
for article_id in ("G03", "G08"):
    text = by_id[article_id]["content"]
    assert f"до {transfers['internal_minutes']} минуты" in text
    assert f"до {transfers['external_days']} рабочего дня" in text
    assert "СБП" in text and f"до {transfers['sbp_days']} рабочего дня" in text

assert f"в течение {catalog['card_usage']['statement_posting_workdays']} рабочих дней" in by_id["G08"]["content"]
debit = catalog["products"]["debit"]
cash_text = by_id["D04"]["content"]
assert f"до {debit['cash_free_month']} ₽" in cash_text
assert f"{debit['cash_over_pct']}%, минимум {debit['cash_over_min']} ₽" in cash_text
delivery_text = by_id["D02"]["content"]
assert "Доставка бесплатная" in delivery_text
assert not re.search(r"встреча.{0,30}бесплатн|бесплатн.{0,30}встреч", delivery_text, re.I)
print(f"Проверено: {len(articles)} статьи, версия {catalog['version']}; каталог и база согласованы.")

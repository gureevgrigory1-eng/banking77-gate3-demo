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
print(f"Проверено: {len(articles)} статьи, версия {catalog['version']}; каталог и база согласованы.")

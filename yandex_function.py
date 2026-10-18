"""HTTP adapter for Yandex Cloud Functions."""

import base64
from urllib.parse import urlencode

from app import app


def handler(event, context):
    headers = event.get("headers") or {}
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body)

    path = event.get("path") or "/"
    params = event.get("queryStringParameters") or {}
    if params:
        path += "?" + urlencode(params)

    identity = (event.get("requestContext") or {}).get("identity") or {}
    source_ip = identity.get("sourceIp") or "unknown"
    with app.test_client() as client:
        response = client.open(
            path,
            method=event.get("httpMethod") or "GET",
            headers=headers,
            data=body,
            environ_base={"REMOTE_ADDR": source_ip},
        )
    return {
        "statusCode": response.status_code,
        "headers": {"Content-Type": response.content_type},
        "isBase64Encoded": False,
        "body": response.get_data(as_text=True),
    }

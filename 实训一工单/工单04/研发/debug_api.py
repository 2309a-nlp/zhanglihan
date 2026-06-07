# -*- coding: utf-8 -*-
"""Windows API connectivity test (urllib version)"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import config
import urllib.request
import urllib.error
import json

print(f"API Key prefix: {config.DEEPSEEK_API_KEY[:15]}...")
print(f"API Base: {config.DEEPSEEK_API_BASE}")
print(f"Model: {config.DEEPSEEK_MODEL}")

try:
    payload = json.dumps({
        "model": config.DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{config.DEEPSEEK_API_BASE}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"HTTP Status: {resp.status}")
        print(f"API test success!")
        print(f"Response: {data['choices'][0]['message']['content']}")

except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.reason}")
    print(f"Body: {e.read().decode('utf-8', errors='replace')[:200]}")
except urllib.error.URLError as e:
    print(f"Connection error: {e.reason}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

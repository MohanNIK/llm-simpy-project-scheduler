import os

import requests

URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def main() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        raise SystemExit("Set DASHSCOPE_API_KEY or QWEN_API_KEY before running this connectivity check.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "qwen-plus",
        "messages": [
            {"role": "system", "content": "You are a connectivity check."},
            {"role": "user", "content": "Reply with the single word: success"},
        ],
    }
    response = requests.post(URL, headers=headers, json=payload, timeout=30)
    print(response.status_code)
    print(response.text)


if __name__ == "__main__":
    main()

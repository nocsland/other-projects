import requests

url = "http://127.0.0.1:8000/translate/ru-to-en/"
payload = {"text": "Быстрая коричневая лиса перепрыгивает через ленивую собаку."}
response = requests.post(url, json=payload)
print(response.json())

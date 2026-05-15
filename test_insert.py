"""
Script de teste para a rota POST /sensores.
Envia um JSON simulando dados do ESP32 para a API local.

Uso:
    python test_insert.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8100/sensores")
API_KEY = os.getenv("API_KEY", "")

payload = {
    "cidade": "São Paulo",
    "estado": "SP",
    "pais": "Brasil",
    "latitude": -23.5505,
    "longitude": -46.6333,
    "temperatura": 25.4,
    "humidade": 60.2,
    "co2": 412,
    "device_id": "ESP32-TEST-001",
}

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
}

print("=" * 60)
print("Teste de inserção - POST /sensores")
print("=" * 60)
print(f"URL:     {API_URL}")
print(f"Payload: {payload}")
print(f"API Key: {'***' + API_KEY[-4:] if len(API_KEY) > 4 else '(vazia)'}")
print("-" * 60)

try:
    response = requests.post(API_URL, json=payload, headers=headers, timeout=10)
    print(f"Status:  {response.status_code}")
    print(f"Resposta: {response.json()}")

    if response.status_code == 201:
        print("\n✅ Inserção realizada com sucesso!")
    else:
        print(f"\n❌ Erro: HTTP {response.status_code}")
except requests.exceptions.ConnectionError:
    print("\n❌ Erro de conexão. Verifique se a API está rodando.")
except Exception as exc:
    print(f"\n❌ Erro inesperado: {exc}")

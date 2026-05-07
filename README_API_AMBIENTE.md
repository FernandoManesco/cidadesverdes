# README - API AMBIENTE (FastAPI + Firebird)

Este documento adiciona **uma nova API** para dados ambientais na porta **8101**, sem alterar a API existente na porta **8100**.

## 1) Estrutura criada

```text
cidadesverdes/
├── app/                         # API existente (porta 8100)
├── app_ambiente/
│   ├── __init__.py
│   └── main.py                  # Nova API (porta 8101)
├── config.py                    # Variáveis de ambiente compartilhadas
├── sql/
│   └── create_ambiente.sql      # Script SQL da tabela ambiente
├── scripts/
│   ├── setup_firebird.sh        # Instala/verifica Firebird 3.0
│   ├── create_database.py       # Cria banco e schema ambiente
│   └── run_api_ambiente.sh      # Sobe API ambiente
├── systemd/
│   └── api-ambiente.service     # Serviço systemd
├── .env.example
└── requirements.txt
```

## 2) Instalação do Firebird

### 2.1 Script automático

```bash
cd /home/deploy/var/www/sistema-gestao/cidadesverdes
chmod +x scripts/setup_firebird.sh
./scripts/setup_firebird.sh
```

> O script instala Firebird 3.0 e prepara `/home/deploy/var/databases`.

### 2.2 Criar banco e tabela

```bash
cd /home/deploy/var/www/sistema-gestao/cidadesverdes
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/create_database.py
```

Banco padrão: `/home/deploy/var/databases/cidadesverdes.fdb`.

## 3) Variáveis de ambiente e GitHub Secrets

Copie `.env.example` para `.env` e ajuste os valores reais:

```bash
cp .env.example .env
```

Secrets recomendados no GitHub Actions (Repository → Settings → Secrets and variables → Actions):

- `FIREBIRD_HOST`
- `FIREBIRD_PORT`
- `FIREBIRD_DATABASE`
- `FIREBIRD_USER`
- `FIREBIRD_PASSWORD`
- `FIREBIRD_CHARSET`
- `API_PORT` (8101)
- `API_HOST` (0.0.0.0)
- `CORS_ALLOW_ORIGINS`

No workflow, exporte os secrets para variáveis de ambiente antes de iniciar a API.

## 4) Rodar API localmente

```bash
cd /home/deploy/var/www/sistema-gestao/cidadesverdes
source .venv/bin/activate
pip install -r requirements.txt
./scripts/run_api_ambiente.sh
```

A API responderá em `http://127.0.0.1:8101`.

Swagger:
- `http://127.0.0.1:8101/docs`
- `http://127.0.0.1:8101/redoc`

## 5) Endpoints

### Healthcheck
- `GET /`

### Inserir registro
- `POST /ambiente`

Body JSON:

```json
{
  "temperatura": 26.5,
  "umidade": 68.2,
  "co2": 412.1,
  "bairro": "Centro",
  "cidade": "Curitiba",
  "estado": "PR",
  "pais": "Brasil",
  "latitude": "-25.4284",
  "longitude": "-49.2733",
  "data": "2026-05-07",
  "hora": "14:30:00",
  "origemdaleitura": "sensor_a1"
}
```

### Excluir registro
- `DELETE /ambiente/{id}`

### Listar registros
- `GET /ambiente`
- `GET /ambiente?cidade=Curitiba`
- `GET /ambiente?data=2026-05-07`
- `GET /ambiente?bairro=Centro`
- (filtros podem ser combinados)

### Buscar por ID
- `GET /ambiente/{id}`

## 6) Exemplos de requisições (curl)

```bash
# Inserir
curl -X POST 'http://127.0.0.1:8101/ambiente' \
  -H 'Content-Type: application/json' \
  -d '{
    "temperatura": 25.0,
    "umidade": 60.0,
    "co2": 400.0,
    "bairro": "Centro",
    "cidade": "São Paulo",
    "estado": "SP",
    "pais": "Brasil",
    "latitude": "-23.5505",
    "longitude": "-46.6333",
    "data": "2026-05-07",
    "hora": "10:00:00",
    "origemdaleitura": "iot_gateway"
  }'

# Listar todos
curl 'http://127.0.0.1:8101/ambiente'

# Filtrar por cidade
curl 'http://127.0.0.1:8101/ambiente?cidade=São%20Paulo'

# Buscar por ID
curl 'http://127.0.0.1:8101/ambiente/1'

# Excluir por ID
curl -X DELETE 'http://127.0.0.1:8101/ambiente/1'
```

## 7) Serviço systemd (auto-start)

```bash
sudo cp systemd/api-ambiente.service /etc/systemd/system/api-ambiente.service
sudo systemctl daemon-reload
sudo systemctl enable api-ambiente
sudo systemctl restart api-ambiente
sudo systemctl status api-ambiente
```

## 8) Observações importantes

- A API existente (`app/main.py`, porta 8100) **não foi substituída**.
- A nova API fica em `app_ambiente/main.py`, porta 8101.
- Se usar Nginx, crie bloco/rota separado para 8101 conforme necessidade.

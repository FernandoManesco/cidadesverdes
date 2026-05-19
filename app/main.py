from datetime import date, datetime, time
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import get_settings
from app.db import get_connection


settings = get_settings()


# ---------------------------------------------------------------------------
# Proteção por API Key
# ---------------------------------------------------------------------------

def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """Valida o header X-API-Key contra a chave definida no .env."""
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY não configurada no servidor",
        )
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key inválida",
        )
    return x_api_key


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------

class AmbienteCreate(BaseModel):
    temperatura: float
    umidade: float
    co2: float
    bairro: str = Field(..., min_length=1, max_length=120)
    cidade: str = Field(..., min_length=1, max_length=120)
    estado: str = Field(..., min_length=1, max_length=120)
    pais: str = Field(..., min_length=1, max_length=120)
    latitude: str = Field(..., min_length=1, max_length=30)
    longitude: str = Field(..., min_length=1, max_length=30)
    data: date
    hora: time
    origemdaleitura: str = Field(..., min_length=1, max_length=255)


class AmbienteOut(AmbienteCreate):
    id: int
    created_at: datetime


class SensorCreate(BaseModel):
    """Payload recebido do ESP32."""
    cidade: str = Field(..., max_length=100)
    estado: str = Field(..., max_length=50)
    pais: str = Field(..., max_length=50)
    latitude: float
    longitude: float
    temperatura: float
    humidade: float
    co2: int
    device_id: str = Field(..., max_length=100)


class SensorOut(BaseModel):
    """Resposta de sucesso ao inserir sensor."""
    success: bool = True
    message: str = "Dados inseridos com sucesso"
    device_id: str


class ErrorResponse(BaseModel):
    detail: str


def _row_to_dict(row: tuple) -> Dict[str, Any]:
    return {
        "id": row[0],
        "temperatura": row[1],
        "umidade": row[2],
        "co2": row[3],
        "bairro": row[4],
        "cidade": row[5],
        "estado": row[6],
        "pais": row[7],
        "latitude": row[8],
        "longitude": row[9],
        "data": row[10].isoformat() if row[10] else None,
        "hora": row[11].isoformat() if row[11] else None,
        "origemdaleitura": row[12],
        "created_at": row[13].isoformat() if row[13] else None,
    }


app = FastAPI(
    title="Cidades Verdes - API Ambiente",
    description="API para operações CRUD de dados ambientais no Firebird",
    version="1.0.0",
)

allow_origins = ["*"] if settings.cors_allow_origins.strip() == "*" else [
    origin.strip() for origin in settings.cors_allow_origins.split(",") if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def healthcheck():
    return {"status": "ok", "api": "ambiente", "port": settings.api_port}


@app.post(
    "/ambiente",
    response_model=AmbienteOut,
    status_code=status.HTTP_201_CREATED,
    responses={500: {"model": ErrorResponse}},
)
def criar_ambiente(payload: AmbienteCreate):
    insert_sql = """
        INSERT INTO ambiente (
            temperatura, umidade, co2, bairro, cidade, estado, pais,
            latitude, longitude, data, hora, origemdaleitura
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id, created_at
    """

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            insert_sql,
            (
                payload.temperatura,
                payload.umidade,
                payload.co2,
                payload.bairro,
                payload.cidade,
                payload.estado,
                payload.pais,
                payload.latitude,
                payload.longitude,
                payload.data,
                payload.hora,
                payload.origemdaleitura,
            ),
        )
        returned = cur.fetchone()
        conn.commit()

        if not returned:
            raise HTTPException(status_code=500, detail="Falha ao inserir registro")

        return {
            **payload.model_dump(),
            "id": returned[0],
            "created_at": returned[1],
        }


@app.delete(
    "/ambiente/{registro_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def excluir_ambiente(registro_id: int):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM ambiente WHERE id = ?", (registro_id,))
        found = cur.fetchone()

        if not found:
            raise HTTPException(status_code=404, detail="Registro não encontrado")

        cur.execute("DELETE FROM ambiente WHERE id = ?", (registro_id,))
        conn.commit()


@app.get(
    "/ambiente/{registro_id}",
    response_model=AmbienteOut,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def buscar_ambiente_por_id(registro_id: int):
    sql = """
        SELECT id, temperatura, umidade, co2, bairro, cidade, estado, pais,
               latitude, longitude, data, hora, origemdaleitura, created_at
        FROM ambiente
        WHERE id = ?
    """

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, (registro_id,))
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Registro não encontrado")

    return _row_to_dict(row)


@app.get(
    "/ambiente",
    response_model=List[AmbienteOut],
    responses={500: {"model": ErrorResponse}},
)
def listar_ambiente(
    cidade: Optional[str] = Query(default=None),
    data: Optional[date] = Query(default=None),
    bairro: Optional[str] = Query(default=None),
):
    base_sql = """
        SELECT id, temperatura, umidade, co2, bairro, cidade, estado, pais,
               latitude, longitude, data, hora, origemdaleitura, created_at
        FROM ambiente
    """

    conditions = []
    params: List[Any] = []

    if cidade:
        conditions.append("cidade = ?")
        params.append(cidade)

    if data:
        conditions.append("data = ?")
        params.append(data)

    if bairro:
        conditions.append("bairro = ?")
        params.append(bairro)

    if conditions:
        base_sql += " WHERE " + " AND ".join(conditions)

    base_sql += " ORDER BY id DESC"

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(base_sql, tuple(params))
        rows = cur.fetchall()

    return [_row_to_dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Rota POST /sensores  –  recebe dados do ESP32
# ---------------------------------------------------------------------------

@app.post(
    "/sensores",
    response_model=SensorOut,
    status_code=status.HTTP_201_CREATED,
    responses={
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def criar_sensor(
    payload: SensorCreate,
    _api_key: str = Depends(verify_api_key),
):
    """
    Recebe leituras de sensores (temperatura, humidade, CO2) enviadas
    pelo ESP32 e insere na tabela SENSORES do Firebird.
    Requer header ``X-API-Key`` válido.
    """
    insert_sql = """
        INSERT INTO SENSORES (
            CIDADE, ESTADO, PAIS, LATITUDE, LONGITUDE,
            TEMPERATURA, HUMIDADE, CO2, DEVICE_ID
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    with get_connection() as conn:
        cur = conn.cursor()

        # Verifica ultima leitura do mesmo device
        sql_ultima = (
            "SELECT FIRST 1 TEMPERATURA, HUMIDADE, CO2 "
            "FROM SENSORES WHERE DEVICE_ID = ? ORDER BY ID DESC"
        )
        cur.execute(sql_ultima, (payload.device_id,))
        ultima = cur.fetchone()

        # Se valores iguais, nao grava
        if ultima:
            temp_igual = abs(float(ultima[0]) - payload.temperatura) < 0.2
            umid_igual = abs(float(ultima[1]) - payload.humidade) < 1.0
            co2_igual  = abs(int(ultima[2]) - payload.co2) < 10
            if temp_igual and umid_igual and co2_igual:
                return SensorOut(
                    device_id=payload.device_id,
                    message="Dados iguais a ultima leitura - nao gravado."
                )

        cur.execute(
            insert_sql,
            (
                payload.cidade,
                payload.estado,
                payload.pais,
                payload.latitude,
                payload.longitude,
                payload.temperatura,
                payload.humidade,
                payload.co2,
                payload.device_id,
            ),
        )
        conn.commit()

    return SensorOut(device_id=payload.device_id)

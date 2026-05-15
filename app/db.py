"""
Módulo de conexão com o banco de dados Firebird.
Utiliza variáveis de ambiente definidas no arquivo .env
"""

from contextlib import contextmanager

import fdb
from fastapi import HTTPException, status

from config import get_settings


def _make_dsn() -> str:
    """Monta a string DSN para conexão com o Firebird."""
    settings = get_settings()
    return f"{settings.firebird_host}/{settings.firebird_port}:{settings.firebird_database}"


@contextmanager
def get_connection():
    """
    Context manager que fornece uma conexão com o Firebird.
    Fecha a conexão automaticamente ao sair do bloco.
    """
    settings = get_settings()
    conn = None
    try:
        conn = fdb.connect(
            dsn=_make_dsn(),
            user=settings.firebird_user,
            password=settings.firebird_password,
            charset=settings.firebird_charset,
        )
        yield conn
    except fdb.DatabaseError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao conectar/operar no Firebird: {exc}",
        ) from exc
    finally:
        if conn:
            conn.close()

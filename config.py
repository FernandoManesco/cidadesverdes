"""Configurações compartilhadas carregadas por variáveis de ambiente."""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    firebird_host: str = os.getenv("FIREBIRD_HOST", "127.0.0.1")
    firebird_database: str = os.getenv(
        "FIREBIRD_DATABASE", "/home/deploy/var/databases/cidadesverdes.fdb"
    )
    firebird_user: str = os.getenv("FIREBIRD_USER", "SYSDBA")
    firebird_password: str = os.getenv("FIREBIRD_PASSWORD", "masterkey")
    firebird_charset: str = os.getenv("FIREBIRD_CHARSET", "UTF8")
    firebird_port: int = int(os.getenv("FIREBIRD_PORT", "3050"))

    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8101"))

    cors_allow_origins: str = os.getenv("CORS_ALLOW_ORIGINS", "*")

    api_key: str = os.getenv("API_KEY", "")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

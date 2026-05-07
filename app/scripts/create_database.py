#!/usr/bin/env python3
"""Cria o banco Firebird e a tabela ambiente."""

from pathlib import Path
import sys

import fdb

# Permite importar o config.py da raiz do projeto
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import get_settings  # noqa: E402


def make_dsn(host: str, port: int, database: str) -> str:
    return f"{host}/{port}:{database}"


def table_exists(conn: fdb.Connection, table_name: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1
        FROM rdb$relations
        WHERE TRIM(rdb$relation_name) = ?
        """,
        (table_name.upper(),),
    )
    return cur.fetchone() is not None


def sequence_exists(conn: fdb.Connection, sequence_name: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1
        FROM rdb$generators
        WHERE TRIM(rdb$generator_name) = ?
        """,
        (sequence_name.upper(),),
    )
    return cur.fetchone() is not None


def trigger_exists(conn: fdb.Connection, trigger_name: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1
        FROM rdb$triggers
        WHERE TRIM(rdb$trigger_name) = ?
        """,
        (trigger_name.upper(),),
    )
    return cur.fetchone() is not None


def create_schema(conn: fdb.Connection):
    cur = conn.cursor()

    if not table_exists(conn, "ambiente"):
        cur.execute(
            """
            CREATE TABLE ambiente (
                id INTEGER NOT NULL,
                temperatura FLOAT,
                umidade FLOAT,
                co2 FLOAT,
                bairro VARCHAR(120),
                cidade VARCHAR(120),
                estado VARCHAR(120),
                pais VARCHAR(120),
                latitude VARCHAR(30),
                longitude VARCHAR(30),
                data DATE,
                hora TIME,
                origemdaleitura VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT pk_ambiente PRIMARY KEY (id)
            )
            """
        )
        print("Tabela ambiente criada.")
    else:
        print("Tabela ambiente já existe.")

    if not sequence_exists(conn, "ambiente_seq"):
        cur.execute("CREATE SEQUENCE ambiente_seq")
        print("Sequence ambiente_seq criada.")
    else:
        print("Sequence ambiente_seq já existe.")

    if not trigger_exists(conn, "bi_ambiente"):
        cur.execute(
            """
            CREATE TRIGGER bi_ambiente FOR ambiente
            ACTIVE BEFORE INSERT POSITION 0
            AS
            BEGIN
                IF (NEW.id IS NULL) THEN
                    NEW.id = NEXT VALUE FOR ambiente_seq;
                IF (NEW.created_at IS NULL) THEN
                    NEW.created_at = CURRENT_TIMESTAMP;
            END
            """
        )
        print("Trigger bi_ambiente criada.")
    else:
        print("Trigger bi_ambiente já existe.")

    conn.commit()


def main():
    settings = get_settings()
    db_path = Path(settings.firebird_database)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    dsn = make_dsn(settings.firebird_host, settings.firebird_port, settings.firebird_database)

    if not db_path.exists():
        create_stmt = (
            f"CREATE DATABASE '{dsn}' "
            f"USER '{settings.firebird_user}' PASSWORD '{settings.firebird_password}' "
            f"DEFAULT CHARACTER SET {settings.firebird_charset}"
        )
        fdb.create_database(create_stmt)
        print(f"Banco criado em: {settings.firebird_database}")
    else:
        print(f"Banco já existe em: {settings.firebird_database}")

    conn = fdb.connect(
        dsn=dsn,
        user=settings.firebird_user,
        password=settings.firebird_password,
        charset=settings.firebird_charset,
    )
    try:
        create_schema(conn)
    finally:
        conn.close()

    print("Inicialização concluída com sucesso.")


if __name__ == "__main__":
    main()

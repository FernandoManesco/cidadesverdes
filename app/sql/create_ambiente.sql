-- Tabela ambiente para Firebird 3.0
-- Observação: uso de sequence + trigger para auto-incremento compatível

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
);

CREATE SEQUENCE ambiente_seq;

SET TERM ^ ;
CREATE TRIGGER bi_ambiente FOR ambiente
ACTIVE BEFORE INSERT POSITION 0
AS
BEGIN
    IF (NEW.id IS NULL) THEN
        NEW.id = NEXT VALUE FOR ambiente_seq;
    IF (NEW.created_at IS NULL) THEN
        NEW.created_at = CURRENT_TIMESTAMP;
END^
SET TERM ; ^

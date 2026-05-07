#!/usr/bin/env bash
set -euo pipefail

EXPECTED_VERSION="3.0.13"
DB_DIR="/home/deploy/var/databases"

echo "[1/4] Verificando instalação do Firebird..."
INSTALLED_VERSION=""
if dpkg -s firebird3.0-server >/dev/null 2>&1; then
  INSTALLED_VERSION="$(dpkg-query -W -f='${Version}' firebird3.0-server | cut -d- -f1 || true)"
fi

if [[ "$INSTALLED_VERSION" == "$EXPECTED_VERSION" ]]; then
  echo "Firebird ${EXPECTED_VERSION} já está instalado."
else
  echo "[2/4] Instalando/atualizando Firebird 3.0..."
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y firebird3.0-server firebird3.0-utils

  INSTALLED_VERSION="$(dpkg-query -W -f='${Version}' firebird3.0-server | cut -d- -f1 || true)"
  if [[ "$INSTALLED_VERSION" != "$EXPECTED_VERSION" ]]; then
    echo "AVISO: versão instalada é ${INSTALLED_VERSION}."
    echo "O repositório do sistema pode não disponibilizar exatamente ${EXPECTED_VERSION}."
  fi
fi

echo "[3/4] Habilitando e iniciando serviço Firebird..."
sudo systemctl enable firebird3.0 || true
sudo systemctl restart firebird3.0
sudo systemctl --no-pager --full status firebird3.0 | head -n 20 || true

echo "[4/4] Criando diretório padrão do banco: ${DB_DIR}"
sudo mkdir -p "${DB_DIR}"
sudo chown -R firebird:firebird "${DB_DIR}" || true
sudo chmod 775 "${DB_DIR}" || true

echo "Concluído."
echo "Para criar o banco/tabela execute:"
echo "  python3 scripts/create_database.py"

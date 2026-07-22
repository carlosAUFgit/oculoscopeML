#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "execute com sudo: sudo bash install.sh"
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_USER="${SUDO_USER:-pi}"

echo "==> [1/5] instalando pacotes do sistema (apt)..."
apt-get update
apt-get install -y python3-pip python3-opencv python3-picamera2 python3-lgpio

echo "==> [2/5] instalando pacotes python (pip)..."
pip3 install --break-system-packages gpiozero flask
pip3 install --break-system-packages tflite-runtime \
    || pip3 install --break-system-packages ai-edge-litert

echo "==> [3/5] verificando o modelo..."
if [[ ! -f "${REPO_DIR}/model.tflite" ]]; then
    echo "erro: model.tflite nao encontrado em ${REPO_DIR}"
    exit 1
fi

echo "==> [4/5] gerando o servico systemd (usuario: ${RUN_USER}, pasta: ${REPO_DIR})..."
cat > /etc/systemd/system/oculoscope.service <<EOF
[Unit]
Description=Oculoscope capture + classify + server
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/bin/python3 ${REPO_DIR}/main.py
WorkingDirectory=${REPO_DIR}
Restart=always
RestartSec=3
User=${RUN_USER}

[Install]
WantedBy=multi-user.target
EOF

echo "==> [5/5] habilitando e iniciando o servico..."
systemctl daemon-reload
systemctl enable --now oculoscope.service

IP="$(hostname -I | awk '{print $1}')"
echo
echo "pronto! o oculoscope ja esta rodando e vai iniciar sozinho a cada boot."
echo
echo "acesse pelo celular (na mesma rede wi-fi):"
echo "  http://$(hostname).local:8000/"
echo "  http://${IP}:8000/"
echo
echo "acompanhe o log com: journalctl -u oculoscope -f"

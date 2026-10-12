#!/usr/bin/env bash
# Instala o VALTEXA ROVER RUNTIME na Jetson Orin NX (JetPack 6, Ubuntu 22.04) do Rover V3.
#
# Uso, como root, na raiz do repositório VALTEXA-ROVER copiado para a Jetson:
#
#   sudo deploy/jetson/instalar.sh --ort-wheel onnxruntime_gpu-<versão>-cp310-cp310-linux_aarch64.whl
#   sudo deploy/jetson/instalar.sh --ort-indice <índice pip da NVIDIA para o JetPack 6>
#   sudo deploy/jetson/instalar.sh --ort-cpu        # só CPU (PyPI): roda, mais devagar
#
# O ONNX Runtime com CUDA/TensorRT para Jetson não está no PyPI: vem da NVIDIA (Jetson AI
# Lab / Jetson Zoo), para o Python 3.10 do JetPack 6. Por isso o pacote é instalado com
# --no-deps e as dependências uma a uma (DEPENDENCIAS abaixo; o teste
# tests/test_jetson_deploy.py confere que batem com o pyproject.toml): senão o pip puxaria
# o onnxruntime de CPU do PyPI por cima do da NVIDIA.
#
# Idempotente: rodar de novo atualiza o pacote e não sobrescreve configuração existente
# (rover.env, rover.json, hardware.json) nem a identidade do robô.
set -euo pipefail

DEPENDENCIAS=("numpy>=1.26" "pillow>=10.0" "cryptography>=42.0")
DEPENDENCIAS_ROVER_V3=("depthai>=2.24,<3" "pyserial>=3.5" "python-can>=4.3" "opencv-python-headless>=4.8")

VENV=/opt/valtexa/venv
DADOS=/var/lib/valtexa-rover
CONF=/etc/valtexa
ORT_WHEEL=""
ORT_INDICE=""
ORT_CPU=0
SEM_APT=0

uso() { sed -n '2,20p' "$0"; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ort-wheel) ORT_WHEEL="$2"; shift 2 ;;
    --ort-indice) ORT_INDICE="$2"; shift 2 ;;
    --ort-cpu) ORT_CPU=1; shift ;;
    --sem-apt) SEM_APT=1; shift ;;   # imagem já com python3-venv e can-utils (sem internet)
    -h|--help) uso 0 ;;
    *) echo "opção desconhecida: $1" >&2; uso 2 ;;
  esac
done

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
[[ -f "$REPO/pyproject.toml" ]] || { echo "rode a partir do repositório VALTEXA-ROVER" >&2; exit 2; }
[[ $EUID -eq 0 ]] || { echo "rode como root (sudo)" >&2; exit 2; }
if [[ -z "$ORT_WHEEL$ORT_INDICE" && $ORT_CPU -eq 0 ]]; then
  echo "escolha o ONNX Runtime: --ort-wheel, --ort-indice ou --ort-cpu" >&2; exit 2
fi
if [[ -f /etc/nv_tegra_release ]]; then
  echo "L4T: $(head -1 /etc/nv_tegra_release)"
else
  echo "AVISO: /etc/nv_tegra_release não existe — isto não parece uma Jetson." >&2
fi
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || { echo "precisa de Python 3.10 ou mais novo" >&2; exit 2; }

echo "== pacotes do sistema"
if [[ $SEM_APT -eq 0 ]]; then
  apt-get update -qq
  apt-get install -y -qq python3-venv python3-pip can-utils
fi

echo "== usuário e pastas"
id valtexa >/dev/null 2>&1 || useradd --system --user-group --home-dir "$DADOS" \
  --shell /usr/sbin/nologin valtexa
install -d -o valtexa -g valtexa -m 0750 "$DADOS" "$DADOS/ota" "$DADOS/ota/models" "$DADOS/trt-cache"
install -d -o valtexa -g valtexa -m 0750 "$CONF"

echo "== ambiente Python em $VENV"
[[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip setuptools wheel
if [[ -n "$ORT_WHEEL" ]]; then
  "$VENV/bin/pip" install -q "$ORT_WHEEL"
elif [[ -n "$ORT_INDICE" ]]; then
  "$VENV/bin/pip" install -q --extra-index-url "$ORT_INDICE" onnxruntime-gpu
else
  "$VENV/bin/pip" install -q onnxruntime
fi
"$VENV/bin/pip" install -q "${DEPENDENCIAS[@]}" "${DEPENDENCIAS_ROVER_V3[@]}"
# Sem -e: o PYTHONPATH do slot do OTA precisa vencer o site-packages.
"$VENV/bin/pip" install -q --no-deps --force-reinstall "$REPO"
"$VENV/bin/python" - <<'PY'
import onnxruntime as ort
p = ort.get_available_providers()
print("ONNX Runtime", ort.__version__, p)
if "CUDAExecutionProvider" not in p and "TensorrtExecutionProvider" not in p:
    print("AVISO: sem CUDA/TensorRT — a Edge AI vai rodar na CPU.")
PY

echo "== configuração (não sobrescreve a existente)"
copiar() {  # origem destino modo
  if [[ -e "$2" ]]; then echo "   mantido: $2"; else install -o valtexa -g valtexa -m "$3" "$1" "$2"; echo "   novo:    $2"; fi
}
copiar "$REPO/deploy/jetson/rover.env" "$CONF/rover.env" 0640
copiar "$REPO/deploy/rover.json.exemplo" "$CONF/rover.json" 0640
copiar "$REPO/deploy/hardware.rover-v3.json.exemplo" "$CONF/hardware.json" 0640
"$VENV/bin/valtexa-rover" validar-config "$CONF/rover.json"

echo "== systemd, CAN e GNSS"
install -m 0644 "$REPO/deploy/valtexa-rover.service" /etc/systemd/system/valtexa-rover.service
install -d /etc/systemd/system/valtexa-rover.service.d
install -m 0644 "$REPO/deploy/jetson/valtexa-rover.service.d/rover-v3.conf" \
  /etc/systemd/system/valtexa-rover.service.d/rover-v3.conf
install -m 0644 "$REPO/deploy/jetson/80-can0.network" /etc/systemd/network/80-can0.network
install -m 0644 "$REPO/deploy/jetson/99-valtexa-gnss.rules" /etc/udev/rules.d/99-valtexa-gnss.rules
echo mttcan > /etc/modules-load.d/valtexa-can.conf
modprobe mttcan 2>/dev/null || echo "AVISO: módulo mttcan não carregou (confira o pinmux do CAN)" >&2
udevadm control --reload-rules && udevadm trigger --subsystem-match=tty || true
systemctl enable --now systemd-networkd >/dev/null
systemctl daemon-reload

cat <<FIM

Instalado. Antes de ligar o serviço:

  1. Ajuste $CONF/hardware.json (IPs e alturas das OAK-D, porta do GNSS, IDs da Xsens)
     e a URL da plataforma em $CONF/rover.env.
  2. Confira o hardware peça por peça (não precisa de identidade nem de rede):
       sudo -u valtexa VALTEXA_HARDWARE=$CONF/hardware.json \
         $VENV/bin/valtexa-rover diagnostico --dados $DADOS --salvar /tmp/diagnostico
  3. Provisione (uma vez; o token vem da plataforma, pela entrada padrão):
       sudo -u valtexa $VENV/bin/valtexa-rover provisionar --url <URL> \
         --serial <número de série> --identidade $DADOS/identity.json
  4. Prepare cada modelo (.onnx + .labels.json + .edge.json calibrado lado a lado):
       sudo -u valtexa $VENV/bin/valtexa-rover preparar-modelo banana_foliar.onnx --dados $DADOS
     e designe na plataforma o sha256 impresso para o talhão.
  5. systemctl enable --now valtexa-rover && journalctl -u valtexa-rover -f
FIM

"""todas as constantes ajustáveis do oculoscope. nenhum número mágico em outros arquivos.

"left"/"right" em todo o código significam esquerda/direita da imagem (ou seja,
a esquerda da imagem geralmente é o olho direito do paciente); o mapeamento
clínico é responsabilidade do operador.
"""
import os

# --- seleção de modo ---
# simulation=1 (variável de ambiente) seleciona o backend simulado. padrão: real.
SIMULATION: bool = os.environ.get("SIMULATION", "0") == "1"

# --- pinos gpio (numeração bcm) ---
BUTTON_PIN = 17                    # botão momentâneo -> gnd, pull-up interno
COAXIAL_LED_PIN = 22               # anel no eixo óptico (reflexo vermelho)
SURROUND_LED_PIN = 23              # anel difuso (sem reflexo)
FIXATION_LED_PINS = [24, 25, 26]   # alvos verdes "olhe aqui", múltiplas direções
CAPTURE_FIXATION_PIN = 25          # alvo verde central: pisca durante toda a
                                   # captura para travar o olhar do paciente
FIXATION_BLINK_INTERVAL = 0.25     # segundos aceso / segundos apagado

# --- câmera / recorte ---
FRAME_SIZE = (2592, 1944)          # resolução máxima do ov5647 (largura, altura)

# calibração aproximada (operador, 2026-07-21): a câmera fica centralizada
# entre os olhos, então o quadro é dividido na linha vertical central - cada
# metade contém um olho - com um pequeno corte em cada borda externa. altura
# completa. coordenadas (x0, y0, x1, y1) em pixels; mesmas rois para os dois
# modos de iluminação. refinar na montagem.
_HALF_W = FRAME_SIZE[0] // 2
_EDGE_TRIM = FRAME_SIZE[0] // 10   # corte horizontal "pequeno"
EYE_ROI_LEFT = (_EDGE_TRIM, 0, _HALF_W, FRAME_SIZE[1])
EYE_ROI_RIGHT = (_HALF_W, 0, FRAME_SIZE[0] - _EDGE_TRIM, FRAME_SIZE[1])

BURST_N = 5                        # fotos por captura; mantém a mais nítida
NN_INPUT = (160, 160)              # entrada do modelo (largura, altura)

# ganhos bgr por canal (leucocoria é uma decisão de cor). o sensor puxa
# levemente para o verde com esta iluminação, então atenua g e sobe b/r um
# pouco. valores aproximados (operador, 2026-07-21) - travar com cartão de
# referência na montagem.
WB_REFERENCE_GAINS = (1.05, 0.90, 1.05)

# as capturas saem escuras nesses níveis de led: ganho de exposição por
# software aplicado em capture.post_process junto com o ajuste de branco.
BRIGHTNESS_GAIN = 1.30

# --- limiares de decisão ---
CATARACT_THRESHOLD = 0.50          # equilibrado (não é emergência)
RETINOBLASTOMA_THRESHOLD = 0.20    # baixo de propósito -> encaminhar demais, não perder nada
QUALITY_THRESHOLD = 0.60           # abaixo disso -> recorte do olho é inavaliável

# --- estágio de funcionalidades ---
ENABLE_RB = False                  # v1: apenas catarata; true habilita a cabeça de rb

# --- caminhos / servidor ---
MODEL_PATH = "model.tflite"        # um único modelo float16: backbone compartilhado, 3 cabeças
TEST_IMAGES_DIR = "test_images"    # fonte da câmera simulada (fotos com os dois olhos)
SERVER_PORT = 8000
JPEG_QUALITY = 90

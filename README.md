# Oculoscope

Dispositivo portátil de triagem ocular construído sobre um Raspberry Pi 5 e uma
câmera OV5647. Funciona sem monitor e sem teclado: o operador aponta o
dispositivo para o rosto do paciente, aperta um único botão físico e, alguns
segundos depois, uma imagem anotada aparece em qualquer celular ou computador
conectado à mesma rede Wi-Fi. Cada captura fotografa os dois olhos de uma vez
e analisa cada um separadamente, triando catarata a partir de uma foto com
iluminação difusa.

> **Este dispositivo não produz diagnóstico.** Ele produz um sinal de triagem
> cuja finalidade é motivar um exame de acompanhamento com um profissional
> qualificado. Um resultado "Normal" não é atestado de saúde, e um resultado
> positivo não é confirmação de doença.

---

## 1. Como o código funciona

O projeto é um único processo Python de longa duração, dividido em 7 módulos
no mesmo diretório:

| módulo | responsabilidade |
|---|---|
| `config.py` | todas as constantes ajustáveis: pinos GPIO, recortes por olho (ROIs), limiares de decisão, ganhos de cor/brilho, porta do servidor. nenhum outro arquivo tem número mágico. os comentários estão em português. |
| `hardware.py` | camada de abstração de hardware. `get_backend()` devolve um `Backend` com câmera, botão, LEDs e carregador de modelo. com `SIMULATION=1` tudo é simulado (câmera lê fotos do disco, botão é o Enter/HTTP, LEDs vão para o log); no Pi real usa `picamera2`, `gpiozero` e `tflite-runtime`. o modelo é carregado UMA vez, na inicialização, e o carregador confere se o arquivo tem exatamente 3 saídas — modelo errado falha alto na hora, não silenciosamente depois. |
| `capture.py` | a sequência de captura. o LED verde central de fixação começa a piscar (prende o olhar do paciente) e só para quando a captura termina, mesmo se ela falhar. o anel difuso acende, a câmera tira uma rajada de 5 fotos, e fica a mais nítida (maior variância do laplaciano). a foto escolhida passa por correção de cor (o sensor puxa para o verde) e ganho de brilho, e é recortada em dois retângulos fixos, um por olho — não há detecção de rosto nem de olho: a geometria rígida do aparelho (estilo óculos de VR) garante a posição. |
| `classify.py` | redimensiona cada recorte para 160×160 e faz UMA passada no modelo, que devolve três probabilidades: qualidade, catarata e leucocoria. qualidade abaixo do limiar marca o olho como "UNGRADABLE" (refazer); catarata acima do limiar vira o rótulo "Cataract". |
| `annotate.py` | desenha na imagem completa os retângulos de cada olho e os rótulos com as probabilidades. |
| `server.py` | servidor Flask. `GET /` devolve uma página escura que se atualiza sozinha a cada 1,5 s; `GET /latest` devolve o último JPEG anotado (ou 503 se ainda não houve captura). |
| `main.py` | orquestra tudo: sobe a thread de captura (que fica parada esperando o botão) e o servidor na thread principal. o único estado compartilhado é o último JPEG, em memória, protegido por um lock — **nada é gravado em disco durante a operação**, nenhuma foto de paciente persiste no aparelho. |

O fluxo completo de um aperto de botão:

1. botão (GPIO 17) → a thread de captura acorda;
2. LED de fixação central pisca; anel difuso acende; rajada de 5 fotos;
3. fica a foto mais nítida → correção de cor → ganho de brilho → recorte dos
   dois olhos;
4. modelo roda uma vez por olho → (qualidade, catarata, leucocoria);
5. imagem anotada é comprimida em JPEG e substitui a anterior na memória;
6. qualquer navegador na rede vê o resultado novo em até 1,5 s.

Decisões de projeto que valem conhecer:

- **sensibilidade primeiro**: os limiares preferem encaminhar demais a deixar
  passar um caso.
- **v1 = só catarata**: o pipeline de leucocoria/retinoblastoma (iluminação
  coaxial, `ENABLE_RB` em `config.py`) está inteiro no código, mas desligado
  até existir dado de treino suficiente para confiar nele.
- **2 GB de RAM**: um único modelo `.tflite` float16 (~1,9 MB) com backbone
  compartilhado e 3 cabeças — nunca TensorFlow completo no Pi.
- se uma captura falhar (quadro ruim, câmera engasgou), o erro vai para o log
  e a thread volta a esperar o botão — o processo nunca morre por causa de um
  quadro ruim, e o systemd reinicia o serviço se algo pior acontecer.

---

## 2. Setup no Raspberry Pi

Ao final desta seção o dispositivo liga sozinho, captura no botão e serve as
imagens na rede — sem monitor, sem teclado, sem passo manual depois do boot.

### 2.1 O que você precisa

- Raspberry Pi 5 (2 GB ou mais) + fonte oficial;
- cartão microSD (16 GB+);
- câmera OV5647 no conector CSI (atenção: o Pi 5 usa conector de 22 pinos —
  módulos antigos precisam do cabo adaptador 15→22 pinos);
- botão momentâneo e LEDs ligados nos pinos abaixo;
- uma rede Wi-Fi — **a mesma** em que o celular vai ficar.

Pinagem esperada pelo código (`config.py`, numeração BCM):

| função | BCM | pino físico | ligação |
|---|---|---|---|
| botão de captura | 17 | 11 | entre o pino e GND (pull-up interno, sem resistor externo) |
| anel coaxial (reflexo vermelho, v2) | 22 | 15 | via driver/transistor |
| anel difuso (captura de catarata) | 23 | 16 | via driver/transistor |
| LED de fixação | 24 | 18 | alvo "olhe aqui" |
| LED de fixação **central** | 25 | 22 | **pisca durante toda a captura** |
| LED de fixação | 26 | 37 | alvo "olhe aqui" |

(pinos de GND disponíveis no conector: 6, 9, 14, 20, 25, 30, 34, 39)

Se a sua montagem usar outros pinos, ajuste as constantes correspondentes em
`config.py` — é o único lugar onde elas existem.

### 2.2 Gravar o cartão SD

Use o **Raspberry Pi Imager** (raspberrypi.com/software):

1. *Choose OS* → **Raspberry Pi OS Lite (64-bit)** — sem desktop; o aparelho
   não tem tela.
2. *Choose Storage* → o cartão SD.
3. Clique em **Next → Edit Settings** (ou `Ctrl+Shift+X`) e preencha — este
   passo é o que torna o resto do setup 100 % remoto:
   - **hostname**: `oculoscope` (é isso que faz `http://oculoscope.local`
     funcionar depois);
   - **usuário e senha**: anote os dois;
   - **Wi-Fi**: SSID e senha da **mesma rede que o seu celular usa**, país
     `BR`;
   - aba *Services* → **habilite SSH** (com senha).
4. Grave, coloque o cartão no Pi (com a câmera já conectada — sempre com o
   Pi desligado) e ligue a fonte.

### 2.3 Acessar por SSH

Espere ~1 minuto após ligar e, do seu computador (na mesma rede):

```bash
ssh SEU_USUARIO@oculoscope.local
```

Se `oculoscope.local` não resolver (algumas redes bloqueiam mDNS), descubra o
IP do Pi na lista de dispositivos do seu roteador e use
`ssh SEU_USUARIO@IP_DO_PI`.

Antes de instalar, confira a câmera:

```bash
rpicam-hello --list-cameras
```

Deve listar o sensor `ov5647`. Se não listar, desligue o Pi e confira o cabo
CSI (lado dos contatos, trava bem fechada, conector CAM correto).

### 2.4 Instalar (2 comandos)

O repositório já vem pronto para ser clonado direto no Pi — o `install.sh`
faz todo o resto (dependências, serviço, boot automático):

```bash
git clone https://github.com/carlosAUFgit/oculoscopeML.git ~/oculoscope
sudo bash ~/oculoscope/install.sh
```

O que o script faz, na ordem:

1. **apt**: `python3-pip`, `python3-opencv`, `python3-picamera2`,
   `python3-lgpio` (backend de GPIO do Pi 5) — como pacotes do sistema,
   porque compilar OpenCV/câmera via pip num Pi é lento e frágil;
2. **pip** (`--break-system-packages`, exigido pelo Raspberry Pi OS atual):
   `gpiozero`, `flask` e `tflite-runtime` — se não houver wheel de
   `tflite-runtime` para a sua versão de Python, ele instala o sucessor
   `ai-edge-litert` automaticamente (o código aceita os dois);
3. confere que `model.tflite` está presente (já vem no clone);
4. gera `/etc/systemd/system/oculoscope.service` apontando para a pasta onde
   você clonou e para o seu usuário — por isso funciona em qualquer pasta e
   com qualquer nome de usuário;
5. `systemctl enable --now` — o serviço sobe na hora e passa a subir sozinho
   em todo boot, reiniciando em 3 s se travar.

Ao final o script imprime os dois endereços de acesso.

### 2.5 Conferir se está tudo rodando

```bash
sudo systemctl status oculoscope    # deve mostrar "active (running)"
journalctl -u oculoscope -f         # log ao vivo
```

No log você deve ver a linha de inicialização (`oculoscope up
(simulation=False) - press the button`). Aperte o botão físico: aparecem as
linhas de LED e, ao final, `capture complete: status=...`. Esse `journalctl`
é a principal ferramenta de depuração do aparelho — todo erro de captura cai
ali.

### 2.6 Problemas comuns

| sintoma | causa provável / solução |
|---|---|
| serviço reiniciando em loop | `journalctl -u oculoscope -e` mostra o erro real. os três clássicos: câmera não detectada (cabo CSI), `model.tflite` ausente, import de pacote pip que falhou |
| câmera não listada | cabo invertido/solto, conector errado; teste com `rpicam-hello --list-cameras` |
| LEDs não acendem | confira se ligou pelos números **BCM** (não pelo pino físico) e se `python3-lgpio` está instalado |
| botão não dispara | precisa fechar o contato entre BCM 17 e GND; teste apertando e olhando o `journalctl -f` |
| `pip3 install tflite-runtime` falhou no manual | use `pip3 install --break-system-packages ai-edge-litert` — o código tenta os dois imports |

---

## 3. Ver as imagens no celular (mesma rede)

O Pi não envia nada para a nuvem: ele serve as imagens **somente na rede
local**. A única condição é que **o celular esteja no mesmo Wi-Fi que o Pi**
— o mesmo SSID que você configurou no Imager.

**Passo a passo:**

1. No celular, desligue os dados móveis (só para o teste — evita que o
   navegador tente sair pela 4G/5G) e confirme que o Wi-Fi conectado é o
   mesmo do Pi.
2. Abra o navegador e acesse:

   ```
   http://oculoscope.local:8000/
   ```

   Em iPhone isso funciona nativamente. **Em Android, endereços `.local`
   muitas vezes não resolvem** — nesse caso use o IP direto:

   ```
   http://IP_DO_PI:8000/
   ```

   Para descobrir o IP: `hostname -I` numa sessão SSH, ou a lista de
   dispositivos do app do roteador (procure por "oculoscope"). O
   `install.sh` também imprime o IP ao terminar.
3. A página é escura e **se atualiza sozinha a cada 1,5 s** — não precisa
   recarregar. Antes da primeira captura ela mostra um aviso (o servidor
   responde 503 porque ainda não existe imagem); depois do primeiro aperto
   de botão, a foto anotada aparece em poucos segundos, com o retângulo e o
   rótulo de cada olho.
4. Para baixar/compartilhar o JPEG cru, acesse `http://IP_DO_PI:8000/latest`
   e salve a imagem.

**Dicas para uso no dia a dia:**

- **IP fixo**: reserve o IP do Pi no roteador (reserva DHCP) e salve o
  endereço como favorito/atalho na tela inicial do celular — vira um "app".
- **Rede de convidados não funciona**: redes guest normalmente isolam os
  aparelhos entre si (AP isolation). Celular e Pi precisam estar na rede
  principal, na mesma faixa.
- Vários celulares/computadores podem abrir a página ao mesmo tempo — todos
  veem a mesma última captura.
- Sem internet tudo continua funcionando: basta o roteador (ou um hotspot)
  criando a rede local entre o celular e o Pi.

---

## 4. Trocar o modelo

O dispositivo espera **um** arquivo `model.tflite` ao lado de `main.py`
(caminho em `config.MODEL_PATH`). É um modelo único multi-cabeça — backbone
compartilhado — o que mantém tudo dentro dos 2 GB do Pi.

Contrato exato do arquivo:

- **entrada**: `(1, 160, 160, 3)` float32, pixels crus 0–255;
- **saída**: exatamente 3 tensores sigmoides em `[0, 1]`, **nesta ordem**:
  `quality`, `cataract`, `leukocoria`. O carregador confere a contagem no
  startup e aborta se não bater;
- **formato**: float16 TFLite (INT8 foi testado e degradou demais as
  predições deste backbone; o carregador aceita ambos e decide sozinho pelo
  mapa de quantização do arquivo).

Para trocar: substitua o arquivo e reinicie —

```bash
sudo systemctl restart oculoscope
```

O modelo é carregado uma única vez no startup, nunca por captura. Quando
houver um modelo com a cabeça de leucocoria validada, habilite em
`config.py` com `ENABLE_RB = True` — a v1 mantém desligado de propósito.

---

## 5. Calibração (uma vez, no aparelho montado)

Os valores atuais em `config.py` são uma primeira aproximação medida pelo
operador; refine-os no aparelho físico antes de triagem real:

- **`EYE_ROI_LEFT` / `EYE_ROI_RIGHT`** — os retângulos fixos de recorte.
  Hoje: quadro dividido ao meio na vertical (câmera centralizada entre os
  olhos), ~10 % cortados de cada borda externa, altura completa. Para
  refinar: capture um quadro completo, abra num visualizador e leia as
  coordenadas de um retângulo em volta de cada olho com folga para o
  movimento normal da cabeça.
- **`WB_REFERENCE_GAINS`** — ganhos B/G/R, hoje `(1.05, 0.90, 1.05)` para
  compensar o tom esverdeado do sensor. Para travar de vez: fotografe um
  cartão cinza/branco neutro na distância de trabalho, meça a média de cada
  canal e ajuste os ganhos até os três canais ficarem iguais. Isso importa
  mais do que parece: leucocoria é uma decisão de **cor** (reflexo vermelho
  × branco), então qualquer desvio de cor vira viés na predição.
- **`BRIGHTNESS_GAIN`** — ganho de exposição por software (hoje `1.30`,
  porque as capturas saem escuras nesses níveis de LED); ajuste junto com os
  ganhos de cor.

Sem essa passada o pipeline funciona de ponta a ponta, mas recorte e cor são
só aproximados para a óptica da sua unidade.

---

## 6. Modo de simulação (desenvolvimento)

Para desenvolver sem Pi: `pip install -r requirements.txt`, coloque fotos
"dois olhos" numa pasta `test_images/` e rode `SIMULATION=1 python main.py`.
O botão vira o Enter no terminal (ou `POST /trigger`), a câmera cicla pelas
fotos da pasta e o modelo é simulado. A suíte de testes roda com
`python -m pytest`.

`test_images/` está no `.gitignore` de propósito e nunca deve ser commitada:
fotos de exemplo tendem a ser fotos reais de olhos — potencialmente de
pacientes — e não podem entrar permanentemente no histórico do git.

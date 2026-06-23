# Hand Tracker

Aplicação de captura de fotos por gestos usando MediaPipe e OpenCV.

## Como funciona

O app detecta suas mãos via webcam e permite interagir com fotos usando gestos:

### Capturar foto
- Levante **duas mãos** na frente da câmera
- Faça um losango/retângulo com os polegares e indicadores
- Na **mão esquerda**, abaixe os 3 dedos depois do indicador (meio, anelar e mindinho)
- Segure por **0.5 segundos** para capturar

### Mover fotos
- Use **pinça** (polegar + indicador juntos) em cima de uma foto
- Arraste para mover

### Zoom
- Use **duas pinças** (uma em cada mão) sobre uma foto
- Afasta ou junta as mãos para dar zoom

## Instalação

1. Execute `iniciar.bat` no Windows
2. As dependências serão instaladas automaticamente
3. A câmera será aberta automaticamente

### Dependências
- Python 3.8+
- OpenCV
- MediaPipe
- Pillow
- NumPy

## Controles

| Gestão | Ação |
|--------|------|
| Duas mãos + 3 dedos abaixados (esquerda) | Capturar foto |
| Pinça em cima de foto | Mover foto |
| Duas pinças sobre foto | Zoom in/out |

## Arquivos

- `hand_tracker.py` - Código principal
- `hand_landmarker.task` - Modelo MediaPipe (baixado automaticamente)
- `iniciar.bat` - Script de inicialização

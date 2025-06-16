import streamlit as st
import cv2
import numpy as np
import av
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import mediapipe as mp
import os

# Configuração da página
st.set_page_config(page_title="Filtros de Óculos", layout="wide")

# Verifica se a pasta de filtros existe
if not os.path.exists("filtros"):
    os.makedirs("filtros")
    st.warning("Pasta 'filtros' criada. Adicione seus filtros PNG nela.")

# Carregar filtros da pasta "filtros"
filter_files = []
if os.path.exists("filtros"):
    filter_files = sorted([
        f for f in os.listdir("filtros") 
        if f.startswith("filter") and f.endswith(".png")
    ])

# Verifica se há filtros disponíveis
if not filter_files:
    st.error("Nenhum filtro encontrado na pasta 'filtros'!")
    st.info("Adicione imagens PNG nomeadas como 'filter1.png', 'filter2.png', etc.")
    st.stop()

filters = [cv2.imread(os.path.join("filtros", f), cv2.IMREAD_UNCHANGED) for f in filter_files]
filter_names = [f.split('.')[0].capitalize() for f in filter_files]

# Sidebar com seleção de filtro
selected_filter_index = st.sidebar.selectbox(
    "Escolha o filtro:", range(len(filters)), format_func=lambda x: filter_names[x]
)

# Armazena seleção no estado
st.session_state["selected_filter"] = selected_filter_index

# Inicializa o MediaPipe com tratamento de erro
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
except Exception as e:
    st.error(f"Erro ao inicializar MediaPipe: {str(e)}")
    st.stop()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.filter_img = filters[selected_filter_index]
        self.selected_index = selected_filter_index

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        try:
            image = frame.to_ndarray(format="bgr24")
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Atualiza o filtro selecionado
            self.selected_index = st.session_state.get("selected_filter", self.selected_index)
            self.filter_img = filters[self.selected_index]
            
            results = face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    image = self.apply_filter(image, face_landmarks.landmark)
            
            return image
        except Exception as e:
            st.error(f"Erro no processamento do frame: {str(e)}")
            return np.zeros((480, 640, 3), dtype=np.uint8)  # Retorna frame preto em caso de erro

    def apply_filter(self, frame, landmarks):
        try:
            ih, iw, _ = frame.shape
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            x1 = int(left_eye.x * iw)
            y1 = int(left_eye.y * ih)
            x2 = int(right_eye.x * iw)
            y2 = int(right_eye.y * ih)

            # Centro dos olhos
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Tamanho do filtro baseado na distância entre os olhos
            w = int(1.6 * abs(x2 - x1))
            h = int(self.filter_img.shape[0] * (w / self.filter_img.shape[1]))

            filter_resized = cv2.resize(self.filter_img, (w, h))
            fh, fw, fc = filter_resized.shape

            x1 = cx - fw // 2
            y1 = cy - fh // 2

            # Aplica o filtro com transparência
            for c in range(3):
                for i in range(fh):
                    for j in range(fw):
                        if 0 <= x1 + j < iw and 0 <= y1 + i < ih:
                            alpha = filter_resized[i, j, 3] / 255.0
                            frame[y1 + i, x1 + j, c] = (
                                1 - alpha
                            ) * frame[y1 + i, x1 + j, c] + alpha * filter_resized[i, j, c]
            return frame
        except Exception as e:
            st.error(f"Erro ao aplicar filtro: {str(e)}")
            return frame

# Estilo para melhorar a exibição
st.markdown(
    """
    <style>
        [data-testid="stVideo"] video {
            width: 100% !important;
            border-radius: 10px;
        }
        .st-emotion-cache-1y4p8pa {
            padding: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Título da aplicação
st.title("Teste de Óculos Virtual")
st.caption("Selecione o oculos na barra lateral e posicione seu rosto na câmera")

# Iniciar webcam com tratamento de erro
try:
    webrtc_streamer(
        key="filtros-webcam",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "facingMode": "user"
            }, 
            "audio": False
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
        mode=WebRtcMode.SENDRECV,
        translations={
            "start": "Iniciar câmera",
            "stop": "Parar câmera",
            "select_device": "Selecionar dispositivo",
            "media_api_not_available": "API de mídia não disponível",
            "device_ask_permission": "Permissão para acessar a câmera",
            "device_not_available": "Dispositivo não disponível",
            "device_access_denied": "Acesso à câmera negado"
        }
    )
except Exception as e:
    st.error(f"Erro ao iniciar a câmera: {str(e)}")
    st.info("""
        Dicas para solucionar problemas:
        1. Verifique se você permitiu o acesso à câmera
        2. Atualize a página
        3. Tente em outro navegador (Chrome ou Edge funcionam melhor)
        4. Verifique se há outros aplicativos usando a câmera
    """)

# Adiciona informações de uso
st.sidebar.markdown("### Como usar:")
st.sidebar.info("""
1. Selecione um filtro na lista
2. Posicione seu rosto na câmera
3. Ajuste a posição até que o óculos fique alinhado
""")

# Verificação de requisitos
st.sidebar.markdown("### Requisitos:")
st.sidebar.success("""
- Navegador moderno (Chrome, Edge, Firefox)
- Conexão estável com a internet
- Webcam funcionando
""")

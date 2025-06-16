import streamlit as st
import cv2
import numpy as np
import av
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import os

# Configuração da página
st.set_page_config(page_title="Filtros de Óculos", layout="wide")

# Carregar filtros da pasta "filtros"
filter_files = sorted([
    f for f in os.listdir("filtros") if f.startswith("filter") and f.endswith(".png")
])
filters = [cv2.imread(os.path.join("filtros", f), cv2.IMREAD_UNCHANGED) for f in filter_files]
filter_names = [f.split('.')[0].capitalize() for f in filter_files]

# Sidebar com seleção de filtro (sem escala)
selected_filter_index = st.sidebar.selectbox(
    "Escolha o filtro:", range(len(filters)), format_func=lambda x: filter_names[x]
)

# Escala fixa em 1.0
scale = 1.0

# Armazena seleção no estado
st.session_state["selected_filter"] = selected_filter_index
# Removido: st.session_state["filter_scale"]

# Inicializa o MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.filter_img = filters[selected_filter_index]
        self.selected_index = selected_filter_index
        self.scale = scale

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        image = frame.to_ndarray(format="bgr24")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        # Atualiza o filtro em tempo real
        self.selected_index = st.session_state.get("selected_filter", self.selected_index)
        # Sempre escala 1.0
        self.scale = 1.0
        self.filter_img = filters[self.selected_index]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                image = self.apply_filter(image, face_landmarks.landmark)
        return image

    def apply_filter(self, frame, landmarks):
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

        # Novo tamanho do filtro fixo (escala 1.0)
        w = int(1.0 * 1.6 * abs(x2 - x1))
        h = int(self.filter_img.shape[0] * (w / self.filter_img.shape[1]))

        filter_resized = cv2.resize(self.filter_img, (w, h))
        fh, fw, fc = filter_resized.shape

        x1 = cx - fw // 2
        y1 = cy - fh // 2

        for c in range(3):
            for i in range(fh):
                for j in range(fw):
                    if 0 <= x1 + j < iw and 0 <= y1 + i < ih:
                        alpha = filter_resized[i, j, 3] / 255.0
                        frame[y1 + i, x1 + j, c] = (
                            1 - alpha
                        ) * frame[y1 + i, x1 + j, c] + alpha * filter_resized[i, j, c]
        return frame

# Estilo para forçar o vídeo a ocupar a largura da tela
st.markdown(
    """
    <style>
        [data-testid="stVideo"] video {
            width: 100% !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Iniciar webcam com o transformador
webrtc_streamer(
    key="filtros-webcam",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False}
)

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from mobile_sam import sam_model_registry, SamPredictor

st.set_page_config(page_title="Simulador IA - Modo Rápido", layout="wide")

@st.cache_resource
def load_sam():
    # Carga el modelo una sola vez
    return SamPredictor(sam_model_registry["vit_t"](checkpoint="mobile_sam.pt"))

predictor = load_sam()

# --- MEMORIA ---
if 'capas' not in st.session_state: st.session_state.capas = []
if 'img_base' not in st.session_state: st.session_state.img_base = None

st.title("🖌️ Simulador de Paredes (Versión Fluida)")

archivo = st.file_uploader("1. Sube tu foto", type=["jpg", "png", "jpeg"])

if archivo:
    img_p = Image.open(archivo).convert("RGB")
    st.session_state.img_base = np.array(img_p)
    # Solo procesamos la imagen una vez
    predictor.set_image(cv2.cvtColor(st.session_state.img_base, cv2.COLOR_RGB2BGR))

if st.session_state.img_base is not None:
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Configuración")
        color_p = st.color_picker("Color de pintura", "#4b5d52")
        h, w, _ = st.session_state.img_base.shape
        
        # SLIDERS: Ahora deberían responder al instante
        x = st.slider("Posición X", 0, w, w // 2)
        y = st.slider("Posición Y", 0, h, h // 2)
        
        st.write(f"Coordenadas: {x}, {y}")
        
        if st.button("🎨 PINTAR PARED"):
            with st.spinner("IA trabajando..."):
                masks, scores, _ = predictor.predict(np.array([[x, y]]), np.array([1]), multimask_output=True)
                mask = masks[np.argmax(scores)]
                st.session_state.capas.append({'mask': mask, 'color': color_p})
                st.rerun()

        if st.button("🗑️ Borrar todo"):
            st.session_state.capas = []
            st.rerun()

    with col1:
        # Creamos una copia para mostrar
        res = st.session_state.img_base.copy()
        
        # Aplicamos las capas guardadas
        for c in st.session_state.capas:
            # Efecto de pintura realista simple
            hex_c = c['color'].lstrip('#')
            rgb_c = np.array([int(hex_c[i:i+2], 16) for i in (0, 2, 4)])
            res[c['mask']] = (res[c['mask']] * 0.5 + rgb_c * 0.5).astype(np.uint8)
        
        # DIBUJAMOS EL PUNTERO (Amarillo neón para que resalte)
        cv2.circle(res, (x, y), 25, (255, 255, 0), -1) # Círculo amarillo
        cv2.circle(res, (x, y), 30, (0, 0, 0), 3)       # Borde negro
        
        st.image(res, use_container_width=True)
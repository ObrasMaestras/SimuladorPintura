import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import urllib.request

# DESCARGA AUTOMÁTICA DEL MODELO
if not os.path.exists("mobile_sam.pt"):
    with st.spinner("Descargando motor de IA... espera un momento."):
        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        urllib.request.urlretrieve(url, "mobile_sam.pt")

from mobile_sam import sam_model_registry, SamPredictor

# Cargar IA
@st.cache_resource
def load_sam():
    return SamPredictor(sam_model_registry["vit_t"](checkpoint="mobile_sam.pt"))

predictor = load_sam()

# Interfaz básica para que el punto se mueva SI O SI
st.title("Simulador de Pintura Profesional")
archivo = st.file_uploader("Sube la foto", type=["jpg", "png", "jpeg"])

if archivo:
    img = Image.open(archivo).convert("RGB")
    img_np = np.array(img)
    predictor.set_image(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    
    h, w, _ = img_np.shape
    x = st.slider("Mover horizontal", 0, w, w//2)
    y = st.slider("Mover vertical", 0, h, h//2)
    
    # Dibujar el punto
    preview = img_np.copy()
    cv2.circle(preview, (x, y), 25, (255, 255, 0), -1)
    st.image(preview, use_container_width=True)
    
    if st.button("PINTAR"):
        st.write("¡IA trabajando!")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import urllib.request

# 1. DESCARGA EL PESO (CEREBRO) DE LA IA
if not os.path.exists("mobile_sam.pt"):
    url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    urllib.request.urlretrieve(url, "mobile_sam.pt")

from mobile_sam import sam_model_registry, SamPredictor

@st.cache_resource
def load_ia():
    return SamPredictor(sam_model_registry["vit_t"](checkpoint="mobile_sam.pt"))

predictor = load_ia()

st.title("🖌️ Simulador Pro (Modo Web)")

archivo = st.file_uploader("Sube tu foto", type=["jpg", "png", "jpeg"])

if archivo:
    img = Image.open(archivo).convert("RGB")
    img_np = np.array(img)
    predictor.set_image(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    
    h, w, _ = img_np.shape
    x = st.slider("Mover Horizontal", 0, w, w//2)
    y = st.slider("Mover Vertical", 0, h, h//2)
    
    # Vista previa del punto
    res = img_np.copy()
    cv2.circle(res, (x, y), 25, (255, 255, 0), -1)
    st.image(res, use_container_width=True)
    
    if st.button("PINTAR PARED"):
        masks, scores, _ = predictor.predict(np.array([[x, y]]), np.array([1]), multimask_output=True)
        mask = masks[np.argmax(scores)]
        
        # Pintado rápido
        img_np[mask] = img_np[mask] * 0.5 + np.array([75, 93, 82]) * 0.5
        st.image(img_np, caption="Resultado", use_container_width=True)

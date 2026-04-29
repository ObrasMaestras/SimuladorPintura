import os
import subprocess
import sys


# Ejecutamos la instalación antes de que falle el código
install_requirements()

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_canvas import st_canvas
import urllib.request

# 2. CONFIGURACIÓN DE LA IA
if not os.path.exists("mobile_sam.pt"):
    url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    urllib.request.urlretrieve(url, "mobile_sam.pt")

from mobile_sam import sam_model_registry, SamPredictor

@st.cache_resource
def load_ia():
    return SamPredictor(sam_model_registry["vit_t"](checkpoint="mobile_sam.pt"))

predictor = load_ia()

# 3. INTERFAZ DEL SIMULADOR
st.title("🖌️ Simulador de Pintura Pro")
st.write("Sube una foto y haz clic en la pared que quieras pintar.")

archivo = st.file_uploader("Sube tu foto", type=["jpg", "png", "jpeg"])

if archivo:
    img = Image.open(archivo).convert("RGB")
    w, h = img.size
    canvas_width = 700
    ratio = canvas_width / w
    canvas_height = int(h * ratio)
    img_res = img.resize((canvas_width, canvas_height))

    # Lienzo para hacer clic
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        background_image=img_res,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="point",
        point_display_radius=6,
        key="canvas",
    )

    if canvas_result.json_data is not None:
        puntos = canvas_result.json_data["objects"]
        if len(puntos) > 0:
            last = puntos[-1]
            x, y = int(last["left"] / ratio), int(last["top"] / ratio)
            
            if st.button("🎨 PINTAR AHORA"):
                with st.spinner("Pintando..."):
                    predictor.set_image(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
                    masks, scores, _ = predictor.predict(np.array([[x, y]]), np.array([1]), multimask_output=True)
                    mask = masks[np.argmax(scores)]
                    
                    res_final = np.array(img)
                    # Color verde musgo (puedes cambiarlo aquí)
                    res_final[mask] = res_final[mask] * 0.5 + np.array([100, 120, 100]) * 0.5
                    st.image(res_final, caption="Resultado", use_container_width=True)


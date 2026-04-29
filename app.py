import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_canvas import st_canvas # La herramienta mágica
import os
import urllib.request

# Descarga de IA (Automática)
if not os.path.exists("mobile_sam.pt"):
    url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    urllib.request.urlretrieve(url, "mobile_sam.pt")

from mobile_sam import sam_model_registry, SamPredictor

@st.cache_resource
def load_ia():
    return SamPredictor(sam_model_registry["vit_t"](checkpoint="mobile_sam.pt"))

predictor = load_ia()

st.title("🖌️ Simulador Pro: Toca para Pintar")
st.write("Haz clic directamente en la pared que quieras cambiar de color.")

archivo = st.file_uploader("Sube tu foto", type=["jpg", "png", "jpeg"])

if archivo:
    img = Image.open(archivo).convert("RGB")
    w, h = img.size
    # Ajustar tamaño para que quepa en pantalla
    canvas_width = 800
    ratio = canvas_width / w
    canvas_height = int(h * ratio)
    img_resized = img.resize((canvas_width, canvas_height))

    # EL CANVAS: Aquí es donde haces clic
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=1,
        background_image=img_resized,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="point", # Solo puntos
        point_display_radius=5,
        key="canvas",
    )

    # Si el usuario hace clic:
    if canvas_result.json_data is not None:
        puntos = canvas_result.json_data["objects"]
        if len(puntos) > 0:
            last_point = puntos[-1]
            x = int(last_point["left"] / ratio)
            y = int(last_point["top"] / ratio)
            
            st.success(f"Pared seleccionada en coordenadas: {x}, {y}")
            
            if st.button("🎨 APLICAR PINTURA"):
                predictor.set_image(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
                masks, scores, _ = predictor.predict(np.array([[x, y]]), np.array([1]), multimask_output=True)
                mask = masks[np.argmax(scores)]
                
                # Pintado
                final = np.array(img)
                final[mask] = final[mask] * 0.5 + np.array([75, 93, 82]) * 0.5
                st.image(final, caption="Resultado Final", use_container_width=True)

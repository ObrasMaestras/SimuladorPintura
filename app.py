import streamlit as st
import numpy as np
from PIL import Image
import torch
import os
import requests
from streamlit_drawable_canvas import st_canvas

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Simulador de Pintura", layout="wide")

# 2. FUNCIONES DE INTELIGENCIA ARTIFICIAL
@st.cache_data
def descargar_modelo():
    modelo_path = "mobile_sam.pt"
    if not os.path.exists(modelo_path):
        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        with st.spinner("⏳ Descargando cerebro de la IA (esto solo pasa una vez)..."):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(modelo_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                st.error(f"❌ Error al descargar modelo: {e}")
                return None
    return modelo_path

@st.cache_resource
def cargar_predictor():
    try:
        from mobile_sam import sam_model_registry, SamPredictor
        path = descargar_modelo()
        if path:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            modelo = sam_model_registry["vit_t"](checkpoint=path)
            modelo.to(device=device)
            modelo.eval()
            return SamPredictor(modelo)
    except Exception as e:
        st.error(f"❌ Error en la IA: {e}")
    return None

# 3. INICIALIZACIÓN DE MEMORIA
if 'areas_pintadas' not in st.session_state:
    st.session_state.areas_pintadas = []
if 'imagen_base' not in st.session_state:
    st.session_state.imagen_base = None

# 4. INTERFAZ DE USUARIO
st.title("🎨 Simulador de Pintura Pro")
archivo = st.file_uploader("📸 Sube la foto de tu pared", type=["jpg", "jpeg", "png"])

if archivo:
    # Procesar imagen
    img_orig = Image.open(archivo).convert("RGB")
    
    # Redimensionar para que no sea pesada
    max_size = 700
    ratio = max_size / max(img_orig.size)
    nuevo_tam = (int(img_orig.size[0] * ratio), int(img_orig.size[1] * ratio))
    img_res = img_orig.resize(nuevo_tam, Image.LANCZOS)
    
    if st.session_state.imagen_base is None:
        st.session_state.imagen_base = np.array(img_res)

    # Selector de Color
    col1, col2 = st.columns([2, 1])
    with col1:
        color_hex = st.color_picker("Elige tu color de pintura:", "#3498db")
    with col2:
        if st.button("🔄 REINICIAR TODO"):
            st.session_state.areas_pintadas = []
            st.rerun()

    # Mostrar resultado acumulado
    imagen_mostrar = st.session_state.imagen_base.copy()
    for area in st.session_state.areas_pintadas:
        c = area['color'].lstrip('#')
        rgb = tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
        imagen_mostrar[area['mascara']] = rgb
    
    st.image(imagen_mostrar, caption="Vista Previa", use_container_width=True)

    # 5. LIENZO PARA CLIC (CANVAS)
    st.markdown("### 👇 Toca la pared que quieres pintar:")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=0,
        background_image=Image.fromarray(st.session_state.imagen_base),
        update_streamlit=True,
        height=img_res.size[1],
        width=img_res.size[0],
        drawing_mode="point",
        point_display_radius=10,
        key="canvas_puntos",
    )

    # 6. LÓGICA DE PINTURA
    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        punto = canvas_result.json_data["objects"][-1] # El último clic
        x, y = int(punto["left"]), int(punto["top"])
        
        if st.button("🖌️ APLICAR PINTURA AHORA", type="primary", use_container_width=True):
            with st.spinner("IA analizando la pared..."):
                predictor = cargar_predictor()
                if predictor:
                    predictor.set_image(st.session_state.imagen_base)
                    mascaras, scores, _ = predictor.predict(
                        point_coords=np.array([[x, y]]),
                        point_labels=np.array([1]),
                        multimask_output=True
                    )
                    mejor_mask = mascaras[np.argmax(scores)]
                    
                    # Guardar en memoria
                    st.session_state.areas_pintadas.append({
                        'mascara': mejor_mask,
                        'color': color_hex
                    })
                    st.success("¡Pintado! Cargando resultado...")
                    st.rerun()
else:
    st.info("Por favor, sube una foto para empezar.")
                
        
        

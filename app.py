import streamlit as st
import numpy as np
from PIL import Image
import torch
from streamlit_image_coordinates import streamlit_image_coordinates
import os
import requests

# Configuración
st.set_page_config(page_title="Simulador de Pintura", layout="wide")

# Función para descargar el modelo
@st.cache_data
def descargar_modelo():
    modelo_path = "mobile_sam.pt"
    if not os.path.exists(modelo_path):
        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        with st.spinner("⏳ Descargando modelo..."):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(modelo_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                st.error(f"Error: {e}")
                return None
    return modelo_path

# Función para cargar predictor
@st.cache_resource
def cargar_predictor():
    try:
        from mobile_sam import sam_model_registry, SamPredictor
        modelo_path = descargar_modelo()
        if modelo_path is None:
            return None
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        modelo = sam_model_registry["vit_t"](checkpoint=modelo_path)
        modelo.to(device=device)
        modelo.eval()
        return SamPredictor(modelo)
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Inicializar
if 'paredes' not in st.session_state:
    st.session_state.paredes = []
if 'imagen_original' not in st.session_state:
    st.session_state.imagen_original = None

# Título
st.title("🎨 Simulador de Pintura Simple")

# Subir foto
archivo = st.file_uploader("📸 Sube tu foto", type=["jpg", "jpeg", "png"])

if archivo:
    # Cargar imagen
    img = Image.open(archivo).convert("RGB")
    
    # Redimensionar
    max_size = 800
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        nuevo_w = int(img.size[0] * ratio)
        nuevo_h = int(img.size[1] * ratio)
        img = img.resize((nuevo_w, nuevo_h), Image.Resampling.LANCZOS)
    
    # Guardar original
    if st.session_state.imagen_original is None:
        st.session_state.imagen_original = np.array(img)
    
    # Selector de color
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        color = st.color_picker("Color:", "#FF6B6B")
    
    with col2:
        if st.session_state.paredes:
            if st.button("🗑️ BORRAR TODO"):
                st.session_state.paredes = []
                st.rerun()
    
    # Mostrar resultado actual
    st.markdown("---")
    if st.session_state.paredes:
        img_resultado = st.session_state.imagen_original.copy()
        for pared in st.session_state.paredes:
            rgb = tuple(int(pared['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            img_resultado[pared['mask']] = rgb
        img_mostrar = Image.fromarray(img_resultado)
        st.markdown(f"**✅ {len(st.session_state.paredes)} pared(es) pintada(s)**")
    else:
        img_mostrar = img
        st.markdown("**👇 HAZ CLIC EN LA PARED QUE QUIERES PINTAR:**")
    
    # Capturar clic en la imagen
    value = streamlit_image_coordinates(
        img_mostrar,
        key="image_coords"
    )
    
    # Procesar clic
    if value is not None:
        x = value["x"]
        y = value["y"]
        
        st.success(f"📍 Clic en: ({x}, {y})")
        
        if st.button("🎨 PINTAR ESTA PARED", type="primary"):
            with st.spinner("🤖 Detectando pared..."):
                predictor = cargar_predictor()
                
                if predictor:
                    try:
                        img_np = np.array(img)
                        predictor.set_image(img_np)
                        
                        masks, scores, _ = predictor.predict(
                            point_coords=np.array([[x, y]]),
                            point_labels=np.array([1]),
                            multimask_output=True,
                        )
                        
                        # Mejor máscara
                        mejor = np.argmax(scores)
                        mask = masks[mejor]
                        
                        # Guardar
                        st.session_state.paredes.append({
                            'mask': mask,
                            'color': color
                        })
                        
                        st.success(f"✅ Pared pintada!")
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Lista de paredes
    if st.session_state.paredes:
        st.markdown("---")
        st.markdown("### 📋 Paredes:")
        for i, p in enumerate(st.session_state.paredes):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f'<div style="background:{p["color"]}; height:40px; border-radius:5px; border:2px solid black;"></div>', unsafe_allow_html=True)
            with col_b:
                if st.button("❌", key=f"del{i}"):
                    st.session_state.paredes.pop(i)
                    st.rerun()

else:
    st.info("👆 Sube una foto de tu habitación")

st.markdown("---")
st.markdown("""
### 📖 Instrucciones:
1. Sube una foto
2. Elige un color
3. **HAZ CLIC DIRECTAMENTE EN LA PARED**
4. Presiona "PINTAR ESTA PARED"
5. Repite para más paredes
""")
                
        
        

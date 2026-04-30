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
        with st.spinner("⏳ Descargando cerebro de la IA (esto solo pasa una vez)..."):
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
st.title("🎨 Simulador de Pintura Realista")

# Subir foto
archivo = st.file_uploader("📸 Sube tu foto", type=["jpg", "jpeg", "png"])

if archivo:
    # Cargar imagen
    img_pil = Image.open(archivo).convert("RGB")
    
    # Redimensionar
    max_size = 800
    if max(img_pil.size) > max_size:
        ratio = max_size / max(img_pil.size)
        nuevo_w = int(img_pil.size[0] * ratio)
        nuevo_h = int(img_pil.size[1] * ratio)
        img_pil = img_pil.resize((nuevo_w, nuevo_h), Image.Resampling.LANCZOS)
    
    # Guardar original en formato numpy para procesar
    if st.session_state.imagen_original is None:
        st.session_state.imagen_original = np.array(img_pil)
    
    # Selector de color
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        color_elegido = st.color_picker("Elige el color de pintura:", "#FF6B6B")
    
    with col2:
        if st.session_state.paredes:
            if st.button("🗑️ BORRAR TODO"):
                st.session_state.paredes = []
                st.rerun()
    
    # --- PROCESO DE PINTURA REALISTA ---
    img_resultado = st.session_state.imagen_original.copy().astype(float)
    
    if st.session_state.paredes:
        for pared in st.session_state.paredes:
            # Convertir HEX a RGB
            hex_color = pared['color'].lstrip('#')
            rgb_nuevo = np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])
            
            # MÁSCARA: Donde está la pared seleccionada
            mask = pared['mask']
            
            # FÓRMULA DE REALISMO (Alpha Blending)
            # Mezclamos el 40% del color nuevo con el 60% de la textura original
            img_resultado[mask] = (img_resultado[mask] * 0.6) + (rgb_nuevo * 0.4)
            
        img_final = Image.fromarray(img_resultado.astype(np.uint8))
        st.markdown(f"**✅ {len(st.session_state.paredes)} pared(es) pintada(s) con realismo**")
    else:
        img_final = img_pil
        st.markdown("**👇 HAZ CLIC EN LA PARED QUE QUIERES PINTAR:**")
    
    # Capturar clic en la imagen
    value = streamlit_image_coordinates(
        img_final,
        key="image_coords"
    )
    
    # Procesar clic
    if value is not None:
        x, y = value["x"], value["y"]
        st.success(f"📍 Punto marcado: ({x}, {y})")
        
        if st.button("🎨 PINTAR ESTA PARED", type="primary", use_container_width=True):
            with st.spinner("🤖 La IA está analizando la textura de la pared..."):
                predictor = cargar_predictor()
                
                if predictor:
                    try:
                        # Usar la imagen original para que la IA no se confunda con colores previos
                        predictor.set_image(st.session_state.imagen_original)
                        
                        masks, scores, _ = predictor.predict(
                            point_coords=np.array([[x, y]]),
                            point_labels=np.array([1]),
                            multimask_output=True,
                        )
                        
                        # Seleccionar la máscara con mejor puntuación
                        mask = masks[np.argmax(scores)]
                        
                        # Guardar en la lista
                        st.session_state.paredes.append({
                            'mask': mask,
                            'color': color_elegido
                        })
                        
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error técnico: {e}")
    
    # Lista de historial de paredes
    if st.session_state.paredes:
        st.markdown("---")
        st.markdown("### 📋 Capas de pintura aplicadas:")
        for i, p in enumerate(st.session_state.paredes):
            col_a, col_b = st.columns([5, 1])
            with col_a:
                st.markdown(f'<div style="background:{p["color"]}; height:35px; border-radius:5px; border:1px solid #ccc;"></div>', unsafe_allow_html=True)
            with col_b:
                if st.button("Eliminar", key=f"del{i}"):
                    st.session_state.paredes.pop(i)
                    st.rerun()

else:
    st.info("👆 Por favor, sube una foto para comenzar el simulador.")

st.markdown("---")
st.caption("Desarrollado con MobileSAM IA para un acabado profesional.")
                
        
        

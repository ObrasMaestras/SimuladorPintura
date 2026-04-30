import streamlit as st
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
import torch
import os
import requests
from scipy import ndimage

st.set_page_config(page_title="Simulador de Pintura", layout="wide")

def aplicar_color_directo(imagen_np, mascara, color_hex, intensidad=0.6):
    """
    Aplica color de forma DIRECTA con blend simple
    Garantiza que el color se vea
    """
    # Convertir hex a RGB
    color_rgb = np.array([int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])
    
    # Crear copia de la imagen
    resultado = imagen_np.copy().astype(float)
    
    # BLEND DIRECTO: mezclar color con imagen original
    # Formula: resultado = original * (1 - intensidad) + color * intensidad
    for canal in range(3):
        resultado[mascara, canal] = (
            imagen_np[mascara, canal] * (1 - intensidad) +
            color_rgb[canal] * intensidad
        )
    
    return resultado.astype(np.uint8)

def mejorar_mascara_simple(mascara_sam, imagen_np):
    """Mejora la máscara de forma simple"""
    # 1. Rellenar huecos
    mascara = ndimage.binary_fill_holes(mascara_sam)
    
    # 2. Expandir y suavizar
    estructura = np.ones((7, 7))
    mascara = ndimage.binary_closing(mascara, structure=estructura, iterations=4)
    mascara = ndimage.binary_dilation(mascara, structure=np.ones((5, 5)), iterations=2)
    
    # 3. Excluir círculos grandes (ventilador)
    gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
    circulos = cv2.HoughCircles(
        gris, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=100, param2=35, minRadius=30, maxRadius=150
    )
    
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for c in circulos[0, :]:
            cx, cy, r = c
            yy, xx = np.ogrid[:imagen_np.shape[0], :imagen_np.shape[1]]
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            circulo_mask = dist <= (r * 1.4)
            mascara = np.logical_and(mascara, ~circulo_mask)
    
    return mascara

@st.cache_data
def descargar_modelo():
    modelo_path = "mobile_sam.pt"
    if not os.path.exists(modelo_path):
        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        with st.spinner("⏳ Descargando MobileSAM..."):
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

st.title("🎨 Simulador de Pintura")
st.markdown("**Aplicación de color directa y visible**")

st.markdown("---")
col_int, col_info = st.columns([1, 2])
with col_int:
    intensidad = st.slider("💧 Intensidad", 0.3, 1.0, 0.7, 0.05)
with col_info:
    st.info(f"✨ Intensidad: {int(intensidad*100)}% - Más alto = color más fuerte")

archivo = st.file_uploader("📸 Sube tu foto", type=["jpg", "jpeg", "png"])

if archivo:
    img = Image.open(archivo).convert("RGB")
    
    max_size = 800
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        nuevo_w = int(img.size[0] * ratio)
        nuevo_h = int(img.size[1] * ratio)
        img = img.resize((nuevo_w, nuevo_h), Image.Resampling.LANCZOS)
    
    if st.session_state.imagen_original is None:
        st.session_state.imagen_original = np.array(img)
    
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("**🎨 Elige tu color:**")
        color = st.color_picker("", "#4CAF50", label_visibility="collapsed")
    
    with col2:
        st.markdown("**Preview del color:**")
        st.markdown(f'<div style="background:{color}; height:50px; border-radius:10px; border:3px solid #333; margin-top:8px; box-shadow:3px 3px 10px rgba(0,0,0,0.3);"></div>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.paredes:
            if st.button("🗑️"):
                st.session_state.paredes = []
                st.rerun()
    
    st.markdown("---")
    
    # Mostrar resultado
    if st.session_state.paredes:
        img_resultado = st.session_state.imagen_original.copy()
        
        # Aplicar cada pared
        for pared in st.session_state.paredes:
            img_resultado = aplicar_color_directo(
                img_resultado, 
                pared['mask'], 
                pared['color'],
                pared['intensidad']
            )
        
        img_mostrar = Image.fromarray(img_resultado)
        st.success(f"✅ {len(st.session_state.paredes)} pared(es) pintada(s) con color visible")
    else:
        img_mostrar = img
        st.info("👇 **HAZ CLIC** en la pared")
    
    # Capturar clic
    value = streamlit_image_coordinates(img_mostrar, key="image_coords")
    
    if value is not None:
        x = value["x"]
        y = value["y"]
        
        st.success(f"📍 Clic en: ({x}, {y})")
        
        if st.button("🎨 PINTAR PARED", type="primary"):
            with st.spinner("🤖 Detectando pared..."):
                predictor = cargar_predictor()
                
                if predictor:
                    try:
                        img_np = np.array(img)
                        predictor.set_image(img_np)
                        
                        # Generar máscaras
                        masks, scores, _ = predictor.predict(
                            point_coords=np.array([[x, y]]),
                            point_labels=np.array([1]),
                            multimask_output=True,
                        )
                        
                        # Mejor máscara
                        mejor_idx = np.argmax(scores)
                        mascara_sam = masks[mejor_idx]
                        
                        # Mejorar
                        mascara_final = mejorar_mascara_simple(mascara_sam, img_np)
                        
                        # Guardar
                        st.session_state.paredes.append({
                            'mask': mascara_final,
                            'color': color,
                            'intensidad': intensidad
                        })
                        
                        st.success(f"🎉 ¡Pared #{len(st.session_state.paredes)} pintada!")
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.exception(e)
    
    # Lista de paredes
    if st.session_state.paredes:
        st.markdown("---")
        st.markdown("### 📋 Paredes:")
        for i, p in enumerate(st.session_state.paredes):
            col_a, col_b, col_c, col_d = st.columns([1, 2, 2, 1])
            with col_a:
                st.markdown(f"**{i+1}**")
            with col_b:
                st.markdown(f'<div style="background:{p["color"]}; height:40px; border-radius:8px; border:2px solid #000;"></div>', unsafe_allow_html=True)
            with col_c:
                st.markdown(f"*{int(p['intensidad']*100)}%*")
            with col_d:
                if st.button("❌", key=f"d{i}"):
                    st.session_state.paredes.pop(i)
                    st.rerun()

else:
    st.info("👆 Sube una foto")

st.markdown("---")
st.markdown("""
### 🚀 Aplicación Directa de Color:

✅ **Blend RGB directo** - Mezcla pixel por pixel  
✅ **70% intensidad** - Color visible y realista  
✅ **Fórmula simple** - `original × 30% + color × 70%`  
✅ **Garantizado** - El color SIEMPRE se ve

💡 **Sube intensidad al 90-100% para color más fuerte**
""")

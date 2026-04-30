import streamlit as st
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import torch
import os
import requests
from scipy import ndimage
import cv2

st.set_page_config(page_title="Simulador de Pintura Preciso", layout="wide")

def aplicar_alpha_blending(imagen_np, mascara, color_hex, alpha_color=0.4):
    """Alpha Blending: 0.6 original + 0.4 color"""
    color_rgb = np.array([int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)], dtype=float)
    resultado = imagen_np.copy().astype(float)
    alpha_original = 1.0 - alpha_color
    
    for canal in range(3):
        resultado[mascara, canal] = (
            imagen_np[mascara, canal] * alpha_original +
            color_rgb[canal] * alpha_color
        )
    
    return resultado.astype(np.uint8)

def refinar_mascara_inteligente(mascara_sam, imagen_np, punto_clic):
    """
    Refina la máscara para excluir techo, ventiladores, objetos
    """
    x, y = punto_clic
    alto, ancho = imagen_np.shape[:2]
    
    # 1. Usar SOLO la región conectada al punto clickeado
    etiquetas, num = ndimage.label(mascara_sam)
    if etiquetas[y, x] > 0:
        mascara_region = (etiquetas == etiquetas[y, x])
    else:
        mascara_region = mascara_sam
    
    # 2. EXCLUIR área superior (techo) - 25% superior de la imagen
    limite_techo = int(alto * 0.25)
    mascara_sin_techo = mascara_region.copy()
    mascara_sin_techo[:limite_techo, :] = False
    
    # 3. DETECTAR Y EXCLUIR CÍRCULOS (ventiladores, lámparas)
    gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
    circulos = cv2.HoughCircles(
        gris, cv2.HOUGH_GRADIENT, dp=1, minDist=80,
        param1=100, param2=30, minRadius=25, maxRadius=120
    )
    
    mascara_final = mascara_sin_techo.copy()
    
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for c in circulos[0, :]:
            cx, cy, r = c
            # Solo excluir círculos en la parte SUPERIOR (ventiladores en techo)
            if cy < alto * 0.5:
                yy, xx = np.ogrid[:alto, :ancho]
                dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
                circulo_mask = dist <= (r * 1.5)
                mascara_final = np.logical_and(mascara_final, ~circulo_mask)
    
    # 4. Rellenar huecos pequeños
    mascara_final = ndimage.binary_fill_holes(mascara_final)
    
    # 5. Suavizar bordes
    kernel = np.ones((3, 3))
    mascara_final = ndimage.binary_closing(mascara_final, structure=kernel, iterations=2)
    
    return mascara_final

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

st.title("🎨 Simulador de Pintura - Versión Precisa")
st.markdown("**Excluye techo, ventiladores y objetos automáticamente**")

st.markdown("---")
alpha_color = st.slider("💧 Intensidad del color", 0.2, 0.8, 0.4, 0.05)

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
        color = st.color_picker("🎨 Color:", "#8FBC8F")
    
    with col2:
        st.markdown(f'<div style="background:{color}; height:50px; border-radius:10px; border:3px solid #333; margin-top:25px;"></div>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.paredes:
            if st.button("🗑️"):
                st.session_state.paredes = []
                st.rerun()
    
    st.markdown("---")
    
    if st.session_state.paredes:
        img_resultado = st.session_state.imagen_original.copy()
        
        for pared in st.session_state.paredes:
            img_resultado = aplicar_alpha_blending(
                img_resultado, 
                pared['mask'], 
                pared['color'],
                pared['alpha']
            )
        
        img_mostrar = Image.fromarray(img_resultado)
        st.success(f"✅ {len(st.session_state.paredes)} pared(es)")
    else:
        img_mostrar = img
        st.info("👇 **HAZ CLIC** en la pared (NO en el techo ni ventilador)")
    
    value = streamlit_image_coordinates(img_mostrar, key="image_coords")
    
    if value is not None:
        x = value["x"]
        y = value["y"]
        
        st.success(f"📍 Clic: ({x}, {y})")
        
        if st.button("🎨 PINTAR SOLO LA PARED", type="primary"):
            with st.spinner("🤖 Segmentando con precisión..."):
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
                        
                        # Usar máscara MÁS PEQUEÑA (más conservadora)
                        # En lugar de la de mayor score, usar la de menor área
                        areas = [np.sum(m) for m in masks]
                        idx_pequeña = np.argmin(areas)
                        mascara_inicial = masks[idx_pequeña]
                        
                        st.info(f"🎯 Usando máscara conservadora (área: {areas[idx_pequeña]} px)")
                        
                        # REFINAR para excluir techo y objetos
                        mascara_final = refinar_mascara_inteligente(
                            mascara_inicial,
                            img_np,
                            (x, y)
                        )
                        
                        # Guardar
                        st.session_state.paredes.append({
                            'mask': mascara_final,
                            'color': color,
                            'alpha': alpha_color
                        })
                        
                        st.success(f"🎉 ¡Pared #{len(st.session_state.paredes)}!")
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    if st.session_state.paredes:
        st.markdown("---")
        st.markdown("### 📋 Paredes:")
        for i, p in enumerate(st.session_state.paredes):
            col_a, col_b, col_c = st.columns([1, 3, 1])
            with col_a:
                st.markdown(f"**{i+1}**")
            with col_b:
                st.markdown(f'<div style="background:{p["color"]}; height:40px; border-radius:8px; border:2px solid #000;"></div>', unsafe_allow_html=True)
            with col_c:
                if st.button("❌", key=f"d{i}"):
                    st.session_state.paredes.pop(i)
                    st.rerun()

else:
    st.info("👆 Sube una foto")

st.markdown("---")
st.markdown("### 🚀 Mejoras de Precisión:")
st.markdown("✅ **Máscara conservadora** - Usa la más pequeña, no la de mayor score")
st.markdown("✅ **Excluye 25% superior** - NO pinta el techo")
st.markdown("✅ **Detecta círculos** - NO pinta ventiladores")
st.markdown("✅ **Solo región conectada** - Pinta área donde hiciste clic")
st.markdown("💡 **Haz clic en la PARED directamente, no cerca del techo**")

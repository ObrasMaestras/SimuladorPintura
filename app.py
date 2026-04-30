import streamlit as st
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
import torch
import os
import requests
from scipy import ndimage

st.set_page_config(page_title="Simulador de Pintura Profesional", layout="wide")

def detectar_objetos_3d_inteligente(imagen_np, punto_pared):
    """
    Detecta objetos que sobresalen (ventilador, cuadros) vs pared plana
    """
    alto, ancho = imagen_np.shape[:2]
    gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
    
    # 1. DETECTAR CÍRCULOS (ventiladores, lámparas)
    circulos = cv2.HoughCircles(
        gris, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=100, param2=30, minRadius=15, maxRadius=200
    )
    
    mascara_objetos = np.zeros((alto, ancho), dtype=bool)
    
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for c in circulos[0, :]:
            cx, cy, r = c
            # Crear máscara circular
            yy, xx = np.ogrid[:alto, :ancho]
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            mascara_objetos |= (dist <= r * 1.3)
    
    # 2. DETECTAR RECTÁNGULOS (cuadros, puertas)
    bordes = cv2.Canny(gris, 50, 150)
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bordes_cerrados = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel_rect)
    
    contornos, _ = cv2.findContours(bordes_cerrados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        # Solo objetos medianos (cuadros, no toda la pared)
        if 500 < area < 50000:
            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = float(w) / h if h > 0 else 0
            # Rectángulos típicos (cuadros, puertas)
            if 0.3 < aspect_ratio < 3:
                cv2.drawContours(mascara_objetos.astype(np.uint8), [contorno], -1, True, -1)
    
    # 3. DETECTAR ÁREAS CON ALTA VARIACIÓN DE TEXTURA (cuadros con detalles)
    laplaciano = cv2.Laplacian(gris, cv2.CV_64F)
    textura = np.abs(laplaciano)
    umbral_textura = np.percentile(textura, 97)
    alta_textura = textura > umbral_textura
    
    # Dilatar áreas de alta textura
    alta_textura = ndimage.binary_dilation(alta_textura, iterations=3)
    mascara_objetos |= alta_textura
    
    return mascara_objetos

def expandir_pared_inteligente(mascara_sam, imagen_np, punto_clic):
    """
    Expande la pared de manera controlada, excluyendo objetos 3D
    """
    x, y = punto_clic
    alto, ancho = imagen_np.shape[:2]
    
    # 1. COLOR DE REFERENCIA DE LA PARED
    radio = 20
    x1, x2 = max(0, x-radio), min(ancho, x+radio)
    y1, y2 = max(0, y-radio), min(alto, y+radio)
    region = imagen_np[y1:y2, x1:x2]
    color_pared = np.median(region.reshape(-1, 3), axis=0)
    
    # 2. SIMILITUD DE COLOR (LAB más preciso)
    img_lab = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2LAB).astype(float)
    color_lab = cv2.cvtColor(np.uint8([[color_pared]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    
    diff_color = np.sqrt(np.sum((img_lab - color_lab) ** 2, axis=2))
    
    # Usar máscara SAM para calcular umbral adaptativo
    umbral_base = np.percentile(diff_color[mascara_sam], 75)
    mascara_color_similar = diff_color < (umbral_base * 1.8)
    
    # 3. DETECTAR OBJETOS 3D
    mascara_objetos_3d = detectar_objetos_3d_inteligente(imagen_np, (x, y))
    
    # 4. EXPANDIR CON WATERSHED (expansión controlada)
    # Marcadores: pared segura (SAM) vs fondo
    marcadores = np.zeros((alto, ancho), dtype=np.int32)
    marcadores[mascara_sam] = 1  # Pared segura
    marcadores[mascara_objetos_3d] = 2  # Objetos
    
    # Aplicar watershed
    gradiente = cv2.morphologyEx(cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY), 
                                   cv2.MORPH_GRADIENT, np.ones((3, 3)))
    gradiente_color = cv2.cvtColor(gradiente, cv2.COLOR_GRAY2RGB)
    
    marcadores = cv2.watershed(gradiente_color, marcadores)
    mascara_watershed = (marcadores == 1)
    
    # 5. COMBINAR: Watershed Y similitud de color, PERO NO objetos 3D
    mascara_expandida = np.logical_and(
        np.logical_or(mascara_watershed, mascara_color_similar),
        ~mascara_objetos_3d
    )
    
    # 6. MANTENER SOLO REGIÓN CONECTADA
    etiquetas, _ = ndimage.label(mascara_expandida)
    if etiquetas[y, x] > 0:
        mascara_region = (etiquetas == etiquetas[y, x])
    else:
        mascara_region = mascara_expandida
    
    # 7. RELLENAR HUECOS PEQUEÑOS (cables, plantas pequeñas)
    mascara_rellena = ndimage.binary_fill_holes(mascara_region)
    
    # 8. SUAVIZAR BORDES
    estructura = np.ones((5, 5))
    mascara_final = ndimage.binary_closing(mascara_rellena, structure=estructura, iterations=3)
    
    return mascara_final

def aplicar_color_realista(imagen_np, mascara, color_hex, intensidad=0.6):
    """Aplica color preservando luminosidad"""
    color_rgb = np.array([int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])
    
    img_hsv = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2HSV).astype(float)
    color_hsv = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2HSV)[0, 0].astype(float)
    
    img_resultado = img_hsv.copy()
    img_resultado[mascara, 0] = color_hsv[0]
    img_resultado[mascara, 1] = color_hsv[1] * intensidad + img_hsv[mascara, 1] * (1 - intensidad)
    
    img_resultado = img_resultado.astype(np.uint8)
    return cv2.cvtColor(img_resultado, cv2.COLOR_HSV2RGB)

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

st.title("🎨 Simulador de Pintura Inteligente")
st.markdown("**Detección de objetos 3D + Expansión controlada**")

st.markdown("---")
intensidad = st.slider("💧 Intensidad del color", 0.3, 1.0, 0.6, 0.05)

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
        st.markdown("**🎨 Color:**")
        color = st.color_picker("", "#8FBC8F", label_visibility="collapsed")
    
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
            img_resultado = aplicar_color_realista(
                img_resultado, 
                pared['mask'], 
                pared['color'],
                pared['intensidad']
            )
        img_mostrar = Image.fromarray(img_resultado)
        st.success(f"✅ {len(st.session_state.paredes)} pared(es)")
    else:
        img_mostrar = img
        st.info("👇 **HAZ CLIC** en la pared")
    
    value = streamlit_image_coordinates(img_mostrar, key="image_coords")
    
    if value is not None:
        x = value["x"]
        y = value["y"]
        
        st.success(f"📍 Clic: ({x}, {y})")
        
        if st.button("🎨 PINTAR PARED", type="primary"):
            with st.spinner("🤖 Analizando..."):
                predictor = cargar_predictor()
                
                if predictor:
                    try:
                        img_np = np.array(img)
                        predictor.set_image(img_np)
                        
                        # Máscara inicial SAM
                        masks, scores, _ = predictor.predict(
                            point_coords=np.array([[x, y]]),
                            point_labels=np.array([1]),
                            multimask_output=True,
                        )
                        
                        mejor_idx = np.argmax(scores)
                        mascara_sam = masks[mejor_idx]
                        
                        # EXPANSIÓN INTELIGENTE
                        mascara_final = expandir_pared_inteligente(
                            mascara_sam,
                            img_np,
                            (x, y)
                        )
                        
                        # Guardar
                        st.session_state.paredes.append({
                            'mask': mascara_final,
                            'color': color,
                            'intensidad': intensidad
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
st.markdown("""
### 🚀 Algoritmo Inteligente:

1. ✅ **Detección de círculos** - Encuentra ventiladores/lámparas automáticamente
2. ✅ **Detección de rectángulos** - Encuentra cuadros/puertas
3. ✅ **Análisis de textura** - Identifica objetos con detalles
4. ✅ **Watershed** - Expansión controlada desde SAM
5. ✅ **Similitud de color** - Solo pinta áreas del mismo color
6. ✅ **Exclusión de objetos 3D** - NO pinta objetos que sobresalen

💡 **Equilibrio perfecto** entre cobertura completa y precisión
""")

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

def expandir_pared_completa(mascara_inicial, imagen_np, punto_clic):
    """
    EXPANSIÓN AGRESIVA: Detecta TODA la pared incluyendo áreas detrás de objetos
    """
    x, y = punto_clic
    alto, ancho = imagen_np.shape[:2]
    
    # 1. ANÁLISIS DE COLOR - Obtener color de referencia de la pared
    radio = 15
    x1, x2 = max(0, x-radio), min(ancho, x+radio)
    y1, y2 = max(0, y-radio), min(alto, y+radio)
    region_ref = imagen_np[y1:y2, x1:x2]
    color_pared_rgb = np.median(region_ref.reshape(-1, 3), axis=0)
    
    # 2. CONVERTIR A MÚLTIPLES ESPACIOS DE COLOR
    img_lab = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2LAB).astype(float)
    color_lab = cv2.cvtColor(np.uint8([[color_pared_rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    
    img_hsv = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2HSV).astype(float)
    color_hsv = cv2.cvtColor(np.uint8([[color_pared_rgb]]), cv2.COLOR_RGB2HSV)[0, 0].astype(float)
    
    # 3. SIMILITUD DE COLOR (muy tolerante para incluir sombras)
    diff_lab = np.sqrt(np.sum((img_lab - color_lab) ** 2, axis=2))
    diff_h = np.abs(img_hsv[:, :, 0] - color_hsv[0])
    diff_h = np.minimum(diff_h, 180 - diff_h)  # Circular
    
    # Umbrales MUY generosos para capturar toda la pared
    umbral_lab = np.percentile(diff_lab[mascara_inicial], 85)
    umbral_h = np.percentile(diff_h[mascara_inicial], 90)
    
    mascara_color = (diff_lab < umbral_lab * 2.0) & (diff_h < umbral_h * 1.5)
    
    # 4. DETECTAR OBJETOS 3D QUE SOBRESALEN (cuadros, cables)
    gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
    
    # Detectar bordes MUY fuertes (objetos con marco)
    bordes_fuertes = cv2.Canny(gris, 100, 200)
    
    # Detectar texturas diferentes (cuadros tienen textura distinta a pared lisa)
    laplaciano = cv2.Laplacian(gris, cv2.CV_64F)
    textura_alta = np.abs(laplaciano) > np.percentile(np.abs(laplaciano), 95)
    
    # 5. FLOOD FILL desde el punto clickeado (expansión agresiva)
    mascara_flood = np.zeros((alto + 2, ancho + 2), dtype=np.uint8)
    semilla = mascara_color.astype(np.uint8) * 255
    cv2.floodFill(semilla, mascara_flood, (x, y), 255, 
                  loDiff=(30, 30, 30), upDiff=(30, 30, 30),
                  flags=cv2.FLOODFILL_FIXED_RANGE)
    mascara_flood = semilla > 0
    
    # 6. COMBINAR: color similar O flood fill
    mascara_expandida = np.logical_or(mascara_color, mascara_flood)
    
    # 7. EXCLUIR solo objetos 3D muy obvios
    # Crear máscara de objetos: bordes fuertes Y textura alta
    kernel = np.ones((7, 7), np.uint8)
    bordes_dilatados = cv2.dilate(bordes_fuertes, kernel, iterations=2)
    objetos_3d = np.logical_and(bordes_dilatados > 0, textura_alta)
    
    # Dilatar objetos para excluir bien
    objetos_3d = ndimage.binary_dilation(objetos_3d, structure=np.ones((5, 5)), iterations=3)
    
    # Aplicar exclusión
    mascara_sin_objetos = np.logical_and(mascara_expandida, ~objetos_3d)
    
    # 8. MANTENER SOLO REGIÓN CONECTADA AL CLIC
    etiquetas, num = ndimage.label(mascara_sin_objetos)
    if 0 <= y < alto and 0 <= x < ancho and etiquetas[y, x] > 0:
        mascara_region = (etiquetas == etiquetas[y, x])
    else:
        mascara_region = mascara_sin_objetos
    
    # 9. RELLENAR HUECOS INTERNOS (pintar detrás de cables, plantas pequeñas)
    mascara_rellena = ndimage.binary_fill_holes(mascara_region)
    
    # 10. EXPANSIÓN MORFOLÓGICA para llegar a esquinas
    estructura = np.ones((7, 7))
    mascara_final = ndimage.binary_dilation(mascara_rellena, structure=estructura, iterations=3)
    
    # 11. CIERRE para conectar áreas cercanas
    mascara_final = ndimage.binary_closing(mascara_final, structure=estructura, iterations=5)
    
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

st.title("🎨 Simulador de Pintura - Detección Completa")
st.markdown("**Expansión agresiva: pinta TODA la pared incluyendo áreas detrás de objetos**")

st.markdown("---")
intensidad = st.slider("💧 Intensidad del color", 0.3, 1.0, 0.6, 0.05,
                       help="Controla qué tan fuerte se ve el color")

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
        st.success(f"✅ {len(st.session_state.paredes)} pared(es) - Expansión completa")
    else:
        img_mostrar = img
        st.info("👇 **HAZ CLIC** en la pared")
    
    value = streamlit_image_coordinates(img_mostrar, key="image_coords")
    
    if value is not None:
        x = value["x"]
        y = value["y"]
        
        st.success(f"📍 Clic: ({x}, {y})")
        
        if st.button("🎨 PINTAR PARED COMPLETA", type="primary"):
            with st.spinner("🤖 Analizando y expandiendo..."):
                predictor = cargar_predictor()
                
                if predictor:
                    try:
                        img_np = np.array(img)
                        predictor.set_image(img_np)
                        
                        # Máscara inicial con SAM
                        masks, scores, _ = predictor.predict(
                            point_coords=np.array([[x, y]]),
                            point_labels=np.array([1]),
                            multimask_output=True,
                        )
                        
                        mejor_idx = np.argmax(scores)
                        mascara_inicial = masks[mejor_idx]
                        
                        # EXPANSIÓN COMPLETA
                        mascara_final = expandir_pared_completa(
                            mascara_inicial,
                            img_np,
                            (x, y)
                        )
                        
                        # Guardar
                        st.session_state.paredes.append({
                            'mask': mascara_final,
                            'color': color,
                            'intensidad': intensidad
                        })
                        
                        st.success(f"🎉 ¡Pared completa pintada!")
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
### 🚀 Algoritmo de Expansión Completa:

1. ✅ **Análisis de color LAB + HSV** - Detecta toda el área de color similar
2. ✅ **Flood Fill agresivo** - Expande desde el clic hasta los bordes
3. ✅ **Detección de objetos 3D** - Identifica cuadros por bordes + textura
4. ✅ **Relleno de huecos** - Pinta detrás de cables, plantas, sombras
5. ✅ **Expansión morfológica** - Llega hasta todas las esquinas
6. ✅ **Exclusión inteligente** - NO pinta los objetos mismos

💡 **Resultado:** Cubre TODA la pared incluyendo áreas ocultas
""")

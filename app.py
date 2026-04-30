import streamlit as st
import numpy as np
from PIL import Image
import torch
from streamlit_image_coordinates import streamlit_image_coordinates
import os
import requests
from scipy import ndimage
import cv2

# Configuración
st.set_page_config(page_title="Simulador de Pintura", layout="wide")

def detectar_pared_inteligente(imagen_np, mascara_inicial, punto_clic):
    """
    Analiza la imagen para detectar SOLO la pared, excluyendo objetos
    Similar a como lo hace Gemini
    """
    x, y = punto_clic
    alto, ancho = imagen_np.shape[:2]
    
    # 1. Obtener color de referencia de la pared (promedio en área pequeña)
    radio = 20
    x1, x2 = max(0, x-radio), min(ancho, x+radio)
    y1, y2 = max(0, y-radio), min(alto, y+radio)
    area_referencia = imagen_np[y1:y2, x1:x2]
    color_pared = np.mean(area_referencia, axis=(0, 1))
    
    # 2. Convertir a LAB para mejor comparación de color
    img_lab = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2LAB)
    color_pared_lab = cv2.cvtColor(np.uint8([[color_pared]]), cv2.COLOR_RGB2LAB)[0, 0]
    
    # 3. Calcular similitud de color en toda la imagen
    diferencia_color = np.sqrt(np.sum((img_lab - color_pared_lab) ** 2, axis=2))
    
    # 4. Crear máscara de píxeles similares (tolerancia adaptativa)
    umbral_color = np.percentile(diferencia_color[mascara_inicial], 75)
    mascara_similar = diferencia_color < umbral_color * 1.5
    
    # 5. Detectar bordes fuertes (objetos como cuadros, puertas)
    gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
    bordes = cv2.Canny(gris, 50, 150)
    bordes_dilatados = cv2.dilate(bordes, np.ones((3, 3), np.uint8), iterations=2)
    
    # 6. Combinar: píxeles similares en color PERO excluir bordes fuertes
    mascara_pared = np.logical_and(mascara_similar, bordes_dilatados == 0)
    
    # 7. Mantener solo la región conectada donde hiciste clic
    etiquetas, num_regiones = ndimage.label(mascara_pared)
    if 0 <= y < alto and 0 <= x < ancho:
        etiqueta_clic = etiquetas[y, x]
        if etiqueta_clic > 0:
            mascara_pared = (etiquetas == etiqueta_clic)
    
    # 8. Rellenar huecos internos
    mascara_pared = ndimage.binary_fill_holes(mascara_pared)
    
    # 9. Suavizar bordes ligeramente
    mascara_pared = ndimage.binary_closing(mascara_pared, structure=np.ones((3, 3)), iterations=2)
    
    return mascara_pared

def aplicar_color_realista(imagen_np, mascara, color_hex):
    """Aplica color preservando iluminación y textura"""
    color_rgb = np.array([int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])
    
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(imagen_np)
    img_hsv = np.array(img_pil.convert('HSV'))
    
    color_pil = PILImage.new('RGB', (1, 1), tuple(color_rgb))
    color_hsv = np.array(color_pil.convert('HSV'))[0, 0]
    
    # Aplicar color manteniendo luminosidad original
    img_hsv[mascara, 0] = color_hsv[0]  # Hue
    img_hsv[mascara, 1] = int(color_hsv[1] * 0.85)  # Saturación reducida para naturalidad
    
    img_resultado_pil = PILImage.fromarray(img_hsv, mode='HSV').convert('RGB')
    return np.array(img_resultado_pil)

@st.cache_data
def descargar_modelo():
    modelo_path = "mobile_sam.pt"
    if not os.path.exists(modelo_path):
        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        with st.spinner("⏳ Descargando modelo IA..."):
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

st.title("🎨 Simulador de Pintura con IA Avanzada")
st.markdown("**Un solo clic - La IA detecta automáticamente toda la pared**")

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
        color = st.color_picker("", "#8FBC8F", label_visibility="collapsed")
    
    with col2:
        st.markdown("**Vista previa:**")
        st.markdown(f'<div style="background:{color}; height:50px; border-radius:10px; border:3px solid #333; box-shadow: 2px 2px 8px rgba(0,0,0,0.2);"></div>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.paredes:
            if st.button("🗑️ Borrar"):
                st.session_state.paredes = []
                st.rerun()
    
    st.markdown("---")
    
    if st.session_state.paredes:
        img_resultado = st.session_state.imagen_original.copy()
        for pared in st.session_state.paredes:
            img_resultado = aplicar_color_realista(img_resultado, pared['mask'], pared['color'])
        img_mostrar = Image.fromarray(img_resultado)
        st.success(f"✅ {len(st.session_state.paredes)} pared(es) pintada(s) con IA avanzada")
    else:
        img_mostrar = img
        st.info("👇 **HAZ UN CLIC** en cualquier parte de la pared")
    
    value = streamlit_image_coordinates(img_mostrar, key="image_coords")
    
    if value is not None:
        x = value["x"]
        y = value["y"]
        
        st.success(f"📍 Punto seleccionado: ({x}, {y})")
        
        if st.button("🎨 PINTAR ESTA PARED", type="primary"):
            with st.spinner("🤖 Analizando pared con IA avanzada..."):
                predictor = cargar_predictor()
                
                if predictor:
                    try:
                        img_np = np.array(img)
                        predictor.set_image(img_np)
                        
                        # Obtener máscara inicial con SAM
                        masks, scores, _ = predictor.predict(
                            point_coords=np.array([[x, y]]),
                            point_labels=np.array([1]),
                            multimask_output=True,
                        )
                        
                        # Usar la mejor máscara inicial
                        mejor = np.argmax(scores)
                        mascara_inicial = masks[mejor]
                        
                        # ANÁLISIS INTELIGENTE: Detectar solo la pared
                        mascara_pared = detectar_pared_inteligente(
                            img_np, 
                            mascara_inicial, 
                            (x, y)
                        )
                        
                        # Guardar
                        st.session_state.paredes.append({
                            'mask': mascara_pared,
                            'color': color
                        })
                        
                        st.success(f"🎉 ¡Pared #{len(st.session_state.paredes)} pintada con IA!")
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.exception(e)
    
    if st.session_state.paredes:
        st.markdown("---")
        st.markdown("### 📋 Paredes Pintadas:")
        for i, p in enumerate(st.session_state.paredes):
            col_a, col_b, col_c = st.columns([1, 3, 1])
            with col_a:
                st.markdown(f"**Pared {i+1}**")
            with col_b:
                st.markdown(f'<div style="background:{p["color"]}; height:40px; border-radius:8px; border:2px solid #000;"></div>', unsafe_allow_html=True)
            with col_c:
                if st.button("❌", key=f"d{i}"):
                    st.session_state.paredes.pop(i)
                    st.rerun()

else:
    st.info("👆 Sube una foto de tu habitación")

st.markdown("---")
st.markdown("""
### 📖 Cómo funciona la IA:
1. **Sube** una foto bien iluminada
2. **Elige** el color que quieres probar
3. **Haz UN CLIC** en la pared
4. **La IA analiza:**
   - ✅ Color y textura de la pared
   - ✅ Detecta bordes de objetos (cuadros, puertas)
   - ✅ Excluye automáticamente cables, muebles, cuadros
   - ✅ Pinta detrás de obstáculos (sillas, cables)
   - ✅ Rellena huecos y sombras
5. **Resultado** realista como Gemini - Un solo clic

💡 **Tecnología:** Análisis de color LAB + Detección de bordes + Segmentación inteligente
""")

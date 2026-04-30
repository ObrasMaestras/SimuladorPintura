import streamlit as st
import numpy as np
from PIL import Image
import google.generativeai as genai
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
import io
import base64

# Configuración
st.set_page_config(page_title="Simulador de Pintura con Gemini", layout="wide")

def aplicar_color_realista(imagen_np, mascara, color_hex):
    """Aplica color preservando iluminación"""
    color_rgb = np.array([int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])
    
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(imagen_np)
    img_hsv = np.array(img_pil.convert('HSV'))
    
    color_pil = PILImage.new('RGB', (1, 1), tuple(color_rgb))
    color_hsv = np.array(color_pil.convert('HSV'))[0, 0]
    
    img_hsv[mascara, 0] = color_hsv[0]
    img_hsv[mascara, 1] = int(color_hsv[1] * 0.85)
    
    img_resultado_pil = PILImage.fromarray(img_hsv, mode='HSV').convert('RGB')
    return np.array(img_resultado_pil)

def detectar_pared_con_gemini(imagen_pil, punto_clic, api_key):
    """Usa Gemini para analizar la imagen y detectar la pared"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro-vision')  # ← CAMBIADO AQUÍ
        
        x, y = punto_clic
        
        # Crear prompt para Gemini
        prompt = f"""Analiza esta imagen de interior. El usuario hizo clic en las coordenadas ({x}, {y}).

Identifica QUÉ es el elemento en esas coordenadas (pared, mueble, cuadro, etc.) y describe:
1. Si es una PARED o no
2. El color aproximado de la pared
3. Los límites aproximados de esa pared en píxeles (izquierda, derecha, arriba, abajo)
4. Qué objetos están SOBRE la pared pero no son parte de ella (cuadros, cables, etc.)

Responde SOLO en formato JSON:
{{
    "es_pared": true/false,
    "color_pared": "descripción del color",
    "limites": {{"x1": numero, "y1": numero, "x2": numero, "y2": numero}},
    "objetos_excluir": ["objeto1", "objeto2"]
}}"""
        
        response = model.generate_content([prompt, imagen_pil])
        
        # Parsear respuesta de Gemini
        import json
        texto = response.text
        # Extraer JSON de la respuesta
        inicio = texto.find('{')
        fin = texto.rfind('}') + 1
        if inicio >= 0 and fin > inicio:
            json_str = texto[inicio:fin]
            resultado = json.loads(json_str)
            return resultado
        else:
            return None
            
    except Exception as e:
        st.error(f"Error con Gemini API: {e}")
        return None

def crear_mascara_desde_limites(imagen_shape, limites, punto_clic, imagen_np):
    """Crea máscara inteligente basada en los límites de Gemini"""
    alto, ancho = imagen_shape[:2]
    x, y = punto_clic
    
    # Crear máscara base con los límites
    mascara = np.zeros((alto, ancho), dtype=bool)
    
    x1 = max(0, limites.get('x1', 0))
    y1 = max(0, limites.get('y1', 0))
    x2 = min(ancho, limites.get('x2', ancho))
    y2 = min(alto, limites.get('y2', alto))
    
    # Análisis de color para refinar
    color_ref = imagen_np[y, x].astype(float)
    img_lab = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2LAB)
    color_ref_lab = cv2.cvtColor(np.uint8([[color_ref]]), cv2.COLOR_RGB2LAB)[0, 0]
    
    diferencia = np.sqrt(np.sum((img_lab - color_ref_lab) ** 2, axis=2))
    umbral = np.percentile(diferencia[y1:y2, x1:x2], 70)
    
    mascara_color = diferencia < (umbral * 1.5)
    
    # Combinar límites con similitud de color
    mascara[y1:y2, x1:x2] = mascara_color[y1:y2, x1:x2]
    
    # Detectar y excluir bordes fuertes (objetos)
    gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
    bordes = cv2.Canny(gris, 40, 120)
    kernel = np.ones((5, 5), np.uint8)
    bordes_dilatados = cv2.dilate(bordes, kernel, iterations=2)
    
    mascara = np.logical_and(mascara, bordes_dilatados == 0)
    
    # Rellenar huecos
    from scipy import ndimage
    mascara = ndimage.binary_fill_holes(mascara)
    
    return mascara

# Inicializar
if 'paredes' not in st.session_state:
    st.session_state.paredes = []
if 'imagen_original' not in st.session_state:
    st.session_state.imagen_original = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

st.title("🎨 Simulador de Pintura con Google Gemini")
st.markdown("**Powered by Gemini API - Detección inteligente como Gemini**")

# API Key
with st.sidebar:
    st.markdown("### 🔑 Configuración")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Obtén tu key en: https://aistudio.google.com/app/apikey"
    )
    
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.success("✅ API Key configurada")
    else:
        st.warning("⚠️ Ingresa tu API Key para comenzar")
    
    st.markdown("---")
    st.markdown("**💡 Cómo obtener API Key:**")
    st.markdown("1. Ve a [Google AI Studio](https://aistudio.google.com/app/apikey)")
    st.markdown("2. Crea una API Key")
    st.markdown("3. Pégala arriba")

archivo = st.file_uploader("📸 Sube tu foto", type=["jpg", "jpeg", "png"])

if archivo and st.session_state.api_key:
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
            img_resultado = aplicar_color_realista(img_resultado, pared['mask'], pared['color'])
        img_mostrar = Image.fromarray(img_resultado)
        st.success(f"✅ {len(st.session_state.paredes)} pared(es) - Gemini AI")
    else:
        img_mostrar = img
        st.info("👇 **HAZ CLIC** en la pared")
    
    value = streamlit_image_coordinates(img_mostrar, key="image_coords")
    
    if value is not None:
        x = value["x"]
        y = value["y"]
        
        st.success(f"📍 Clic: ({x}, {y})")
        
        if st.button("🤖 ANALIZAR CON GEMINI", type="primary"):
            with st.spinner("🤖 Gemini analizando la imagen..."):
                # Detectar con Gemini
                resultado = detectar_pared_con_gemini(img, (x, y), st.session_state.api_key)
                
                if resultado and resultado.get('es_pared'):
                    st.info(f"🎯 Gemini detectó: {resultado.get('color_pared', 'pared')}")
                    
                    # Crear máscara
                    img_np = np.array(img)
                    mascara = crear_mascara_desde_limites(
                        img_np.shape,
                        resultado.get('limites', {}),
                        (x, y),
                        img_np
                    )
                    
                    # Guardar
                    st.session_state.paredes.append({
                        'mask': mascara,
                        'color': color
                    })
                    
                    st.success("🎉 ¡Pared pintada con Gemini AI!")
                    st.balloons()
                    st.rerun()
                else:
                    st.warning("⚠️ Gemini no detectó una pared en ese punto. Intenta hacer clic en la pared directamente.")
    
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

elif archivo and not st.session_state.api_key:
    st.warning("⚠️ Ingresa tu API Key en la barra lateral para continuar")
else:
    st.info("👆 Sube una foto")

st.markdown("---")
st.markdown("""
### 🚀 Powered by Google Gemini
- Gemini analiza la imagen
- Identifica paredes vs objetos  
- Detecta límites precisos
- Excluye cuadros, cables, muebles automáticamente

**⚡ Gratis:** 60 requests/minuto  
**📖 Docs:** [Google AI Studio](https://ai.google.dev/)
""")

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
st.set_page_config(page_title="Simulador de Pintura Pro", layout="wide")

def detectar_objetos_3d(imagen_np, mascara):
    """Detecta y excluye objetos que sobresalen (ventiladores, lámparas)"""
    gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
    
    # Detectar círculos (ventiladores, lámparas redondas)
    circulos = cv2.HoughCircles(
        gris, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=50,
        param1=100, 
        param2=30, 
        minRadius=20, 
        maxRadius=150
    )
    
    mascara_limpia = mascara.copy()
    
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for circulo in circulos[0, :]:
            x, y, r = circulo
            # Crear máscara circular para excluir
            yy, xx = np.ogrid[:imagen_np.shape[0], :imagen_np.shape[1]]
            dist = np.sqrt((xx - x)**2 + (yy - y)**2)
            mascara_circular = dist <= (r * 1.2)  # 20% más grande por seguridad
            mascara_limpia = np.logical_and(mascara_limpia, ~mascara_circular)
    
    return mascara_limpia

def expandir_a_bordes(mascara, imagen_np, punto_clic):
    """Expande la máscara hasta los bordes reales de la pared"""
    x, y = punto_clic
    alto, ancho = mascara.shape
    
    # Obtener color de referencia
    color_ref = imagen_np[y, x].astype(float)
    
    # Convertir imagen a LAB para mejor comparación
    img_lab = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2LAB)
    color_ref_lab = cv2.cvtColor(np.uint8([[color_ref]]), cv2.COLOR_RGB2LAB)[0, 0]
    
    # Calcular similitud de color
    diferencia = np.sqrt(np.sum((img_lab - color_ref_lab) ** 2, axis=2))
    
    # Máscara de píxeles similares (tolerancia generosa)
    umbral = np.percentile(diferencia[mascara], 80)
    mascara_expandida = diferencia < (umbral * 1.8)
    
    # Detectar bordes arquitectónicos fuertes (techo, piso, esquinas)
    gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
    bordes = cv2.Canny(gris, 30, 100)
    
    # Dilatar bordes para crear barreras
    kernel = np.ones((5, 5), np.uint8)
    bordes_dilatados = cv2.dilate(bordes, kernel, iterations=2)
    
    # Combinar: similitud de color PERO respetar bordes fuertes
    mascara_combinada = np.logical_and(mascara_expandida, bordes_dilatados == 0)
    
    # Mantener solo región conectada al punto de clic
    etiquetas, _ = ndimage.label(mascara_combinada)
    if etiquetas[y, x] > 0:
        mascara_final = (etiquetas == etiquetas[y, x])
    else:
        mascara_final = mascara_combinada
    
    # Rellenar huecos internos
    mascara_final = ndimage.binary_fill_holes(mascara_final)
    
    # Operación de cierre para suavizar y conectar áreas cercanas
    estructura = np.ones((7, 7))
    mascara_final = ndimage.binary_closing(mascara_final, structure=estructura, iterations=3)
    
    return mascara_final

def refinar_bordes(mascara, imagen_np):
    """Refina los bordes de la máscara para mayor precisión"""
    # Suavizar bordes ligeramente
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_uint8 = mascara.astype(np.uint8) * 255
    
    # Operación morfológica para suavizar
    mascara_suavizada = cv2.morphologyEx(mascara_uint8, cv2.MORPH_CLOSE, kernel)
    mascara_suavizada = cv2.morphologyEx(mascara_suavizada, cv2.MORPH_OPEN, kernel)
    
    # Aplicar filtro bilateral para bordes más naturales
    mascara_refinada = cv2.bilateralFilter(mascara_suavizada, 9, 75, 75)
    
    return mascara_refinada > 127

def filtrar_sombras_objetos(imagen_np, mascara, punto_clic):
    """Distingue entre sombras de la pared y objetos reales"""
    x, y = punto_clic
    
    # Convertir a espacio HSV
    img_hsv = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2HSV)
    
    # Obtener valor de referencia (luminosidad)
    v_ref = img_hsv[y, x, 2]
    
    # Las sombras mantienen el matiz similar pero bajan el valor
    # Los objetos cambian el matiz
    h_imagen = img_hsv[:, :, 0]
    v_imagen = img_hsv[:, :, 2]
    
    h_ref = img_hsv[y, x, 0]
    
    # Diferencia de matiz (objetos diferentes)
    diff_h = np.abs(h_imagen.astype(float) - h_ref.astype(float))
    diff_h = np.minimum(diff_h, 180 - diff_h)  # Circular en HSV
    
    # Objetos tienen diferencia de matiz > 15
    es_objeto = diff_h > 15
    
    # Excluir objetos de la máscara
    mascara_sin_objetos = np.logical_and(mascara, ~es_objeto)
    
    return mascara_sin_objetos

def aplicar_color_realista(imagen_np, mascara, color_hex):
    """Aplica color preservando iluminación y textura"""
    color_rgb = np.array([int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])
    
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(imagen_np)
    img_hsv = np.array(img_pil.convert('HSV'))
    
    color_pil = PILImage.new('RGB', (1, 1), tuple(color_rgb))
    color_hsv = np.array(color_pil.convert('HSV'))[0, 0]
    
    # Aplicar solo matiz y saturación, mantener luminosidad
    img_hsv[mascara, 0] = color_hsv[0]
    img_hsv[mascara, 1] = int(color_hsv[1] * 0.80)  # 80% saturación
    
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

st.title("🎨 Simulador de Pintura Profesional v2.0")
st.markdown("**IA Avanzada: Detecta objetos 3D, expande a esquinas, filtra sombras**")

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
        st.markdown("**Preview:**")
        st.markdown(f'<div style="background:{color}; height:50px; border-radius:10px; border:3px solid #333; box-shadow: 3px 3px 10px rgba(0,0,0,0.3);"></div>', unsafe_allow_html=True)
    
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
        st.success(f"✅ {len(st.session_state.paredes)} pared(es) - Detección avanzada aplicada")
    else:
        img_mostrar = img
        st.info("👇 **HAZ CLIC** en la pared")
    
    value = streamlit_image_coordinates(img_mostrar, key="image_coords")
    
    if value is not None:
        x = value["x"]
        y = value["y"]
        
        st.success(f"📍 Clic en: ({x}, {y})")
        
        if st.button("🎨 PINTAR CON IA AVANZADA", type="primary"):
            with st.spinner("🤖 Procesando con IA avanzada..."):
                predictor = cargar_predictor()
                
                if predictor:
                    try:
                        img_np = np.array(img)
                        predictor.set_image(img_np)
                        
                        # Paso 1: Máscara inicial con SAM
                        masks, scores, _ = predictor.predict(
                            point_coords=np.array([[x, y]]),
                            point_labels=np.array([1]),
                            multimask_output=True,
                        )
                        
                        mejor = np.argmax(scores)
                        mascara_inicial = masks[mejor]
                        
                        # Paso 2: DETECTAR Y EXCLUIR OBJETOS 3D (ventiladores)
                        mascara_sin_objetos3d = detectar_objetos_3d(img_np, mascara_inicial)
                        
                        # Paso 3: EXPANDIR A BORDES/ESQUINAS
                        mascara_expandida = expandir_a_bordes(mascara_sin_objetos3d, img_np, (x, y))
                        
                        # Paso 4: FILTRAR SOMBRAS DE OBJETOS
                        mascara_sin_sombras = filtrar_sombras_objetos(img_np, mascara_expandida, (x, y))
                        
                        # Paso 5: REFINAR BORDES
                        mascara_final = refinar_bordes(mascara_sin_sombras, img_np)
                        
                        # Guardar
                        st.session_state.paredes.append({
                            'mask': mascara_final,
                            'color': color
                        })
                        
                        st.success(f"🎉 ¡Pared #{len(st.session_state.paredes)} procesada con IA avanzada!")
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.exception(e)
    
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
### 🚀 Mejoras IA v2.0:
✅ **Detección de objetos 3D** - Excluye ventiladores, lámparas circulares  
✅ **Expansión inteligente** - Cubre hasta esquinas y bordes  
✅ **Filtro de sombras** - Distingue sombras de objetos reales  
✅ **Refinamiento de bordes** - Transiciones suaves y naturales  
✅ **Un solo clic** - Como Gemini pero mejor

📊 **Pipeline:** SAM → Objetos 3D → Expansión → Sombras → Bordes → Resultado
""")

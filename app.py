import streamlit as st
import numpy as np
from PIL import Image
import torch
from streamlit_image_coordinates import streamlit_image_coordinates
import os
import requests
from scipy import ndimage

# Configuración
st.set_page_config(page_title="Simulador de Pintura", layout="wide")

# Función MEJORADA para rellenar solo huecos pequeños SIN expandir
def mejorar_mascara(mascara):
    """Rellena solo huecos internos pequeños, NO expande bordes"""
    # Solo rellenar huecos completamente rodeados (binary_fill_holes)
    mascara_mejorada = ndimage.binary_fill_holes(mascara)
    
    # NO hacer dilatación para evitar pintar objetos cercanos
    # Solo una pequeña operación de cierre para suavizar bordes internos
    estructura = np.ones((3, 3))
    mascara_mejorada = ndimage.binary_closing(mascara_mejorada, structure=estructura, iterations=1)
    
    return mascara_mejorada

# Función para aplicar color REALISTA preservando sombras
def aplicar_color_realista(imagen_np, mascara, color_hex):
    """Aplica color preservando la iluminación y sombras originales"""
    color_rgb = np.array([int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])
    
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(imagen_np)
    img_hsv = np.array(img_pil.convert('HSV'))
    
    color_pil = PILImage.new('RGB', (1, 1), tuple(color_rgb))
    color_hsv = np.array(color_pil.convert('HSV'))[0, 0]
    
    # Aplicar solo matiz y saturación, mantener luminosidad
    img_hsv[mascara, 0] = color_hsv[0]
    img_hsv[mascara, 1] = int(color_hsv[1] * 0.9)  # 90% saturación para más natural
    
    img_resultado_pil = PILImage.fromarray(img_hsv, mode='HSV').convert('RGB')
    return np.array(img_resultado_pil)

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
if 'modo' not in st.session_state:
    st.session_state.modo = 'pintar'

# Título
st.title("🎨 Simulador de Pintura Profesional")
st.markdown("**Visualiza colores realistas en tus paredes**")

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
    
    # Controles
    st.markdown("---")
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        st.markdown("**🎨 Color:**")
        color = st.color_picker("", "#8FBC8F", label_visibility="collapsed")
    
    with col2:
        st.markdown("**Preview:**")
        st.markdown(f'<div style="background:{color}; height:50px; border-radius:10px; border:3px solid #333;"></div>', unsafe_allow_html=True)
    
    with col3:
        if st.button("🖌️" if st.session_state.modo == 'borrar' else "🧹"):
            st.session_state.modo = 'borrar' if st.session_state.modo == 'pintar' else 'pintar'
            st.rerun()
    
    with col4:
        if st.session_state.paredes:
            if st.button("🗑️"):
                st.session_state.paredes = []
                st.rerun()
    
    # Indicador de modo
    if st.session_state.modo == 'borrar':
        st.warning("🧹 **BORRADOR** - Clic para quitar pintura")
    else:
        st.info("🖌️ **PINTAR** - Clic en la pared que quieres pintar")
    
    # Mostrar resultado
    st.markdown("---")
    if st.session_state.paredes:
        img_resultado = st.session_state.imagen_original.copy()
        
        for pared in st.session_state.paredes:
            img_resultado = aplicar_color_realista(img_resultado, pared['mask'], pared['color'])
        
        img_mostrar = Image.fromarray(img_resultado)
        st.markdown(f"**✅ {len(st.session_state.paredes)} pared(es)**")
    else:
        img_mostrar = img
    
    # Capturar clic
    value = streamlit_image_coordinates(img_mostrar, key="image_coords")
    
    # Procesar clic
    if value is not None:
        x = value["x"]
        y = value["y"]
        
        st.success(f"📍 ({x}, {y})")
        
        if st.session_state.modo == 'pintar':
            if st.button("🎨 PINTAR", type="primary"):
                with st.spinner("🤖 Detectando..."):
                    predictor = cargar_predictor()
                    
                    if predictor:
                        try:
                            img_np = np.array(img)
                            predictor.set_image(img_np)
                            
                            # Generar 3 máscaras y elegir por SCORE (la IA decide)
                              masks, scores, _ = predictor.predict(
                              point_coords=np.array([[x, y]]),
                              point_labels=np.array([1]),
                              multimask_output=True,
                             )

                            # Elegir por SCORE (mejor detección según la IA)
                              mejor = np.argmax(scores)
                              mask = masks[mejor]
                            
                            # Mejorar máscara (solo rellenar huecos internos)
                            mask_mejorada = mejorar_mascara(mask)
                            
                            # Guardar
                            st.session_state.paredes.append({
                                'mask': mask_mejorada,
                                'color': color
                            })
                            
                            st.success(f"✅ Pintada!")
                            st.balloons()
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        else:  # Borrar
            if st.button("🧹 QUITAR", type="secondary"):
                for i, pared in enumerate(st.session_state.paredes):
                    if pared['mask'][y, x]:
                        st.session_state.paredes.pop(i)
                        st.success(f"✅ Borrado!")
                        st.rerun()
                        break
    
    # Lista
    if st.session_state.paredes:
        st.markdown("---")
        st.markdown("### 📋 Paredes:")
        for i, p in enumerate(st.session_state.paredes):
            col_a, col_b, col_c = st.columns([1, 3, 1])
            with col_a:
                st.markdown(f"**{i+1}**")
            with col_b:
                st.markdown(f'<div style="background:{p["color"]}; height:40px; border-radius:5px; border:2px solid #000;"></div>', unsafe_allow_html=True)
            with col_c:
                if st.button("❌", key=f"d{i}"):
                    st.session_state.paredes.pop(i)
                    st.rerun()

else:
    st.info("👆 Sube una foto")

st.markdown("---")
st.markdown("""
### 📖 Uso:
1. Sube foto
2. Elige color
3. Clic en pared
4. "PINTAR"
5. Usa 🧹 para borrar errores

💡 Ahora detecta la máscara más grande = pared completa
""")

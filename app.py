import streamlit as st
import numpy as np
from PIL import Image
import torch
from streamlit_image_coordinates import streamlit_image_coordinates
import os
import requests

# Configuración
st.set_page_config(page_title="Simulador de Pintura", layout="wide")

# Función para aplicar color REALISTA preservando sombras
def aplicar_color_realista(imagen_np, mascara, color_hex):
    """Aplica color preservando la iluminación y sombras originales"""
    # Convertir hex a RGB
    color_rgb = np.array([int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])
    
    # Convertir imagen a HSV para preservar luminosidad
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(imagen_np)
    img_hsv = np.array(img_pil.convert('HSV'))
    
    # Convertir color objetivo a HSV
    color_pil = PILImage.new('RGB', (1, 1), tuple(color_rgb))
    color_hsv = np.array(color_pil.convert('HSV'))[0, 0]
    
    # Aplicar solo matiz y saturación, mantener luminosidad original
    img_hsv[mascara, 0] = color_hsv[0]  # Hue (matiz)
    img_hsv[mascara, 1] = color_hsv[1]  # Saturation (saturación)
    # img_hsv[mascara, 2] se mantiene = luminosidad original
    
    # Convertir de vuelta a RGB
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
    
    # Selector de color
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("**🎨 Elige tu color:**")
        color = st.color_picker("", "#8FBC8F", label_visibility="collapsed")
    
    with col2:
        # Mostrar preview del color
        st.markdown("**Vista previa:**")
        st.markdown(f'<div style="background:{color}; height:50px; border-radius:10px; border:3px solid #333; box-shadow:2px 2px 5px rgba(0,0,0,0.3);"></div>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.paredes:
            if st.button("🗑️ Borrar"):
                st.session_state.paredes = []
                st.rerun()
    
    # Mostrar resultado actual
    st.markdown("---")
    if st.session_state.paredes:
        img_resultado = st.session_state.imagen_original.copy()
        
        # Aplicar cada pared con color realista
        for pared in st.session_state.paredes:
            img_resultado = aplicar_color_realista(img_resultado, pared['mask'], pared['color'])
        
        img_mostrar = Image.fromarray(img_resultado)
        st.markdown(f"**✅ {len(st.session_state.paredes)} pared(es) pintada(s) - Vista REALISTA**")
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
        
        st.success(f"📍 Punto seleccionado: ({x}, {y})")
        
        if st.button("🎨 PINTAR ESTA PARED", type="primary"):
            with st.spinner("🤖 Detectando pared con IA..."):
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
                        
                        st.success(f"✅ Pared {len(st.session_state.paredes)} pintada!")
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Lista de paredes
    if st.session_state.paredes:
        st.markdown("---")
        st.markdown("### 📋 Paredes Pintadas:")
        for i, p in enumerate(st.session_state.paredes):
            col_a, col_b, col_c = st.columns([1, 3, 1])
            with col_a:
                st.markdown(f"**#{i+1}**")
            with col_b:
                st.markdown(f'<div style="background:{p["color"]}; height:40px; border-radius:5px; border:2px solid black;"></div>', unsafe_allow_html=True)
            with col_c:
                if st.button("❌", key=f"del{i}"):
                    st.session_state.paredes.pop(i)
                    st.rerun()

else:
    st.info("👆 Sube una foto de tu habitación")
    st.markdown("**💡 Ejemplo:** Sube una foto bien iluminada de tu espacio para mejores resultados")

st.markdown("---")
st.markdown("""
### 📖 Cómo usar:
1. **Sube una foto** de tu habitación bien iluminada
2. **Elige un color** que te guste
3. **Haz clic** directamente en la pared que quieres pintar
4. **Presiona** "PINTAR ESTA PARED"
5. **Repite** para probar diferentes colores en otras paredes
6. **Compara** y elige la combinación perfecta

💡 **Tip:** El color se aplica preservando las sombras naturales para un resultado realista
""")
                

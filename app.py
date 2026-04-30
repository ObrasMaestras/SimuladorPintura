import streamlit as st
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
import torch
import os
import requests

# Configuración
st.set_page_config(page_title="Simulador de Pintura Profesional", layout="wide")

# ============ TÉCNICA 1: MODO MULTIPLICAR (BLENDING MODE) ============
def aplicar_color_multiplicar(imagen_np, mascara, color_hex, opacidad=0.4):
    """
    Aplica color usando MODO MULTIPLICAR como Photoshop
    Preserva texturas, sombras y luces originales
    """
    # Convertir color hex a RGB
    color_rgb = np.array([int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)], dtype=float)
    
    # Normalizar a 0-1
    imagen_float = imagen_np.astype(float) / 255.0
    color_float = color_rgb / 255.0
    
    # MODO MULTIPLICAR: multiplicar cada pixel por el color
    imagen_multiplicada = imagen_float.copy()
    for i in range(3):  # RGB
        imagen_multiplicada[:, :, i] = imagen_float[:, :, i] * color_float[i]
    
    # ALPHA BLENDING: mezclar original con multiplicado usando opacidad
    imagen_final = imagen_float.copy()
    for i in range(3):
        imagen_final[mascara, i] = (
            imagen_float[mascara, i] * (1 - opacidad) +
            imagen_multiplicada[mascara, i] * opacidad
        )
    
    # Convertir de vuelta a 0-255
    return (imagen_final * 255).astype(np.uint8)

# ============ TÉCNICA 2: MÁSCARA PRECISA CON MOBILESAM ============
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

st.title("🎨 Simulador de Pintura - Técnica Profesional")
st.markdown("**Modo Multiplicar + Alpha Blending (como Photoshop/Gemini)**")

# Controles superiores
col_opacidad, col_info = st.columns([1, 2])
with col_opacidad:
    opacidad = st.slider("💧 Opacidad del color", 0.1, 1.0, 0.4, 0.05, 
                         help="40% = Realista (recomendado). 100% = Color sólido")

with col_info:
    st.info(f"✨ **Opacidad actual: {int(opacidad*100)}%** - A menor opacidad, más textura natural se preserva")

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
        st.markdown("**Preview del color:**")
        st.markdown(f'<div style="background:{color}; height:50px; border-radius:10px; border:3px solid #333; box-shadow: 3px 3px 10px rgba(0,0,0,0.3);"></div>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.paredes:
            if st.button("🗑️ Borrar"):
                st.session_state.paredes = []
                st.rerun()
    
    st.markdown("---")
    
    if st.session_state.paredes:
        img_resultado = st.session_state.imagen_original.copy()
        for pared in st.session_state.paredes:
            img_resultado = aplicar_color_multiplicar(
                img_resultado, 
                pared['mask'], 
                pared['color'],
                pared['opacidad']
            )
        img_mostrar = Image.fromarray(img_resultado)
        st.success(f"✅ {len(st.session_state.paredes)} pared(es) - Modo Multiplicar aplicado")
    else:
        img_mostrar = img
        st.info("👇 **HAZ UN CLIC** en la pared que quieres pintar")
    
    value = streamlit_image_coordinates(img_mostrar, key="image_coords")
    
    if value is not None:
        x = value["x"]
        y = value["y"]
        
        st.success(f"📍 Punto seleccionado: ({x}, {y})")
        
        if st.button("🎨 PINTAR CON MODO MULTIPLICAR", type="primary"):
            with st.spinner("🤖 MobileSAM detectando pared..."):
                predictor = cargar_predictor()
                
                if predictor:
                    try:
                        img_np = np.array(img)
                        predictor.set_image(img_np)
                        
                        # Generar múltiples máscaras y elegir la mejor
                        masks, scores, _ = predictor.predict(
                            point_coords=np.array([[x, y]]),
                            point_labels=np.array([1]),
                            multimask_output=True,  # Genera 3 opciones
                        )
                        
                        # Elegir máscara con mejor score
                        mejor_idx = np.argmax(scores)
                        mascara = masks[mejor_idx]
                        
                        # Guardar con opacidad actual
                        st.session_state.paredes.append({
                            'mask': mascara,
                            'color': color,
                            'opacidad': opacidad
                        })
                        
                        st.success(f"🎉 ¡Pared #{len(st.session_state.paredes)} pintada con Modo Multiplicar!")
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    if st.session_state.paredes:
        st.markdown("---")
        st.markdown("### 📋 Paredes Pintadas:")
        for i, p in enumerate(st.session_state.paredes):
            col_a, col_b, col_c, col_d = st.columns([1, 2, 2, 1])
            with col_a:
                st.markdown(f"**{i+1}**")
            with col_b:
                st.markdown(f'<div style="background:{p["color"]}; height:40px; border-radius:8px; border:2px solid #000;"></div>', unsafe_allow_html=True)
            with col_c:
                st.markdown(f"*Opacidad: {int(p['opacidad']*100)}%*")
            with col_d:
                if st.button("❌", key=f"d{i}"):
                    st.session_state.paredes.pop(i)
                    st.rerun()

else:
    st.info("👆 Sube una foto de tu habitación")

st.markdown("---")
st.markdown("""
### 🚀 Técnicas Profesionales Implementadas:

#### 1️⃣ **Modo Multiplicar (Multiply Blending)**
- El color se "fusiona" con la textura original
- Las sombras se oscurecen naturalmente
- Las luces se preservan realísticamente

#### 2️⃣ **Alpha Blending (Opacidad)**
- **40% (recomendado)**: Máximo realismo, la textura atraviesa
- **70%**: Color más intenso pero natural
- **100%**: Color sólido (menos realista)

#### 3️⃣ **MobileSAM - Segmentación Precisa**
- Detecta bordes automáticamente
- No pinta cuadros, muebles o techos
- Un solo clic = pared completa

💡 **Resultado:** Idéntico a edición profesional de Photoshop
""")

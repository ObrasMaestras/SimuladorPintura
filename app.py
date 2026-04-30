import streamlit as st
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import torch
import os
import requests
from scipy import ndimage

st.set_page_config(page_title="Simulador de Pintura con Alpha Blending", layout="wide")

def aplicar_alpha_blending(imagen_np, mascara, color_hex, alpha_color=0.4):
    """
    Alpha Blending: 0.6 original + 0.4 color
    Preserva textura, sombras y brillos
    """
    # Convertir hex a RGB
    color_rgb = np.array([int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)], dtype=float)
    
    # Crear copia de la imagen
    resultado = imagen_np.copy().astype(float)
    
    # Alpha Blending: original * (1 - alpha) + color * alpha
    alpha_original = 1.0 - alpha_color
    
    for canal in range(3):  # RGB
        resultado[mascara, canal] = (
            imagen_np[mascara, canal] * alpha_original +
            color_rgb[canal] * alpha_color
        )
    
    return resultado.astype(np.uint8)

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

st.title("🎨 Simulador de Pintura con Alpha Blending")
st.markdown("**Alpha Blending: 60% original + 40% color = Textura visible**")

# Control de alpha
st.markdown("---")
col_alpha, col_info = st.columns([1, 2])
with col_alpha:
    alpha_color = st.slider("🎨 Proporción de color", 0.2, 0.8, 0.4, 0.05,
                           help="0.4 = 40% color + 60% original (recomendado)")
with col_info:
    st.info(f"✨ Alpha: {alpha_color:.1f} color + {1-alpha_color:.1f} original = Preserva textura")

archivo = st.file_uploader("📸 Sube tu foto", type=["jpg", "jpeg", "png"])

if archivo:
    img = Image.open(archivo).convert("RGB")
    
    # Redimensionar
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
        st.markdown("**Vista previa del color:**")
        st.markdown(f'<div style="background:{color}; height:50px; border-radius:10px; border:3px solid #333; margin-top:8px; box-shadow:3px 3px 10px rgba(0,0,0,0.3);"></div>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.paredes:
            if st.button("🗑️ Borrar"):
                st.session_state.paredes = []
                st.rerun()
    
    st.markdown("---")
    
    # Mostrar resultado
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
        st.success(f"✅ {len(st.session_state.paredes)} pared(es) - Alpha Blending aplicado")
    else:
        img_mostrar = img
        st.info("👇 **HAZ UN CLIC** en la pared que quieres pintar")
    
    # Capturar clic
    value = streamlit_image_coordinates(img_mostrar, key="image_coords")
    
    if value is not None:
        x = value["x"]
        y = value["y"]
        
        st.success(f"📍 Punto seleccionado: ({x}, {y})")
        
        if st.button("🎨 PINTAR CON ALPHA BLENDING", type="primary"):
            with st.spinner("🤖 MobileSAM segmentando..."):
                predictor = cargar_predictor()
                
                if predictor:
                    try:
                        img_np = np.array(img)
                        predictor.set_image(img_np)
                        
                        # Generar máscaras con MobileSAM
                        masks, scores, _ = predictor.predict(
                            point_coords=np.array([[x, y]]),
                            point_labels=np.array([1]),
                            multimask_output=True,  # Genera 3 máscaras
                        )
                        
                        # USAR MÁSCARA DE MAYOR CONFIANZA (score más alto)
                        mejor_idx = np.argmax(scores)
                        mascara_mejor = masks[mejor_idx]
                        
                        st.info(f"🎯 Máscara seleccionada: Score {scores[mejor_idx]:.3f} (la de mayor confianza)")
                        
                        # Rellenar huecos internos para respetar bordes de objetos
                        mascara_final = ndimage.binary_fill_holes(mascara_mejor)
                        
                        # Guardar
                        st.session_state.paredes.append({
                            'mask': mascara_final,
                            'color': color,
                            'alpha': alpha_color
                        })
                        
                        st.success(f"🎉 ¡Pared #{len(st.session_state.paredes)} pintada con Alpha Blending!")
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.exception(e)
    
    # Lista de paredes
    if st.session_state.paredes:
        st.markdown("---")
        st.markdown("### 📋 Paredes Pintadas:")
        for i, p in enumerate(st.session_state.paredes):
            col_a, col_b, col_c, col_d = st.columns([1, 2, 2, 1])
            with col_a:
                st.markdown(f"**Pared {i+1}**")
            with col_b:
                st.markdown(f'<div style="background:{p["color"]}; height:40px; border-radius:8px; border:2px solid #000;"></div>', unsafe_allow_html=True)
            with col_c:
                st.markdown(f"*Alpha: {p['alpha']:.1f}*")
            with col_d:
                if st.button("❌", key=f"del{i}"):
                    st.session_state.paredes.pop(i)
                    st.rerun()

else:
    st.info("👆 Sube una foto de tu habitación")

st.markdown("---")
st.markdown("### 🚀 Especificaciones Técnicas:")
st.markdown("#### 1️⃣ Alpha Blending (0.6 original + 0.4 color)")
st.code("pixel_final = pixel_original × 0.6 + color_hex × 0.4", language="python")
st.markdown("- ✅ Preserva **60% de la textura original**")
st.markdown("- ✅ Aplica **40% del color nuevo**")
st.markdown("- ✅ Sombras, brillos y granulado visibles")

st.markdown("#### 2️⃣ MobileSAM - Máscara de Mayor Confianza")
st.markdown("- Genera 3 máscaras candidatas (multimask_output=True)")
st.markdown("- Selecciona automáticamente la de **mayor score**")
st.markdown("- Rellena huecos internos (binary_fill_holes)")
st.markdown("- Respeta bordes de objetos")

st.markdown("#### 3️⃣ Renderizado Realista")
st.markdown("- NO es color sólido opaco")
st.markdown("- La textura de la pared atraviesa el color")
st.markdown("- Efecto como pintura real translúcida")

st.markdown("💡 **Resultado:** Simulación fotorealista de pintura profesional")

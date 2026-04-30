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

# Función para mejorar la máscara (rellenar huecos y limpiar bordes)
def mejorar_mascara(mascara):
    """Rellena huecos pequeños y suaviza bordes"""
    # Rellenar huecos pequeños (hasta 50 píxeles)
    mascara_mejorada = ndimage.binary_fill_holes(mascara)
    
    # Operación de cierre para conectar áreas cercanas
    estructura = ndimage.generate_binary_structure(2, 2)
    mascara_mejorada = ndimage.binary_closing(mascara_mejorada, structure=estructura, iterations=3)
    
    # Pequeña dilatación para cubrir bordes mejor
    mascara_mejorada = ndimage.binary_dilation(mascara_mejorada, structure=estructura, iterations=2)
    
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
    img_hsv[mascara, 1] = color_hsv[1]
    
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
        st.markdown("**🎨 Elige tu color:**")
        color = st.color_picker("", "#8FBC8F", label_visibility="collapsed")
    
    with col2:
        st.markdown("**Vista previa:**")
        st.markdown(f'<div style="background:{color}; height:50px; border-radius:10px; border:3px solid #333;"></div>', unsafe_allow_html=True)
    
    with col3:
        # Modo pintar / borrar
        if st.button("🖌️ Pintar" if st.session_state.modo == 'borrar' else "🧹 Borrar"):
            st.session_state.modo = 'borrar' if st.session_state.modo == 'pintar' else 'pintar'
            st.rerun()
    
    with col4:
        if st.session_state.paredes:
            if st.button("🗑️ Todo"):
                st.session_state.paredes = []
                st.rerun()
    
    # Indicador de modo
    if st.session_state.modo == 'borrar':
        st.warning("🧹 **MODO BORRADOR** - Haz clic en áreas para quitar pintura")
    else:
        st.info("🖌️ **MODO PINTAR** - Haz clic en paredes para pintarlas")
    
    # Mostrar resultado actual
    st.markdown("---")
    if st.session_state.paredes:
        img_resultado = st.session_state.imagen_original.copy()
        
        for pared in st.session_state.paredes:
            img_resultado = aplicar_color_realista(img_resultado, pared['mask'], pared['color'])
        
        img_mostrar = Image.fromarray(img_resultado)
        st.markdown(f"**✅ {len(st.session_state.paredes)} pared(es) pintada(s)**")
    else:
        img_mostrar = img
        st.markdown("**👇 HAZ CLIC EN LA PARED:**")
    
    # Capturar clic en la imagen
    value = streamlit_image_coordinates(
        img_mostrar,
        key="image_coords"
    )
    
    # Procesar clic
    if value is not None:
        x = value["x"]
        y = value["y"]
        
        st.success(f"📍 Clic en: ({x}, {y})")
        
        if st.session_state.modo == 'pintar':
            if st.button("🎨 PINTAR ESTA PARED", type="primary"):
                with st.spinner("🤖 Detectando pared..."):
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
                            
                            # MEJORAR LA MÁSCARA (rellenar huecos)
                            mask_mejorada = mejorar_mascara(mask)
                            
                            # Guardar
                            st.session_state.paredes.append({
                                'mask': mask_mejorada,
                                'color': color
                            })
                            
                            st.success(f"✅ Pared {len(st.session_state.paredes)} pintada!")
                            st.balloons()
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        else:  # Modo borrar
            if st.button("🧹 QUITAR PINTURA AQUÍ", type="secondary"):
                # Buscar qué pared contiene este punto y eliminarla
                for i, pared in enumerate(st.session_state.paredes):
                    if pared['mask'][y, x]:
                        st.session_state.paredes.pop(i)
                        st.success(f"✅ Área borrada!")
                        st.rerun()
                        break
                else:
                    st.warning("⚠️ No hay pintura en ese punto")
    
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

st.markdown("---")
st.markdown("""
### 📖 Cómo usar:
1. **Sube una foto** bien iluminada
2. **Elige un color**
3. **Haz clic** en la pared
4. **Presiona** "PINTAR ESTA PARED"
5. **Modo borrador** para quitar áreas incorrectas
6. **Repite** para más paredes

💡 **Mejoras:** Rellena huecos automáticamente y modo borrador para correcciones
""")

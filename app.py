import streamlit as st
import numpy as np
from PIL import Image
import torch
from streamlit_drawable_canvas import st_canvas
import os
import requests

# Configuración de la página
st.set_page_config(page_title="Simulador de Pintura", layout="wide", initial_sidebar_state="collapsed")

# Función para descargar el modelo MobileSAM
@st.cache_data
def descargar_modelo():
    """Descarga el modelo MobileSAM si no existe localmente"""
    modelo_path = "mobile_sam.pt"
    if not os.path.exists(modelo_path):
        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        with st.spinner("⏳ Descargando modelo MobileSAM..."):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(modelo_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                st.error(f"❌ Error descargando el modelo: {e}")
                return None
    return modelo_path

# Función para cargar el predictor
@st.cache_resource
def cargar_predictor():
    """Carga el modelo MobileSAM en memoria"""
    try:
        from mobile_sam import sam_model_registry, SamPredictor
        modelo_path = descargar_modelo()
        if modelo_path is None:
            return None
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        modelo = sam_model_registry["vit_t"](checkpoint=modelo_path)
        modelo.to(device=device)
        modelo.eval()
        predictor = SamPredictor(modelo)
        return predictor
    except Exception as e:
        st.error(f"❌ Error cargando el predictor: {e}")
        return None

# Inicializar session_state
if 'areas_pintadas' not in st.session_state:
    st.session_state.areas_pintadas = []
if 'imagen_base' not in st.session_state:
    st.session_state.imagen_base = None

# Título
st.title("🎨 Simulador de Pintura para Paredes")
st.markdown("### Selecciona paredes y prueba diferentes colores")

# Layout
archivo_subido = st.file_uploader("📸 Sube una foto de tu habitación", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    # Cargar y preparar imagen
    imagen = Image.open(archivo_subido).convert("RGB")
    
    # Redimensionar para móvil
    max_size = 700
    if max(imagen.size) > max_size:
        ratio = max_size / max(imagen.size)
        nuevo_ancho = int(imagen.size[0] * ratio)
        nuevo_alto = int(imagen.size[1] * ratio)
        imagen = imagen.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
    
    # Guardar imagen base
    if st.session_state.imagen_base is None:
        st.session_state.imagen_base = np.array(imagen)
    
    ancho, alto = imagen.size
    
    # Selector de color GRANDE
    st.markdown("---")
    col_color, col_boton = st.columns([2, 1])
    
    with col_color:
        st.markdown("### 🎨 Elige el color:")
        color_actual = st.color_picker("", "#3498db", key="color_picker", label_visibility="collapsed")
        st.markdown(f'<div style="background:{color_actual}; height:60px; border-radius:10px; border:3px solid #333;"></div>', unsafe_allow_html=True)
    
    with col_boton:
        st.markdown("### ")
        st.markdown("### ")
        if st.session_state.areas_pintadas:
            if st.button("🔄 BORRAR TODO"):
                st.session_state.areas_pintadas = []
                st.rerun()
    
    # Mostrar imagen actual o resultado
    st.markdown("---")
    st.markdown("### 📸 Tu Habitación:")
    
    if st.session_state.areas_pintadas:
        # Mostrar imagen con pinturas aplicadas
        imagen_con_pinturas = st.session_state.imagen_base.copy()
        for area in st.session_state.areas_pintadas:
            color_rgb = tuple(int(area['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            imagen_con_pinturas[area['mascara']] = color_rgb
        
        st.image(imagen_con_pinturas, use_column_width=True, caption=f"{len(st.session_state.areas_pintadas)} área(s) pintada(s)")
    else:
        st.image(imagen, use_column_width=True, caption="Imagen original")
    
    # Canvas simplificado
    st.markdown("---")
    st.markdown("### 👇 HAZ CLIC EN LA PARED QUE QUIERES PINTAR:")
    st.info("💡 Toca UNA VEZ en la pared, luego presiona 'PINTAR'")
    
    # Canvas con imagen de fondo
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=0,
        stroke_color="rgba(0, 0, 0, 0)",
        background_image=Image.fromarray(st.session_state.imagen_base),
        update_streamlit=True,
        height=alto,
        width=ancho,
        drawing_mode="point",
        point_display_radius=15,
        key="canvas_simple",
    )
    
    # Procesar clic
    if canvas_result.json_data is not None:
        objetos = canvas_result.json_data.get("objects", [])
        
        if len(objetos) > 0:
            # Tomar SOLO el primer punto
            punto = objetos[0]
            x = int(punto.get("left", 0))
            y = int(punto.get("top", 0))
            
            st.success(f"✅ Punto seleccionado en la imagen: ({x}, {y})")
            
            # Botón grande para pintar
            if st.button("🎨 PINTAR ESTA PARED", type="primary"):
                with st.spinner("🤖 Detectando la pared con inteligencia artificial..."):
                    predictor = cargar_predictor()
                    
                    if predictor is not None:
                        try:
                            imagen_np = np.array(imagen)
                            
                            # Configurar predictor
                            predictor.set_image(imagen_np)
                            punto_input = np.array([[x, y]])
                            etiqueta_input = np.array([1])
                            
                            # Generar máscara
                            mascaras, scores, _ = predictor.predict(
                                point_coords=punto_input,
                                point_labels=etiqueta_input,
                                multimask_output=True,
                            )
                            
                            # Tomar la máscara con mejor score
                            mejor_idx = np.argmax(scores)
                            mascara = mascaras[mejor_idx]
                            
                            # Guardar área pintada
                            st.session_state.areas_pintadas.append({
                                'mascara': mascara,
                                'color': color_actual,
                                'coordenadas': (x, y)
                            })
                            
                            st.success(f"🎉 ¡Pared pintada! Total: {len(st.session_state.areas_pintadas)} área(s)")
                            st.balloons()
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                    else:
                        st.error("❌ No se pudo cargar el modelo de IA")
        else:
            st.warning("👆 Toca en la imagen de arriba para seleccionar una pared")
    
    # Resumen de áreas pintadas
    if st.session_state.areas_pintadas:
        st.markdown("---")
        st.markdown("### 📋 Paredes Pintadas:")
        
        for idx, area in enumerate(st.session_state.areas_pintadas):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**Pared {idx+1}:**")
                st.markdown(f'<div style="background:{area["color"]}; height:40px; border-radius:5px; border:2px solid #000;"></div>', unsafe_allow_html=True)
            with col2:
                if st.button("❌", key=f"borrar_{idx}"):
                    st.session_state.areas_pintadas.pop(idx)
                    st.rerun()

else:
    st.info("👆 Sube una foto de tu habitación para comenzar")

# Instrucciones
st.markdown("---")
st.markdown("""
### 📖 ¿Cómo usar?
1. **Sube una foto** de tu habitación
2. **Elige un color** con el selector
3. **Toca** en una pared de la imagen
4. **Presiona** "PINTAR ESTA PARED"
5. **Repite** para más paredes con diferentes colores
6. **Compara** los resultados y encuentra tu combinación perfecta
""")
                
        
        

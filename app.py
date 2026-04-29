import streamlit as st
import numpy as np
from PIL import Image
import torch
from streamlit_drawable_canvas import st_canvas
import os
import requests

# Configuración de la página
st.set_page_config(page_title="Simulador de Pintura Pro", layout="wide")

# Función para descargar el modelo MobileSAM
@st.cache_data
def descargar_modelo():
    """Descarga el modelo MobileSAM si no existe localmente"""
    modelo_path = "mobile_sam.pt"
    if not os.path.exists(modelo_path):
        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        with st.spinner("⏳ Descargando modelo MobileSAM... (solo la primera vez)"):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(modelo_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ Modelo descargado exitosamente")
            except Exception as e:
                st.error(f"❌ Error descargando el modelo: {e}")
                return None
    return modelo_path

# Función para cargar el predictor (solo cuando se necesite)
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
if 'areas_seleccionadas' not in st.session_state:
    st.session_state.areas_seleccionadas = []
if 'imagen_actual' not in st.session_state:
    st.session_state.imagen_actual = None

# Interfaz principal
st.title("🎨 Simulador de Pintura Inteligente - Multi Color")
st.markdown("**Selecciona múltiples áreas y prueba diferentes combinaciones de colores**")

# Columnas para layout
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("⚙️ Panel de Control")
    
    archivo_subido = st.file_uploader("📁 Sube una imagen:", type=["jpg", "jpeg", "png"])
    
    st.markdown("---")
    st.subheader("🎨 Selección Actual")
    color_pintura = st.color_picker("Elige el color:", "#FF6B6B")
    
    if archivo_subido is not None:
        st.markdown("---")
        st.subheader("📋 Áreas Guardadas")
        
        if st.session_state.areas_seleccionadas:
            for idx, area in enumerate(st.session_state.areas_seleccionadas):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**Área {idx+1}** 🎨")
                    st.color_picker(f"Color área {idx+1}", area['color'], key=f"color_{idx}", disabled=True)
                with col_b:
                    if st.button("🗑️", key=f"del_{idx}"):
                        st.session_state.areas_seleccionadas.pop(idx)
                        st.rerun()
            
            st.markdown("---")
            if st.button("🔄 LIMPIAR TODO", type="secondary", use_container_width=True):
                st.session_state.areas_seleccionadas = []
                st.session_state.imagen_actual = None
                st.rerun()
                
            if st.button("✨ VER RESULTADO FINAL", type="primary", use_container_width=True):
                st.session_state.mostrar_resultado = True
                st.rerun()
        else:
            st.info("👆 Haz clic en la imagen para agregar áreas")

with col1:
    if archivo_subido is not None:
        # Cargar imagen
        imagen = Image.open(archivo_subido).convert("RGB")
        
        # Redimensionar si es muy grande
        max_size = 800
        if max(imagen.size) > max_size:
            ratio = max_size / max(imagen.size)
            ancho_nuevo = int(imagen.size[0] * ratio)
            alto_nuevo = int(imagen.size[1] * ratio)
            imagen = imagen.resize((ancho_nuevo, alto_nuevo), Image.Resampling.LANCZOS)
        
        # Guardar imagen en session_state
        if st.session_state.imagen_actual is None:
            st.session_state.imagen_actual = np.array(imagen)
        
        # Obtener dimensiones
        ancho, alto = imagen.size
        
        st.subheader("🖼️ Canvas Interactivo")
        st.info("👆 **Paso 1:** Haz clic en un área → **Paso 2:** Presiona 'AGREGAR ÁREA' → **Paso 3:** Repite para más áreas")
        
        # Canvas para capturar clics con la imagen de fondo
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=imagen,
            update_streamlit=True,
            height=alto,
            width=ancho,
            drawing_mode="point",
            point_display_radius=8,
            key="canvas",
        )
        
        # Botón para agregar área
        if canvas_result.json_data is not None:
            objetos = canvas_result.json_data.get("objects", [])
            if objetos:
                ultimo_clic = objetos[-1]
                x = int(ultimo_clic["left"])
                y = int(ultimo_clic["top"])
                
                st.success(f"📍 Punto seleccionado: ({x}, {y})")
                
                if st.button("➕ AGREGAR ESTA ÁREA CON ESTE COLOR", type="primary", use_container_width=True):
                    with st.spinner("🔍 Detectando área..."):
                        predictor = cargar_predictor()
                        
                        if predictor is not None:
                            try:
                                imagen_np = np.array(imagen)
                                
                                predictor.set_image(imagen_np)
                                punto_input = np.array([[x, y]])
                                etiqueta_input = np.array([1])
                                
                                mascaras, _, _ = predictor.predict(
                                    point_coords=punto_input,
                                    point_labels=etiqueta_input,
                                    multimask_output=False,
                                )
                                
                                mascara = mascaras[0]
                                
                                # Guardar área seleccionada
                                st.session_state.areas_seleccionadas.append({
                                    'mascara': mascara,
                                    'color': color_pintura,
                                    'coordenadas': (x, y)
                                })
                                
                                st.success(f"✅ Área #{len(st.session_state.areas_seleccionadas)} agregada!")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
        
        # Mostrar resultado final si se presionó el botón
        if hasattr(st.session_state, 'mostrar_resultado') and st.session_state.mostrar_resultado:
            if st.session_state.areas_seleccionadas:
                st.markdown("---")
                st.subheader("✨ Resultado Final - Combinación de Colores")
                
                imagen_final = st.session_state.imagen_actual.copy()
                
                # Aplicar todos los colores
                for area in st.session_state.areas_seleccionadas:
                    color_rgb = tuple(int(area['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    imagen_final[area['mascara']] = color_rgb
                
                st.image(imagen_final, caption=f"🎨 Combinación con {len(st.session_state.areas_seleccionadas)} colores", use_column_width=True)
                
                if st.button("⬇️ Descargar Resultado", type="secondary"):
                    st.info("💡 Haz clic derecho en la imagen → Guardar imagen como...")
                
                st.session_state.mostrar_resultado = False
    else:
        st.info("👈 Sube una imagen para comenzar")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Puedes seleccionar paredes, muebles, pisos... ¡Experimenta con diferentes combinaciones!")

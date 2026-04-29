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

# Interfaz principal
st.title("🎨 Simulador de Pintura Inteligente")
st.markdown("**Haz clic en cualquier objeto de la imagen para pintarlo**")

# Columnas para layout
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("⚙️ Configuración")
    color_pintura = st.color_picker("Elige el color de pintura:", "#FF6B6B")
    archivo_subido = st.file_uploader("Sube una imagen:", type=["jpg", "jpeg", "png"])

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
        
        # Convertir a numpy
        imagen_np = np.array(imagen)
        ancho, alto = imagen.size
        
        st.subheader("🖼️ Canvas Interactivo")
        st.info("👆 Haz clic en la imagen para seleccionar la zona a pintar")
        
        # Canvas para capturar clics
        canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=2,
    stroke_color="#FF0000",
    background_image=imagen,  # ← CAMBIADO A imagen (PIL Image)
    update_streamlit=True,
    height=alto,
    width=ancho,
    drawing_mode="point",
    key="canvas",
)
        
        # Procesar cuando se hace clic
        if canvas_result.json_data is not None:
            objetos = canvas_result.json_data.get("objects", [])
            if objetos and st.button("🎨 PINTAR AHORA", type="primary"):
                predictor = cargar_predictor()
                
                if predictor is None:
                    st.error("No se pudo cargar el modelo. Verifica la instalación de MobileSAM.")
                else:
                    with st.spinner("🔍 Analizando y pintando..."):
                        try:
                            ultimo_click = objetos[-1]
                            x = int(ultimo_click["left"])
                            y = int(ultimo_click["top"])
                            
                            predictor.set_image(imagen_np)
                            punto_input = np.array([[x, y]])
                            etiqueta_input = np.array([1])
                            
                            mascaras, _, _ = predictor.predict(
                                point_coords=punto_input,
                                point_labels=etiqueta_input,
                                multimask_output=False,
                            )
                            
                            mascara = mascaras[0]
                            
                            color_rgb = tuple(int(color_pintura.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                            imagen_pintada = imagen_np.copy()
                            imagen_pintada[mascara] = color_rgb
                            
                            st.success("✅ ¡Pintado exitoso!")
                            st.image(imagen_pintada, caption="Resultado Final", use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"❌ Error al procesar: {e}")
    else:
        st.info("👈 Sube una imagen para comenzar")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Funciona mejor con imágenes de interiores, muebles y objetos definidos")

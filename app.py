import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

MODEL_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
MODEL_PATH = Path("mobile_sam.pt")


@st.cache_data(show_spinner=False)
def ensure_model_file() -> str:
    """Descarga el checkpoint de MobileSAM solo si no existe."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return str(MODEL_PATH)

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "No se pudo descargar el modelo MobileSAM. "
            "Verifica conectividad y permisos del entorno."
        ) from exc

    return str(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_predictor(model_checkpoint: str):
    """Carga el predictor de MobileSAM en modo lazy (solo cuando se necesita)."""
    try:
        import torch
        from mobile_sam import SamPredictor, sam_model_registry
    except Exception as exc:
        raise RuntimeError(
            "Dependencias de MobileSAM no disponibles. "
            "Revisa requirements.txt (torch, torchvision y MobileSAM)."
        ) from exc

    sam_model = sam_model_registry["vit_t"](checkpoint=model_checkpoint)
    sam_model.to(device="cpu")
    sam_model.eval()
    torch.set_grad_enabled(False)

    return SamPredictor(sam_model)


with col1:
    if archivo_subido is not None:
        # Cargar imagen
        imagen = Image.open(archivo_subido).convert("RGB")
        
        # Redimensionar si es muy grande (para evitar lag en el canvas)
        max_size = 800
        if max(imagen.size) > max_size:
            ratio = max_size / max(imagen.size)
            nuevo_tamaño = tuple([int(dim * ratio) for dim in imagen.size])
            imagen = imagen.resize(nuevo_tamaño, Image.Resampling.LANCZOS)
        
        # Convertir a numpy AQUÍ (antes del canvas)
        imagen_np = np.array(imagen)
        ancho, alto = imagen.size
        
        st.subheader("🖼️ Canvas Interactivo")
        st.info("👆 Haz clic en la imagen para seleccionar la zona a pintar")
        
        # Canvas para capturar clics
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=imagen_np,
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
                # Aquí es donde se carga el modelo (lazy loading)
                predictor = cargar_predictor()
                
                if predictor is None:
                    st.error("No se pudo cargar el modelo. Verifica la instalación de MobileSAM.")
                else:
                    with st.spinner("🔍 Analizando y pintando..."):
                        try:
                            # Obtener coordenadas del último clic
                            ultimo_click = objetos[-1]
                            x = int(ultimo_click["left"])
                            y = int(ultimo_click["top"])
                            
                            # Generar máscara con MobileSAM
                            predictor.set_image(imagen_np)
                            punto_input = np.array([[x, y]])
                            etiqueta_input = np.array([1])
                            
                            mascaras, _, _ = predictor.predict(
                                point_coords=punto_input,
                                point_labels=etiqueta_input,
                                multimask_output=False,
                            )
                            
                            mascara = mascaras[0]
                            
                            # Aplicar color a la máscara
                            color_rgb = tuple(int(color_pintura.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                            imagen_pintada = imagen_np.copy()
                            imagen_pintada[mascara] = color_rgb
                            
                            # Mostrar resultado
                            st.success("✅ ¡Pintado exitoso!")
                            st.image(imagen_pintada, caption="Resultado Final", use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"❌ Error al procesar: {e}")
    else:
        st.info("👈 Sube una imagen para comenzar")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Funciona mejor con imágenes de interiores, muebles y objetos definidos")

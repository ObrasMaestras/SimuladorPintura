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


def paint_wall(image: Image.Image, x: int, y: int, paint_color: tuple[int, int, int]) -> np.ndarray:
    """Ejecuta segmentación por punto y aplica el color seleccionado."""
    predictor = load_predictor(ensure_model_file())

    image_np = np.array(image.convert("RGB"))
    predictor.set_image(image_np)

    masks, scores, _ = predictor.predict(
        point_coords=np.array([[x, y]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )

    if len(scores) == 0:
        raise RuntimeError("MobileSAM no devolvió máscaras válidas para ese punto.")

    best_mask = masks[int(np.argmax(scores))]

    result = image_np.copy().astype(np.float32)
    overlay = np.array(paint_color, dtype=np.float32)
    result[best_mask] = (result[best_mask] * 0.5) + (overlay * 0.5)

    return np.clip(result, 0, 255).astype(np.uint8)


st.set_page_config(page_title="Simulador de Pintura", layout="centered")
st.title("🖌️ Simulador de Pintura Pro")
st.write("Sube una foto y haz clic en la pared que quieras pintar.")

archivo = st.file_uploader("Sube tu foto", type=["jpg", "png", "jpeg"])

if archivo:
    img = Image.open(archivo).convert("RGB")
    w, h = img.size

    canvas_width = min(900, w)
    ratio = canvas_width / w
    canvas_height = int(h * ratio)
    img_res = img.resize((canvas_width, canvas_height))

    # --- CAMBIO AQUÍ ---
    # Pasamos la imagen de Pillow directamente, pero asegurándonos de que el lienzo 
    # la trate como el fondo correcto. 
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        background_image=imagen_np,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="point",
        point_display_radius=6,
        key="canvas",
        display_toolbar=True, # Recomendado para que puedas deshacer clics
    )
    # --- FIN DEL CAMBIO ---

    color_hex = st.color_picker("Color de pintura", "#647864")
    paint_rgb = tuple(int(color_hex[i : i + 2], 16) for i in (1, 3, 5))

    objects = []
    if canvas_result.json_data:
        objects = canvas_result.json_data.get("objects", []) or []

    has_point = len(objects) > 0
    if has_point:
        last = objects[-1]
        x = int(last.get("left", 0) / ratio)
        y = int(last.get("top", 0) / ratio)
        st.caption(f"Punto seleccionado: x={x}, y={y}")
    else:
        st.info("Haz clic sobre la pared en el lienzo para segmentar y pintar.")

    if st.button("🎨 PINTAR AHORA", disabled=not has_point):
        try:
            with st.spinner("Cargando IA y pintando..."):
                resultado = paint_wall(img, x, y, paint_rgb)
            st.image(resultado, caption="Resultado", use_container_width=True)
        except Exception as exc:
            st.error(f"No se pudo completar el proceso: {exc}")
            st.exception(exc)

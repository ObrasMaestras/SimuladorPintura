import subprocess
import sys
import os

# ESTA LÍNEA OBLIGA A LA NUBE A INSTALAR LA LIBRERÍA SÍ O SÍ
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from streamlit_canvas import st_canvas
except ImportError:
    install('streamlit-drawable-canvas')
    from streamlit_canvas import st_canvas

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import urllib.request

# El resto del código sigue igual...
st.title("🖌️ Simulador de Pintura")

# (Aquí sigue el código que ya teníamos de la IA y el Canvas)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import ctypes

# ---- CONFIG ----
input_file = "Comp.mov"
output_file = "Comp_stabilized.avi"
accuracy_threshold = 0.95  # umbral para fallback manual

# Escalar preview según pantalla
user32 = ctypes.windll.user32
screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
screen_w = int(screen_w * 0.9)
screen_h = int(screen_h * 0.9)

# ---- FUNCIONES ----
def scale_preview(frame_bgr, max_w, max_h):
    h, w = frame_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    preview = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)))
    return preview, scale

def select_sprocket(frame_bgr, frame_number):
    """Permite dibujar manualmente un sprocket"""
    point = []
    ih, iw = frame_bgr.shape[:2]
    preview, scale = scale_preview(frame_bgr, screen_w, screen_h)
    disp_w, disp_h = preview.shape[1], preview.shape[0]

    root = tk.Tk()
    root.title(f"Frame {frame_number} - Arrastra para dibujar sprocket")
    root.resizable(False, False)
    tk_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)))
    canvas = tk.Canvas(root, width=disp_w, height=disp_h)
    canvas.pack()
    canvas.create_image(0,0,anchor='nw',image=tk_img)

    rect_id = [None]
    start_pos = [None,None]

    def on_press(event):
        start_pos[0], start_pos[1] = event.x, event.y
        rect_id[0] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='blue', width=2)

    def on_drag(event):
        if rect_id[0] is not None:
            canvas.coords(rect_id[0], start_pos[0], start_pos[1], event.x, event.y)

    def on_release(event):
        x0, y0 = start_pos
        x1, y1 = event.x, event.y
        x0o, y0o = int(x0/scale), int(y0/scale)
        x1o, y1o = int(x1/scale), int(y1/scale)
        x_min, x_max = sorted([x0o,x1o])
        y_min, y_max = sorted([y0o,y1o])
        point.append([x_min, y_min, x_max, y_max])
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.mainloop()

    return point[0]  # [x0,y0,x1,y1]

def detect_sprocket(frame_gray, template_gray, threshold=0.8):
    res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= threshold:
        x, y = max_loc
        h, w = template_gray.shape
        return [x, y, x+w, y+h], max_val
    else:
        return None, max_val

# ---- CARGAR VIDEO ----
cap = cv2.VideoCapture(input_file)
if not cap.isOpened():
    print("❌ No se pudo abrir el archivo")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'IYUV')
out = cv2.VideoWriter(output_file, fourcc, fps, (w,h))

# ---- PRIMER FRAME ----
ret, ref_frame = cap.read()
if not ret:
    print("❌ No se pudo leer el primer frame")
    cap.release()
    exit(1)

frame_idx = 0
sprocket_coords = select_sprocket(ref_frame, frame_idx)

# Crear template en gris
x0, y0, x1, y1 = sprocket_coords
template = ref_frame[y0:y1, x0:x1]
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Calcular desplazamiento vertical relativo para el siguiente frame
dx_rel = 0
dy_rel = y0 - 0  # el de arriba en siguiente frame será la misma posición vertical relativa

# ---- PROCESAR FRAMES ----
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar sprocket automáticamente
    detected_rect, score = detect_sprocket(frame_gray, template_gray)

    if detected_rect is None or score < accuracy_threshold:
        print(f"⚠ Frame {frame_idx} - Falló detección automática, selección manual requerida")
        sprocket_coords = select_sprocket(frame, frame_idx)
        x0, y0, x1, y1 = sprocket_coords
        template = frame[y0:y1, x0:x1]
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        detected_rect = sprocket_coords

    # Traslación simple
    dx = x0 - detected_rect[0]
    dy = y0 - detected_rect[1]
    M = np.float32([[1,0,dx],[0,1,dy]])
    stabilized = cv2.warpAffine(frame, M, (w,h), borderMode=cv2.BORDER_REPLICATE)

    out.write(stabilized)

    # Preview escalado
    preview, _ = scale_preview(stabilized, screen_w, screen_h)
    cv2.imshow("Preview", preview)
    if cv2.waitKey(1) & 0xFF == 27:
        print("⏹ Interrumpido por usuario")
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Estabilización completada: {output_file}")

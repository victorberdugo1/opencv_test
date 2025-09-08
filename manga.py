#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_panels_shojo_improved.py
Versión mejorada para extracción de viñetas de manga shojo con:
- Mejor detección de bordes claros y difusos
- Manejo de elementos que sobresalen de viñetas
- Orden de lectura más preciso (derecha a izquierda, arriba a abajo)
- Técnicas adicionales para viñetas artísticas/no convencionales
"""

import os
import sys
import argparse
from pathlib import Path
import json
import csv
import math
import copy

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

# ----------------------------
# Utilities: box merging, sorting, IO
# ----------------------------
def union_boxes(boxes, merge_dist=40, img_size=None):
    """
    Merge boxes that overlap or are closer than merge_dist.
    boxes: list of (x,y,w,h)
    returns merged list
    """
    if not boxes:
        return []
    
    # Si tenemos información del tamaño de imagen, hacer merge_dist relativo
    if img_size:
        h, w = img_size
        merge_dist = max(merge_dist, int(min(h, w) * 0.02))
    
    rects = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        new_rects = []
        used = [False]*len(rects)
        for i, r1 in enumerate(rects):
            if used[i]:
                continue
            x1,y1,w1,h1 = r1
            rx1,ry1,rx2,ry2 = x1,y1,x1+w1,y1+h1
            merged = [x1,y1,w1,h1]
            for j, r2 in enumerate(rects):
                if i==j or used[j]:
                    continue
                x2,y2,w2,h2 = r2
                sx1,sy1,sx2,sy2 = x2,y2,x2+w2,y2+h2
                
                # Calcular superposición
                inter_w = max(0, min(rx2, sx2) - max(rx1, sx1))
                inter_h = max(0, min(ry2, sy2) - max(ry1, sy1))
                inter_area = inter_w * inter_h
                
                # Calcular área de unión
                union_area = w1*h1 + w2*h2 - inter_area
                
                # Calcular proximidad entre bordes
                horiz_dist = min(abs(rx1 - sx2), abs(rx2 - sx1), 
                                abs(rx1 - sx1), abs(rx2 - sx2))
                vert_dist = min(abs(ry1 - sy2), abs(ry2 - sy1),
                               abs(ry1 - sy1), abs(ry2 - sy2))
                edge_dist = max(horiz_dist, vert_dist)
                
                # Unir si hay superposición significativa o están muy cerca
                overlap_ratio = inter_area / min(w1*h1, w2*h2) if min(w1*h1, w2*h2) > 0 else 0
                
                if overlap_ratio > 0.05 or edge_dist <= merge_dist:
                    nx1 = min(rx1, sx1)
                    ny1 = min(ry1, sy1)
                    nx2 = max(rx2, sx2)
                    ny2 = max(ry2, sy2)
                    merged = [nx1, ny1, nx2-nx1, ny2-ny1]
                    rx1,ry1,rx2,ry2 = nx1,ny1,nx2,ny2
                    used[j] = True
                    changed = True
            new_rects.append(merged)
            used[i] = True
        rects = new_rects
    return [(int(x),int(y),int(w),int(h)) for (x,y,w,h) in rects]

def sort_panels_reading_order(boxes, direction='rtl', img_size=None, row_tol=0.08):
    """
    Sort boxes into reading order for manga (right to left, top to bottom).
    direction: 'rtl' or 'ltr'
    row_tol: relative tolerance (fraction of image height) to group boxes into rows
    boxes: list of (x,y,w,h)
    returns list sorted
    """
    if not boxes:
        return []
    
    if img_size:
        h, w = img_size
    else:
        # Estimar tamaño de imagen a partir de las cajas
        max_x = max([b[0] + b[2] for b in boxes])
        max_y = max([b[1] + b[3] for b in boxes])
        h, w = max_y, max_x
    
    # Agrupar en filas basado en la posición Y
    rows = []
    row_tol_px = int(h * row_tol)
    
    for b in boxes:
        x, y, w, h_box = b
        center_y = y + h_box/2
        
        found_row = False
        for row in rows:
            row_center_y = row['y_center']
            if abs(center_y - row_center_y) <= row_tol_px:
                row['boxes'].append(b)
                row['y_center'] = (row['y_center'] * len(row['boxes']) + center_y) / (len(row['boxes']) + 1)
                found_row = True
                break
        
        if not found_row:
            rows.append({'y_center': center_y, 'boxes': [b]})
    
    # Ordenar filas por posición Y (arriba a abajo)
    rows.sort(key=lambda r: r['y_center'])
    
    # Ordenar cajas dentro de cada fila
    result = []
    for row in rows:
        if direction == 'rtl':
            # Derecha a izquierda
            row['boxes'].sort(key=lambda b: -(b[0] + b[2]/2))  # Ordenar por centro X descendente
        else:
            # Izquierda a derecha
            row['boxes'].sort(key=lambda b: (b[0] + b[2]/2))   # Ordenar por centro X ascendente
        result.extend(row['boxes'])
    
    return result

# ----------------------------
# Core detection tailored for shojo manga
# ----------------------------
def detect_panels_shojo(img_bgr,
                        min_area=3000,
                        kernel_size=25,
                        merge_dist=40,
                        use_watershed=False,
                        debug=False):
    """
    Returns list of boxes (x,y,w,h) best-effort for shojo manga pages.
    Improved version with better handling of white borders and artistic panels.
    """
    h, w = img_bgr.shape[:2]
    
    # Ajustar parámetros basados en el tamaño de imagen
    min_area = max(min_area, int(h * w * 0.0005))  # Al menos 0.05% del área de la imagen
    kernel_size = max(5, min(50, int(min(h, w) * 0.02)))  # Kernel proporcional al tamaño
    merge_dist = max(10, int(min(h, w) * 0.015))  # Distancia de merge proporcional
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Preprocesamiento - mejorar contraste y reducir ruido
    # Equalización de histograma local para mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    
    # Reducción de ruido preservando bordes
    blur = cv2.bilateralFilter(gray_eq, 9, 75, 75)
    
    # 2. Detección de bordes para encontrar contornos de viñetas
    # Detectar bordes con Canny (ajustado para bordes suaves)
    edges_soft = cv2.Canny(blur, 20, 60)
    edges_strong = cv2.Canny(blur, 50, 150)
    edges = cv2.bitwise_or(edges_soft, edges_strong)
    
    # Dilatar bordes para conectar líneas rotas
    kernel_edges = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel_edges, iterations=2)
    
    # 3. Encontrar regiones potenciales de viñetas
    # Usar umbral adaptativo para regiones claras y oscuras
    th_adapt_light = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 201, 5)
    th_adapt_dark = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 201, 5)
    
    # Combinar diferentes métodos
    combined = cv2.bitwise_or(edges, th_adapt_light)
    combined = cv2.bitwise_or(combined, th_adapt_dark)
    
    # Operaciones morfológicas para cerrar contornos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # Eliminar pequeño ruido
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    
    # Rellenar áreas cerradas
    mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
    
    # 4. Encontrar contornos y filtrar por área y relación de aspecto
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        
        # Filtrar por área mínima
        if area < min_area:
            continue
            
        # Filtrar por relación de aspecto (evitar líneas muy delgadas)
        aspect_ratio = ww / hh
        if (aspect_ratio > 20 or aspect_ratio < 0.05) and area < min_area * 5:
            continue
            
        # Expandir ligeramente las cajas para capturar bordes
        expand_x = max(2, int(ww * 0.02))
        expand_y = max(2, int(hh * 0.02))
        x = max(0, x - expand_x)
        y = max(0, y - expand_y)
        ww = min(w - x, ww + 2 * expand_x)
        hh = min(h - y, hh + 2 * expand_y)
        
        boxes.append((x, y, ww, hh))
    
    # 5. Fusionar cajas cercanas
    merged = union_boxes(boxes, merge_dist=merge_dist, img_size=(h, w))
    
    # 6. Si hay pocas cajas, intentar métodos alternativos
    if len(merged) <= 2:
        # Intentar con watershed si está habilitado
        if use_watershed:
            markers = np.zeros((h, w), dtype=np.int32)
            for i, (x, y, ww, hh) in enumerate(merged, start=1):
                markers[y:y+hh, x:x+ww] = i
                
            # Marcar el fondo
            sure_bg = cv2.dilate(mask, kernel, iterations=3)
            markers[sure_bg == 0] = len(merged) + 1
            
            # Aplicar watershed
            img_for_ws = img_bgr.copy()
            cv2.watershed(img_for_ws, markers)
            
            # Extraer nuevas regiones
            new_boxes = []
            for mark in np.unique(markers):
                if mark <= 0 or mark > len(merged):
                    continue
                temp_mask = np.zeros((h, w), dtype=np.uint8)
                temp_mask[markers == mark] = 255
                
                cnts, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    x, y, ww, hh = cv2.boundingRect(cnt)
                    if ww * hh > min_area:
                        new_boxes.append((x, y, ww, hh))
            
            if new_boxes:
                merged = union_boxes(new_boxes, merge_dist=merge_dist, img_size=(h, w))
        
        # Si todavía hay pocas cajas, intentar detección basada en proyecciones
        if len(merged) <= 2:
            # Proyección vertical para encontrar divisiones
            v_proj = np.sum(mask, axis=0) / 255
            v_peaks = np.where(v_proj > np.mean(v_proj) * 1.5)[0]
            
            # Proyección horizontal
            h_proj = np.sum(mask, axis=1) / 255
            h_peaks = np.where(h_proj > np.mean(h_proj) * 1.5)[0]
            
            # Si encontramos picos, crear cajas basadas en ellos
            if len(v_peaks) > 2 and len(h_peaks) > 2:
                v_splits = [0] + [p for p in v_peaks if p > w*0.1 and p < w*0.9] + [w]
                h_splits = [0] + [p for p in h_peaks if p > h*0.1 and p < h*0.9] + [h]
                
                # Crear cajas de la grid
                for i in range(len(h_splits)-1):
                    for j in range(len(v_splits)-1):
                        x1, y1 = v_splits[j], h_splits[i]
                        x2, y2 = v_splits[j+1], h_splits[i+1]
                        area = (x2-x1) * (y2-y1)
                        if area > min_area:
                            merged.append((x1, y1, x2-x1, y2-y1))
    
    # Ordenar por posición (se aplicará orden de lectura después)
    merged.sort(key=lambda b: (b[1], b[0]))
    
    # Preparar imágenes de debug si es necesario
    dbg = {}
    if debug:
        dbg = {
            'gray': gray,
            'gray_eq': gray_eq,
            'edges_soft': edges_soft,
            'edges_strong': edges_strong,
            'edges': edges,
            'th_adapt_light': th_adapt_light,
            'th_adapt_dark': th_adapt_dark,
            'combined': combined,
            'closed': closed,
            'mask': mask
        }
    
    return merged, dbg

# ----------------------------
# Interactive editor to adjust boxes (Tkinter)
# ----------------------------
class InteractiveEditor:
    def __init__(self, image_bgr, boxes, scale_to_screen=True):
        self.image_bgr = image_bgr
        self.orig_h, self.orig_w = image_bgr.shape[:2]
        self.boxes = [list(b) for b in boxes]  # x,y,w,h
        self.selected_idx = None
        self.dragging = False
        self.resizing = False
        self.resize_corner = None
        self.start_xy = (0,0)
        self.start_box = None

        # prepare display image scaled to fit screen
        root_tmp = tk.Tk()
        screen_w = root_tmp.winfo_screenwidth()
        screen_h = root_tmp.winfo_screenheight()
        root_tmp.destroy()

        max_w = int(screen_w * 0.9)
        max_h = int(screen_h * 0.9)
        self.scale = min(1.0, float(max_w) / self.orig_w, float(max_h) / self.orig_h) if scale_to_screen else 1.0
        disp_w = int(round(self.orig_w * self.scale))
        disp_h = int(round(self.orig_h * self.scale))

        pil_img = Image.fromarray(cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB))
        if self.scale != 1.0:
            pil_img = pil_img.resize((disp_w, disp_h), Image.LANCZOS)
        self.disp_img = ImageTk.PhotoImage(pil_img)

        # Build UI
        self.root = tk.Tk()
        self.root.title("Editor interactivo de viñetas: mueve (izq), redimensiona (arrastrar esquina), Del borra, R agrega rectángulo, Enter guarda")
        
        # Frame principal con canvas y scrollbars
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas con scrollbars
        self.canvas = tk.Canvas(frame, width=min(disp_w, max_w), height=min(disp_h, max_h))
        h_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scroll.grid(row=1, column=0, sticky="ew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        self.canvas.create_image(0, 0, anchor='nw', image=self.disp_img)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        
        self.rect_ids = []
        self.handle_ids = []
        self.draw_boxes()

        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_press)  # right-click to start add
        self.canvas.bind("<Key>", self.on_key)
        self.canvas.focus_set()

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill='x')
        tk.Button(btn_frame, text="Aceptar y guardar (Enter)", command=self.on_accept).pack(side='left', padx=8, pady=4)
        tk.Button(btn_frame, text="Cancelar (Esc)", command=self.on_cancel).pack(side='right', padx=8, pady=4)

        self.result = None
        self.drawing_new = False
        self.new_start = None

    def draw_boxes(self):
        # clear existing
        for rid in self.rect_ids + self.handle_ids:
            try:
                self.canvas.delete(rid)
            except Exception:
                pass
        self.rect_ids = []
        self.handle_ids = []
        for i, b in enumerate(self.boxes):
            x,y,w,h = b
            x1,y1,x2,y2 = [int(round(v * self.scale)) for v in (x,y,x+w,y+h)]
            color = 'blue' if i==self.selected_idx else 'yellow'
            rid = self.canvas.create_rectangle(x1,y1,x2,y2, outline=color, width=2, tags="box")
            self.rect_ids.append(rid)
            
            # Dibujar número de panel
            text_id = self.canvas.create_text(x1+10, y1+10, text=str(i+1), fill='red', font=("Arial", 12, "bold"))
            self.rect_ids.append(text_id)
            
            # draw small corner handles (for resizing)
            hs = 6
            for cx,cy in ((x1,y1),(x2,y1),(x2,y2),(x1,y2)):
                hid = self.canvas.create_rectangle(cx-hs, cy-hs, cx+hs, cy+hs, fill='white', outline='black', tags="handle")
                self.handle_ids.append(hid)

    def find_box_at(self, x, y):
        # return index of topmost box containing (disp coords)
        for i in reversed(range(len(self.boxes))):
            bx,by,bw,bh = self.boxes[i]
            x1,y1,x2,y2 = bx*self.scale, by*self.scale, (bx+bw)*self.scale, (by+bh)*self.scale
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return None

    def find_handle_at(self, x, y):
        # find corner handle near point; return (box_idx, corner_id 0..3) corners ordered TL,TR,BR,BL
        hs = 8
        for i, b in enumerate(self.boxes):
            bx,by,bw,bh = b
            corners = [
                (int(round(bx*self.scale)), int(round(by*self.scale))),
                (int(round((bx+bw)*self.scale)), int(round(by*self.scale))),
                (int(round((bx+bw)*self.scale)), int(round((by+bh)*self.scale))),
                (int(round(bx*self.scale)), int(round((by+bh)*self.scale))),
            ]
            for ci, (cx,cy) in enumerate(corners):
                if abs(cx - x) <= hs and abs(cy - y) <= hs:
                    return i, ci
        return None, None

    def on_press(self, event):
        x,y = event.x, event.y
        self.start_xy = (x,y)
        # check handles first for resizing
        bi, corner = self.find_handle_at(x,y)
        if bi is not None:
            self.selected_idx = bi
            self.resizing = True
            self.resize_corner = corner
            self.start_box = copy.deepcopy(self.boxes[bi])
            self.draw_boxes()
            return
        # else check inside box for dragging
        bi = self.find_box_at(x,y)
        if bi is not None:
            self.selected_idx = bi
            self.dragging = True
            self.start_box = copy.deepcopy(self.boxes[bi])
            self.draw_boxes()
            return
        # click empty: deselect
        self.selected_idx = None
        self.draw_boxes()

    def on_motion(self, event):
        x,y = event.x, event.y
        if self.dragging and self.selected_idx is not None:
            sx, sy = self.start_xy
            dx = (x - sx) / self.scale
            dy = (y - sy) / self.scale
            bx,by,bw,bh = self.start_box
            nx = int(round(max(0, min(self.orig_w - bw, bx + dx))))
            ny = int(round(max(0, min(self.orig_h - bh, by + dy))))
            self.boxes[self.selected_idx][0] = nx
            self.boxes[self.selected_idx][1] = ny
            self.draw_boxes()
        elif self.resizing and self.selected_idx is not None:
            # handle resizing based on corner
            bx,by,bw,bh = self.start_box
            # map current mouse pos to orig coords
            mx = int(round(x / self.scale))
            my = int(round(y / self.scale))
            if self.resize_corner == 0:  # TL
                x2 = bx + bw
                y2 = by + bh
                nx = max(0, min(x2-10, mx))
                ny = max(0, min(y2-10, my))
                self.boxes[self.selected_idx] = [nx, ny, x2-nx, y2-ny]
            elif self.resize_corner == 1:  # TR
                x1 = bx
                y2 = by + bh
                nx2 = max(x1+10, mx)
                ny = max(0, min(y2-10, my))
                self.boxes[self.selected_idx] = [x1, ny, nx2-x1, y2-ny]
            elif self.resize_corner == 2:  # BR
                nx2 = max(bx+10, int(round(x/self.scale)))
                ny2 = max(by+10, int(round(y/self.scale)))
                self.boxes[self.selected_idx] = [bx, by, nx2-bx, ny2-by]
            elif self.resize_corner == 3:  # BL
                x2 = bx + bw
                nx = max(0, min(x2-10, int(round(x/self.scale))))
                ny2 = max(by+10, int(round(y/self.scale)))
                self.boxes[self.selected_idx] = [nx, by, x2-nx, ny2-by]
            self.draw_boxes()

    def on_release(self, event):
        self.dragging = False
        self.resizing = False
        self.resize_corner = None
        self.start_box = None

    def on_right_press(self, event):
        # start drawing a new box (right click and drag)
        x,y = event.x, event.y
        self.drawing_new = True
        self.new_start = (x,y)
        # create temporary rectangle
        self.temp_rect = self.canvas.create_rectangle(x,y,x,y, outline='red', width=2)

        # bind motion and release to handle drawing
        self.canvas.bind("<B3-Motion>", self.on_drawing_motion)
        self.canvas.bind("<ButtonRelease-3>", self.on_drawing_release)

    def on_drawing_motion(self, event):
        x,y = event.x, event.y
        x0,y0 = self.new_start
        self.canvas.coords(self.temp_rect, x0, y0, x, y)

    def on_drawing_release(self, event):
        # finalize new box
        x1,y1 = self.new_start
        x2,y2 = event.x, event.y
        self.canvas.delete(self.temp_rect)
        self.canvas.unbind("<B3-Motion>")
        self.canvas.unbind("<ButtonRelease-3>")
        self.drawing_new = False
        self.new_start = None
        # map to orig
        ox1, oy1 = int(round(min(x1,x2)/self.scale)), int(round(min(y1,y2)/self.scale))
        ox2, oy2 = int(round(max(x1,x2)/self.scale)), int(round(max(y1,y2)/self.scale))
        ow, oh = max(10, ox2-ox1), max(10, oy2-oy1)
        self.boxes.append([ox1, oy1, ow, oh])
        self.selected_idx = len(self.boxes)-1
        self.draw_boxes()

    def on_key(self, event):
        k = event.keysym
        if k == 'Delete' or k == 'BackSpace':
            if self.selected_idx is not None:
                del self.boxes[self.selected_idx]
                self.selected_idx = None
                self.draw_boxes()
        elif k == 'Return':
            self.on_accept()
        elif k == 'r' or k == 'R':
            # reset boxes to empty (con firmeza)
            self.boxes = []
            self.draw_boxes()
        elif k == 'Escape':
            self.on_cancel()

    def on_accept(self):
        # finalize and return boxes
        self.result = [tuple(map(int, b)) for b in self.boxes]
        self.root.destroy()

    def on_cancel(self):
        self.result = None
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.result

# ----------------------------
# Main flow
# ----------------------------
def process_file(path, out_dir, opts):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print("No pude leer:", path)
        return []

    basename = os.path.splitext(os.path.basename(path))[0]

    # Procesar la imagen
    boxes, dbg = detect_panels_shojo(img_bgr,
                                    min_area=opts.min_area,
                                    kernel_size=opts.kernel,
                                    merge_dist=opts.merge_dist,
                                    use_watershed=opts.watershed,
                                    debug=opts.debug)

    # Si no se detectan viñetas, intentar con parámetros más agresivos
    if not boxes:
        print("No se detectaron viñetas, intentando con parámetros más sensibles...")
        boxes2, _ = detect_panels_shojo(img_bgr,
                                      min_area=max(500, opts.min_area//4),
                                      kernel_size=max(7, opts.kernel//3),
                                      merge_dist=max(10, opts.merge_dist//2),
                                      use_watershed=True,
                                      debug=False)
        boxes = boxes2

    # Editor interactivo si está habilitado
    if opts.interactive:
        editor = InteractiveEditor(img_bgr, boxes, scale_to_screen=True)
        new_boxes = editor.run()
        if new_boxes is None:
            print("Edición cancelada por usuario; no se guardarán paneles de esta página.")
            return []
        boxes = new_boxes

    # Ordenar final por orden de lectura
    sorted_boxes = sort_panels_reading_order(boxes, direction=opts.direction, img_size=img_bgr.shape[:2])

    # Crear directorio de salida si no existe
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    
    # Guardar cada viñeta
    for i, (x,y,wid,hei) in enumerate(sorted_boxes, start=1):
        # Asegurarse de que las coordenadas estén dentro de la imagen
        x = max(0, x)
        y = max(0, y)
        wid = min(img_bgr.shape[1] - x, wid)
        hei = min(img_bgr.shape[0] - y, hei)
        
        if wid <= 0 or hei <= 0:
            continue
            
        crop = img_bgr[y:y+hei, x:x+wid]
        out_name = f"{basename}_{i:03d}.png"
        out_path = os.path.join(out_dir, out_name)
        try:
            cv2.imwrite(out_path, crop)
            saved.append({'page': path, 'index': i, 'out_path': out_path, 'bbox': [int(x),int(y),int(wid),int(hei)]})
        except Exception as e:
            print("Error guardando panel:", e)

    # Guardar imágenes de debug si se solicita
    if opts.debug and dbg:
        dbg_dir = os.path.join(out_dir, f"{basename}_debug")
        os.makedirs(dbg_dir, exist_ok=True)
        for k,v in dbg.items():
            if v is None:
                continue
            if len(v.shape) == 2:
                out_dbg = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
            else:
                out_dbg = v
            cv2.imwrite(os.path.join(dbg_dir, f"{k}.png"), out_dbg)
        
        # Imagen de visión general con cajas
        vis = img_bgr.copy()
        for i, (x,y,wid,hei) in enumerate(sorted_boxes, start=1):
            cv2.rectangle(vis, (x,y), (x+wid, y+hei), (0,255,0), 2)
            cv2.putText(vis, str(i), (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imwrite(os.path.join(dbg_dir, "overview.png"), vis)

    return saved

def list_input_files(path, recursive=False):
    p = Path(path)
    exts = ('.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp')
    files = []
    if p.is_file():
        files = [str(p)]
    elif p.is_dir():
        if recursive:
            for f in p.rglob('*'):
                if f.suffix.lower() in exts:
                    files.append(str(f))
        else:
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in exts:
                    files.append(str(f))
    else:
        raise FileNotFoundError(path)
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(description="Extrae viñetas especializadas para shojo manga")
    parser.add_argument('input', help='archivo o carpeta con páginas')
    parser.add_argument('-o','--output', default='panels_out', help='carpeta de salida')
    parser.add_argument('--recursive', action='store_true', help='recorrer carpetas recursivamente')
    parser.add_argument('--direction', choices=['rtl','ltr'], default='rtl', help='orden de lectura (rtl por defecto para manga)')
    parser.add_argument('--min-area', type=int, default=3000, help='área mínima en px^2 para panel')
    parser.add_argument('--kernel', type=int, default=25, help='kernel size para morph close')
    parser.add_argument('--merge-dist', type=int, default=40, help='distancia px para fusionar cajas cercanas')
    parser.add_argument('--watershed', action='store_true', help='usar watershed para separar regiones unidas')
    parser.add_argument('--interactive', action='store_true', help='abrir editor interactivo para ajustar cajas antes de guardar')
    parser.add_argument('--debug', action='store_true', help='guardar imágenes intermedias para debug')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    files = list_input_files(args.input, recursive=args.recursive)
    if not files:
        print("No encontré imágenes en:", args.input); sys.exit(1)

    all_saved = []
    for f in files:
        print("Procesando:", f)
        saved = process_file(f, args.output, args)
        all_saved.extend(saved)

    # Escribir metadatos CSV y JSON
    csv_path = os.path.join(args.output, "panels_index.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['page','panel_index','out_path','x','y','w','h'])
        for s in all_saved:
            x,y,wid,hei = s['bbox']
            writer.writerow([s['page'], s['index'], s['out_path'], x,y,wid,hei])

    json_path = os.path.join(args.output, "panels_index.json")
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(all_saved, jf, indent=2, ensure_ascii=False)

    print("Terminado. Paneles guardados en:", args.output)
    print("CSV:", csv_path)
    print("JSON:", json_path)

if __name__ == "__main__":
    main()

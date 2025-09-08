#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filmstab_interactive.py - Versión interactiva corregida
 - La PRIMERA imagen sirve como referencia de tamaño; todas las imágenes con ese mismo
   tamaño NO se procesan.
 - Cuando se encuentra una imagen MÁS GRANDE que la referencia, se abre una ventana Tkinter
   mostrando la imagen completa (escalada si necesario). Puedes mover el rectángulo ROI con
   el ratón y confirmar Sí/No. La decisión se aplica a todas las imágenes del mismo tamaño.
 - Las tiras/strips se guardan en --output-path con sufijo "_strip" (sin "fullh"/"fullstrip").
"""

import os
import sys
import csv
import argparse
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ----------------------------
parser = argparse.ArgumentParser(description="Interactive film stabilization with ROI adjustment (fixed)")
parser.add_argument('Path', help='video file or image sequence or folder')
parser.add_argument('-m','--template', required=True, help='sprocket hole template')
parser.add_argument('-o','--output-path', default='out', help='folder for ROI outputs')
parser.add_argument('--tm-threshold', type=float, default=0.90)
parser.add_argument('--min-threshold', type=float, default=0.60)
parser.add_argument('--threshold-step', type=float, default=0.05)
parser.add_argument('--show', action='store_true')
parser.add_argument('--pad-mode', choices=['constant','replicate','reflect'], default='replicate')
parser.add_argument('--pad-color', type=int, nargs=3, default=(0,0,0))
parser.add_argument('--roi-offset-x', type=int, default=0)
parser.add_argument('--roi-offset-y', type=int, default=0)
parser.add_argument('--roi-width', type=int, default=None)
parser.add_argument('--roi-height', type=int, default=None)
parser.add_argument('--round', action='store_true', help='Round positions to integers (recommended)')
parser.add_argument('--comp-width', type=int, default=1920, help='After Effects comp width')
parser.add_argument('--comp-height', type=int, default=1080, help='After Effects comp height')
parser.add_argument('--frame-rate', type=int, default=24, help='Frame rate for AE script')

args = parser.parse_args()
os.makedirs(args.output_path, exist_ok=True)

# For DNG handling placeholders (kept minimal)
try:
    import rawpy
    DNG_SUPPORT = True
except Exception:
    rawpy = None
    DNG_SUPPORT = False

try:
    import tifffile
    TIFFFILE_SUPPORT = True
except Exception:
    tifffile = None
    TIFFFILE_SUPPORT = False

# ----------------------------
def open_input(path):
    if os.path.isdir(path):
        exts = ('.png','.jpg','.jpeg','.tif','.tiff','.bmp','.dng')
        files = sorted([os.path.join(path,f) for f in os.listdir(path) if f.lower().endswith(exts)])
        return True, files, None
    if '%' in path:
        cap = cv2.VideoCapture(path, cv2.CAP_IMAGES)
        return False, [], cap
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("No pude abrir la entrada:", path); sys.exit(1)
    return False, [], cap

def match_template_adaptive(image_gray, template_gray, initial_threshold, min_threshold, threshold_step):
    if image_gray.shape[0] < template_gray.shape[0] or image_gray.shape[1] < template_gray.shape[1]:
        return 0.0, -1, -1, initial_threshold
    res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    current_threshold = initial_threshold
    while current_threshold >= min_threshold:
        if maxv >= current_threshold:
            return float(maxv), int(maxloc[0]), int(maxloc[1]), current_threshold
        current_threshold -= threshold_step
    return float(maxv), int(maxloc[0]), int(maxloc[1]), current_threshold

def get_border_type(mode):
    if mode == 'constant': return cv2.BORDER_CONSTANT
    if mode == 'replicate': return cv2.BORDER_REPLICATE
    if mode == 'reflect': return cv2.BORDER_REFLECT
    return cv2.BORDER_CONSTANT

def pad_extract_exact(src, roi_x, roi_y, roi_w, roi_h, pad_mode, pad_color):
    ih, iw = src.shape[:2]
    src_x0 = max(0, roi_x); src_y0 = max(0, roi_y)
    src_x1 = min(iw, roi_x + roi_w); src_y1 = min(ih, roi_y + roi_h)
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        available = np.zeros((0,0,3), dtype=np.uint8)
    else:
        available = src[src_y0:src_y1, src_x0:src_x1].copy()
    pad_left = max(0, 0 - roi_x)
    pad_top = max(0, 0 - roi_y)
    pad_right = max(0, (roi_x + roi_w) - iw)
    pad_bottom = max(0, (roi_y + roi_h) - ih)
    padded = False
    if pad_left or pad_top or pad_right or pad_bottom:
        padded = True
        borderType = get_border_type(pad_mode)
        if borderType == cv2.BORDER_CONSTANT:
            bcol = tuple(int(c) for c in pad_color)
            roi = cv2.copyMakeBorder(available, pad_top, pad_bottom, pad_left, pad_right, borderType, value=bcol)
        else:
            roi = cv2.copyMakeBorder(available, pad_top, pad_bottom, pad_left, pad_right, borderType)
        if roi is None or roi.size == 0:
            roi = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
            if borderType == cv2.BORDER_CONSTANT:
                roi[:,:,:] = bcol
    else:
        roi = available
    if roi.shape[0] != roi_h or roi.shape[1] != roi_w:
        final = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        h = min(roi.shape[0], roi_h)
        w = min(roi.shape[1], roi_w)
        if h > 0 and w > 0:
            final[0:h, 0:w] = roi[0:h, 0:w]
        roi = final
    return roi, padded, (pad_left, pad_top, pad_right, pad_bottom)

def extract_full_width_strip_from_bgr(src_bgr, roi_y, roi_h, pad_mode, pad_color):
    ih, iw = src_bgr.shape[:2]
    roi_y_i = int(round(roi_y))
    src_y0 = max(0, roi_y_i)
    src_y1 = min(ih, roi_y_i + roi_h)
    if src_y0 >= src_y1:
        canvas = np.zeros((roi_h, iw, 3), dtype=np.uint8)
        canvas[:] = tuple(int(c) for c in pad_color)
        return canvas
    available = src_bgr[src_y0:src_y1, 0:iw].copy()
    top_pad = max(0, 0 - roi_y_i)
    bottom_pad = max(0, (roi_y_i + roi_h) - ih)
    if top_pad or bottom_pad:
        borderType = get_border_type(pad_mode)
        if borderType == cv2.BORDER_CONSTANT:
            bcol = tuple(int(c) for c in pad_color)
            filled = cv2.copyMakeBorder(available, top_pad, bottom_pad, 0, 0, borderType, value=bcol)
        else:
            filled = cv2.copyMakeBorder(available, top_pad, bottom_pad, 0, 0, borderType)
        if filled.shape[0] != roi_h or filled.shape[1] != iw:
            final = np.zeros((roi_h, iw, 3), dtype=np.uint8)
            h = min(filled.shape[0], roi_h)
            w = min(filled.shape[1], iw)
            if h > 0 and w > 0:
                final[0:h, 0:w] = filled[0:h, 0:w]
            return final
        return filled
    else:
        if available.shape[0] != roi_h:
            final = np.zeros((roi_h, iw, 3), dtype=np.uint8)
            final[:] = tuple(int(c) for c in pad_color)
            final[0:available.shape[0], 0:iw] = available
            return final
        return available

# ----------------------------
use_list, image_files, cap = open_input(args.Path)
seed_tpl = cv2.imread(args.template)
if seed_tpl is None:
    print("No pude leer la plantilla"); sys.exit(1)

tpl_h, tpl_w = seed_tpl.shape[:2]
seed_tpl_gray = cv2.cvtColor(seed_tpl, cv2.COLOR_BGR2GRAY)

roi_width = args.roi_width if args.roi_width is not None else tpl_w
roi_height = args.roi_height if args.roi_height is not None else tpl_h

csv_path = os.path.join(args.output_path, "extraction_log.csv")
csv_f = open(csv_path, 'w', newline='', encoding='utf-8')
csv_w = csv.writer(csv_f)
csv_w.writerow(["frame", "src_file", "roi_file", "strip_file", "hole_x", "hole_y", "roi_x", "roi_y", "score", "padded", "processed", "user_decision"])

total_frames = len(image_files) if use_list else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
print(f"Total frames: {total_frames}")
print(f"Template: {tpl_w}x{tpl_h}")
print(f"ROI: {roi_width}x{roi_height}, offset: ({args.roi_offset_x}, {args.roi_offset_y})")
print(f"Target composition: {args.comp_width}x{args.comp_height}")

frame_idx = 0
saved_count = 0
skipped_count = 0
hole_positions = []
roi_positions = []
img_width = None
img_height = None

ae_filename = os.path.join(args.output_path, "perfect_stabilization.jsx")

# Behavior state:
reference_size = None                 # (w,h) of the first image
group_process_map = dict()            # map size (w,h) -> True/False (user decision for that size > reference)
# reference_size -> False (do not process) by default

# ----------------------------
def ask_user_adjust_roi(frame_bgr, mx, my, tpl_w, tpl_h, frame_number, score):
    """
    Muestra la imagen completa (escalada a la pantalla si hace falta) en una ventana Tkinter,
    permite mover el rectángulo ROI (click+drag dentro del rect) y devuelve:
      (choice_bool, roi_x_orig, roi_y_orig, roi_w, roi_h)
    choice_bool: True => recortar/si (aplicar también al grupo), False => no
    """
    # Convert BGR->RGB for PIL
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ih, iw = img_rgb.shape[:2]

    # Determine screen max available (give margins)
    root_tmp = tk.Tk()
    screen_w = root_tmp.winfo_screenwidth()
    screen_h = root_tmp.winfo_screenheight()
    root_tmp.destroy()

    max_w = int(screen_w * 0.9)
    max_h = int(screen_h * 0.9)

    # scale to fit screen if needed
    scale = min(1.0, float(max_w) / iw, float(max_h) / ih)

    display_w = int(round(iw * scale))
    display_h = int(round(ih * scale))

    # Create scaled PIL image for display
    pil_img = Image.fromarray(img_rgb)
    if scale != 1.0:
        pil_display = pil_img.resize((display_w, display_h), Image.LANCZOS)
    else:
        pil_display = pil_img

    root = tk.Tk()
    root.title(f"Frame {frame_number} — Ajusta ROI y pulsa Sí/No")
    # Prevent the window from being resized by the user (keeps mapping stable)
    root.resizable(False, False)

    tk_img = ImageTk.PhotoImage(pil_display)
    canvas = tk.Canvas(root, width=display_w, height=display_h, highlightthickness=0)
    canvas.pack()
    canvas_img = canvas.create_image(0, 0, anchor='nw', image=tk_img)

    # Initial ROI (in original coordinates)
    roi_w = roi_width
    roi_h = roi_height
    roi_x_orig = mx + args.roi_offset_x
    roi_y_orig = my + args.roi_offset_y

    # Map original coords to display coords
    def to_disp(x, y):
        return int(round(x * scale)), int(round(y * scale))
    def to_orig(x_disp, y_disp):
        return int(round(x_disp / scale)), int(round(y_disp / scale))

    rx1, ry1 = to_disp(roi_x_orig, roi_y_orig)
    rx2, ry2 = to_disp(roi_x_orig + roi_w, roi_y_orig + roi_h)

    rect = canvas.create_rectangle(rx1, ry1, rx2, ry2, outline='blue', width=3)

    drag = {"x": 0, "y": 0, "dragging": False}

    def on_rect_press(event):
        # start drag only if click inside rect
        x, y = event.x, event.y
        coords = canvas.coords(rect)  # [x1,y1,x2,y2]
        if coords[0] <= x <= coords[2] and coords[1] <= y <= coords[3]:
            drag["dragging"] = True
            drag["x"] = x
            drag["y"] = y

    def on_rect_motion(event):
        if not drag["dragging"]:
            return
        dx = event.x - drag["x"]
        dy = event.y - drag["y"]
        canvas.move(rect, dx, dy)
        drag["x"] = event.x
        drag["y"] = event.y

    def on_rect_release(event):
        drag["dragging"] = False

    # Bind mouse events to the canvas (we track clicks inside the rect)
    canvas.bind("<ButtonPress-1>", on_rect_press)
    canvas.bind("<B1-Motion>", on_rect_motion)
    canvas.bind("<ButtonRelease-1>", on_rect_release)

    # Decision container
    result = {"choice": None}

    def choose_yes():
        # Map displayed rect coords back to original
        x1, y1, x2, y2 = canvas.coords(rect)
        ox1, oy1 = to_orig(x1, y1)
        ox2, oy2 = to_orig(x2, y2)
        # clamp within image bounds
        ox1 = max(0, min(iw-1, int(round(ox1))))
        oy1 = max(0, min(ih-1, int(round(oy1))))
        ox2 = max(0, min(iw, int(round(ox2))))
        oy2 = max(0, min(ih, int(round(oy2))))
        w = max(1, ox2 - ox1)
        h = max(1, oy2 - oy1)
        result["choice"] = (True, ox1, oy1, w, h)
        root.destroy()

    def choose_no():
        result["choice"] = (False, None, None, None, None)
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.pack(fill='x', pady=6)
    b_yes = tk.Button(btn_frame, text="Sí (cortar)", command=choose_yes, width=14, bg='green', fg='white')
    b_yes.pack(side='left', padx=12)
    b_no = tk.Button(btn_frame, text="No (continuar)", command=choose_no, width=14, bg='red', fg='white')
    b_no.pack(side='right', padx=12)

    # Run modal
    root.mainloop()

    if result["choice"] is None:
        # User closed window - treat as No
        return False, None, None, None, None
    return result["choice"]

# ----------------------------
print("\n=== INICIANDO PROCESAMIENTO SEGÚN REGLAS DEL USUARIO ===")

while True:
    if use_list:
        if frame_idx >= total_frames:
            break
        src_path = image_files[frame_idx]
        frame_bgr = load_image_file(src_path) if src_path.lower().endswith('.dng') and DNG_SUPPORT else cv2.imread(src_path)
        if frame_bgr is None:
            print(f"[{frame_idx:04d}] No pude leer {src_path}, skipping")
            frame_idx += 1
            skipped_count += 1
            continue
    else:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break
        src_path = None

    ih, iw = frame_bgr.shape[:2]
    size_tuple = (iw, ih)

    # Set first image as reference size
    if reference_size is None:
        reference_size = size_tuple
        group_process_map[reference_size] = False
        print(f"[{frame_idx:04d}] Tamaño de referencia establecido: {reference_size}. Todas las imágenes con este tamaño NO serán procesadas.")
        csv_w.writerow([frame_idx, src_path or "", "", "", -1, -1, -1, -1, 0.0, False, False, "reference_size"])
        skipped_count += 1
        frame_idx += 1
        continue

    # Prepare gray for template matching
    try:
        img_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        img_gray = frame_bgr.copy()

    score, mx, my, used_th = match_template_adaptive(img_gray, seed_tpl_gray, args.tm_threshold, args.min_threshold, args.threshold_step)
    if mx < 0 or my < 0:
        print(f"[{frame_idx:04d}] Template not found (score {score:.3f}) — skipping frame")
        csv_w.writerow([frame_idx, src_path or "", "", "", -1, -1, -1, -1, f"{score:.4f}", False, False, "no_match"])
        skipped_count += 1
        frame_idx += 1
        continue

    # If same as reference -> skip (do not process)
    if size_tuple == reference_size:
        print(f"[{frame_idx:04d}] Tamaño igual a la referencia ({size_tuple}) -> NO procesado.")
        csv_w.writerow([frame_idx, src_path or "", "", "", mx+tpl_w/2.0, my+tpl_h/2.0, mx+args.roi_offset_x, my+args.roi_offset_y, f"{score:.4f}", False, False, "ref_size_skip"])
        skipped_count += 1
        frame_idx += 1
        continue

    # If we already have a decision for this size, apply it
    if size_tuple in group_process_map:
        process_group = group_process_map[size_tuple]
        if not process_group:
            print(f"[{frame_idx:04d}] Tamaño {size_tuple} marcado previamente como NO procesar -> SALTANDO.")
            csv_w.writerow([frame_idx, src_path or "", "", "", mx+tpl_w/2.0, my+tpl_h/2.0, mx+args.roi_offset_x, my+args.roi_offset_y, f"{score:.4f}", False, False, "group_no"])
            skipped_count += 1
            frame_idx += 1
            continue
        else:
            # Process this frame (group decision = yes)
            user_decision = "group_yes"
            # continue to processing below
    else:
        # New size and larger than reference -> ask user with adjustable ROI
        if iw > reference_size[0] or ih > reference_size[1]:
            print(f"[{frame_idx:04d}] Nueva talla detectada {size_tuple} (mayor que referencia) — mostrando para decidir.")
            choice, ox, oy, ow, oh = ask_user_adjust_roi(
                frame_bgr, mx, my, tpl_w, tpl_h, frame_idx, score
            )

            if not choice:
                group_process_map[size_tuple] = False
                print(f"[{frame_idx:04d}] Usuario eligió NO recortar imágenes de tamaño {size_tuple}.")
                csv_w.writerow([
                    frame_idx, src_path or "", "", "",
                    mx + tpl_w / 2.0, my + tpl_h / 2.0,
                    mx + args.roi_offset_x, my + args.roi_offset_y,
                    f"{score:.4f}", False, False, "user_no"
                ])
                skipped_count += 1
                frame_idx += 1
                continue
            else:
                group_process_map[size_tuple] = True
                print(f"[{frame_idx:04d}] Usuario eligió SI recortar imágenes de tamaño {size_tuple}.")
                roi_x, roi_y, roi_w, roi_h = ox, oy, ow, oh
                user_decision = "user_yes_group"

        else:
            # size different but not larger than reference -> process automatically
            print(f"[{frame_idx:04d}] Tamaño {size_tuple} distinto pero no mayor que la referencia -> procesando automáticamente.")
            user_decision = "auto_process"
            roi_x = mx + args.roi_offset_x
            roi_y = my + args.roi_offset_y
            roi_w = roi_width
            roi_h = roi_height

    # If group decision already True (and we didn't get roi_x/roi_y above), set default ROI from match+offset
    if (size_tuple in group_process_map and group_process_map[size_tuple]) and ('roi_x' not in locals()):
        roi_x = mx + args.roi_offset_x
        roi_y = my + args.roi_offset_y
        roi_w = roi_width
        roi_h = roi_height

    # Round if requested
    if args.round:
        roi_x = int(round(roi_x))
        roi_y = int(round(roi_y))
        roi_w = int(round(roi_w))
        roi_h = int(round(roi_h))

    # Extract ROI (with padding if necessary)
    roi_img, padded, pad_tuple = pad_extract_exact(frame_bgr, roi_x, roi_y, roi_w, roi_h, args.pad_mode, args.pad_color)

    # Save ROI
    if src_path:
        base = os.path.splitext(os.path.basename(src_path))[0]
        roi_basename = f"{base}_roi"
    else:
        roi_basename = f"frame_{frame_idx:06d}_roi"
    roi_filename = os.path.join(args.output_path, f"{roi_basename}.png")
    try:
        cv2.imwrite(roi_filename, roi_img)
    except Exception as e:
        print(f"Error guardando ROI {roi_filename}: {e}")
        roi_filename = ""

    # Save strip (same output path) with suffix "_strip" (no "fullh")
    strip_file = ""
    strip_basename = (os.path.splitext(os.path.basename(src_path))[0] if src_path else f"frame_{frame_idx:06d}") + "_strip"
    strip_path = os.path.join(args.output_path, f"{strip_basename}.png")
    try:
        strip = extract_full_width_strip_from_bgr(frame_bgr, roi_y, roi_h, args.pad_mode, args.pad_color)
        cv2.imwrite(strip_path, strip)
        strip_file = strip_path
    except Exception as e:
        print(f"Error guardando strip {strip_path}: {e}")
        strip_file = ""

    hole_x = mx + tpl_w/2.0
    hole_y = my + tpl_h/2.0
    roi_center_x = roi_x + roi_w/2.0
    roi_center_y = roi_y + roi_h/2.0

    hole_positions.append((hole_x, hole_y))
    roi_positions.append((roi_center_x, roi_center_y))

    csv_w.writerow([frame_idx, src_path or "", roi_filename, strip_file, f"{hole_x:.2f}", f"{hole_y:.2f}", f"{roi_x:.2f}", f"{roi_y:.2f}", f"{score:.4f}", padded, True, user_decision])
    saved_count += 1
    print(f"[{frame_idx:04d}] Guardado ROI: {roi_filename}  Strip: {strip_file}  Score: {score:.3f}  Padded: {padded}")

    # Optional quick OpenCV preview
    if args.show:
        disp = frame_bgr.copy()
        try:
            cv2.rectangle(disp, (mx, my), (mx + tpl_w, my + tpl_h), (0,255,0), 2)
            cv2.rectangle(disp, (int(roi_x), int(roi_y)), (int(roi_x + roi_w), int(roi_y + roi_h)), (255,0,0), 2)
            text = f"Frame {frame_idx} Score {score:.3f}"
            cv2.putText(disp, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Preview", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Interrupción por usuario (vista previa). Saliendo.")
                break
        except Exception:
            pass

    frame_idx += 1

# Final
csv_f.close()
if cap is not None and not use_list:
    cap.release()
cv2.destroyAllWindows()

# AE script (same basic export as before)
print("Generando script de After Effects:", ae_filename)
try:
    with open(ae_filename, 'w', encoding='utf-8') as f:
        f.write("// perfect_stabilization.jsx\n")
        f.write("// Generado por filmstab_interactive.py - establece keyframes de posición basados en el centro de ROI\n")
        f.write("var compW = %d;\n" % args.comp_width)
        f.write("var compH = %d;\n" % args.comp_height)
        f.write("var fr = %d;\n" % args.frame_rate)
        f.write("var comp = app.project.items.addComp('stabilization_comp', compW, compH, 1, %f, fr);\n" % (max(1.0, (len(roi_positions)/float(max(1,args.frame_rate))))))
        f.write("app.beginUndoGroup('Import stabilization keyframes');\n")
        f.write("var nullL = comp.layers.addNull();\n")
        f.write("nullL.name = 'stabilization_null';\n")
        f.write("var posProp = nullL.property('Position');\n")
        for i, (cx, cy) in enumerate(roi_positions):
            time = float(i) / float(args.frame_rate)
            if img_width and img_height:
                mapped_x = (float(cx) / float(img_width)) * args.comp_width
                mapped_y = (float(cy) / float(img_height)) * args.comp_height
            else:
                mapped_x = cx
                mapped_y = cy
            f.write("posProp.setValueAtTime(%f, [%f, %f]);\n" % (time, mapped_x, mapped_y))
        f.write("app.endUndoGroup();\n")
    print("AE script escrito correctamente.")
except Exception as e:
    print("No pude escribir el AE script:", e)

print("\n=== RESUMEN ===")
print(f"Frames totales: {total_frames}")
print(f"Procesados (guardados): {saved_count}")
print(f"Saltados: {skipped_count}")
print(f"CSV: {csv_path}")
print(f"AE script: {ae_filename}")
print("FIN.")


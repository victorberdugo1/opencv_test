#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python filmstab.py 01.mov -m template1.png --show --tm-threshold 1.00 
filmstab_perfect.py
Estabilización PERFECTA de película 16mm/8mm.
Genera keyframes HOLD exactos para After Effects con cálculos matemáticamente correctos.
"""
import cv2, os, sys, argparse, numpy as np, csv
from datetime import datetime

parser = argparse.ArgumentParser(description="Perfect film stabilization with exact HOLD keyframes")
parser.add_argument('Path', help='video file or image sequence')
parser.add_argument('-m','--template', required=True, help='sprocket hole template')
parser.add_argument('-o','--output-path', default='out')
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

def open_input(path):
    if os.path.isdir(path):
        exts = ('.png','.jpg','.jpeg','.tif','.tiff','.bmp')
        files = sorted([os.path.join(path,f) for f in os.listdir(path) if f.lower().endswith(exts)])
        return True, files, None
    if '%' in path:
        cap = cv2.VideoCapture(path, cv2.CAP_IMAGES)
        if not cap.isOpened():
            cap.release(); cap = cv2.VideoCapture(path)
        return False, [], cap
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("No pude abrir la entrada:", path); sys.exit(1)
    return False, [], cap

def match_template_adaptive(image_gray, template_gray, initial_threshold, min_threshold, threshold_step):
    """Template matching con threshold adaptativo"""
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
    """Extrae ROI exacto con padding si es necesario"""
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
    
    # Asegurar dimensiones exactas
    if roi.shape[0] != roi_h or roi.shape[1] != roi_w:
        final = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        h = min(roi.shape[0], roi_h)
        w = min(roi.shape[1], roi_w)
        if h > 0 and w > 0:
            final[0:h, 0:w] = roi[0:h, 0:w]
        roi = final
    
    return roi, padded, (pad_left, pad_top, pad_right, pad_bottom)

# Inicialización
use_list, image_files, cap = open_input(args.Path)
seed_tpl = cv2.imread(args.template)
if seed_tpl is None:
    print("No pude leer la plantilla"); sys.exit(1)

tpl_h, tpl_w = seed_tpl.shape[:2]
seed_tpl_gray = cv2.cvtColor(seed_tpl, cv2.COLOR_BGR2GRAY)

# Configurar ROI
roi_width = args.roi_width if args.roi_width is not None else tpl_w
roi_height = args.roi_height if args.roi_height is not None else tpl_h

os.makedirs(args.output_path, exist_ok=True)
csv_path = os.path.join(args.output_path, "extraction_log.csv")
csv_f = open(csv_path, 'w', newline='', encoding='utf-8')
csv_w = csv.writer(csv_f)
csv_w.writerow(["frame", "file", "hole_x", "hole_y", "roi_x", "roi_y", "score", "padded"])

total_frames = len(image_files) if use_list else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
print(f"Total frames: {total_frames}")
print(f"Template: {tpl_w}x{tpl_h}")  
print(f"ROI: {roi_width}x{roi_height}, offset: ({args.roi_offset_x}, {args.roi_offset_y})")
print(f"Target composition: {args.comp_width}x{args.comp_height}")

# Variables de procesamiento
frame_idx = 0
saved_count = 0
hole_positions = []  # Posiciones del centro del template (sprocket hole)
roi_positions = []   # Posiciones del centro del ROI extraído
reference_hole_pos = None
reference_roi_pos = None
img_width = None
img_height = None

# Bucle principal
while True:
    if use_list:
        if frame_idx >= total_frames:
            break
        src = cv2.imread(image_files[frame_idx])
        frame_idx += 1
        cap_frame_number = frame_idx
    else:
        ret, src = cap.read()
        if not ret:
            break
        frame_idx += 1
        cap_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or frame_idx)
    
    if src is None:
        continue
    
    if img_width is None:
        img_height, img_width = src.shape[:2]
        print(f"Image dimensions: {img_width}x{img_height}")
    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    # Buscar template
    tm_score, mx, my, used_threshold = match_template_adaptive(
        gray, seed_tpl_gray, args.tm_threshold, args.min_threshold, args.threshold_step
    )
    
    print(f"[{cap_frame_number:04d}] Score: {tm_score:.3f} at ({mx},{my}) threshold: {used_threshold:.3f}")
    
    if mx >= 0 and my >= 0:  # Template encontrado
        # Calcular posición del centro del template (sprocket hole)
        hole_center_x = mx + tpl_w / 2.0
        hole_center_y = my + tpl_h / 2.0
        
        # Calcular ROI
        roi_x = mx + args.roi_offset_x  
        roi_y = my + tpl_h + args.roi_offset_y  # ROI después del template
        
        # Extraer ROI
        roi, padded, pads = pad_extract_exact(src, roi_x, roi_y, roi_width, roi_height, args.pad_mode, args.pad_color)
        
        # Guardar
        out_name = os.path.join(args.output_path, f"frame_{saved_count:05d}.png")
        cv2.imwrite(out_name, roi)
        
        # Calcular centro del ROI extraído
        roi_center_x = roi_x + roi_width / 2.0
        roi_center_y = roi_y + roi_height / 2.0
        
        # Guardar posiciones
        hole_positions.append((hole_center_x, hole_center_y))
        roi_positions.append((roi_center_x, roi_center_y))
        
        # Establecer referencia en el primer frame
        if reference_hole_pos is None:
            reference_hole_pos = (hole_center_x, hole_center_y)
            reference_roi_pos = (roi_center_x, roi_center_y)
            print(f"REFERENCE set - Hole: {reference_hole_pos}, ROI: {reference_roi_pos}")
        
        csv_w.writerow([cap_frame_number, out_name, hole_center_x, hole_center_y, roi_center_x, roi_center_y, tm_score, padded])
        saved_count += 1
        
        if args.show:
            dbg = src.copy()
            # Template detectado
            cv2.rectangle(dbg, (mx, my), (mx + tpl_w, my + tpl_h), (0, 255, 0), 2)
            # ROI extraído  
            cv2.rectangle(dbg, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)
            # Centros
            cv2.circle(dbg, (int(hole_center_x), int(hole_center_y)), 3, (0, 255, 255), -1)  # Hole center
            cv2.circle(dbg, (int(roi_center_x), int(roi_center_y)), 5, (0, 0, 255), -1)    # ROI center
            
            cv2.imshow("Debug", cv2.resize(dbg, (int(dbg.shape[1] * 0.6), int(dbg.shape[0] * 0.6))))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    else:  # Template no encontrado - usar última posición conocida
        if len(hole_positions) > 0:
            # Repetir última posición
            last_hole = hole_positions[-1]  
            last_roi = roi_positions[-1]
            hole_positions.append(last_hole)
            roi_positions.append(last_roi)
            
            # Extraer ROI usando última posición conocida
            roi_x = int(last_roi[0] - roi_width / 2.0)
            roi_y = int(last_roi[1] - roi_height / 2.0)
            
            roi, padded, pads = pad_extract_exact(src, roi_x, roi_y, roi_width, roi_height, args.pad_mode, args.pad_color)
            
            out_name = os.path.join(args.output_path, f"frame_{saved_count:05d}.png")
            cv2.imwrite(out_name, roi)
            
            csv_w.writerow([cap_frame_number, out_name, last_hole[0], last_hole[1], last_roi[0], last_roi[1], 0.0, padded])
            saved_count += 1
            
            print(f"[{cap_frame_number:04d}] No template - using last position")
            
            if args.show:
                dbg = src.copy()
                cv2.rectangle(dbg, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)
                cv2.circle(dbg, (int(last_roi[0]), int(last_roi[1])), 5, (0, 0, 255), -1)
                cv2.imshow("Debug", cv2.resize(dbg, (int(dbg.shape[1] * 0.6), int(dbg.shape[0] * 0.6))))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print(f"[{cap_frame_number:04d}] No template and no reference yet - skipping")

# Cerrar archivos
csv_f.close()
if not use_list:
    cap.release()
if args.show:
    cv2.destroyAllWindows()

# Generar script de After Effects
if len(roi_positions) > 0 and reference_roi_pos is not None:
    print(f"\nGenerating After Effects script for {len(roi_positions)} frames...")
    
    # Calcular centros de composición
    comp_center_x = args.comp_width / 2.0
    comp_center_y = args.comp_height / 2.0
    
    # Calcular centro de imagen fuente
    img_center_x = img_width / 2.0  
    img_center_y = img_height / 2.0
    
    ref_roi_x, ref_roi_y = reference_roi_pos
    
    ae_filename = os.path.join(args.output_path, "perfect_stabilization.jsx")
    
    with open(ae_filename, 'w', encoding='utf-8') as f:
        f.write("// PERFECT FILM STABILIZATION SCRIPT\n")
        f.write("// Auto-generated - do not edit manually\n\n")
        
        # Generar datos de keyframes
        f.write("var keyframes = {\n")
        for i, (roi_x, roi_y) in enumerate(roi_positions):
            frame_num = i + 1
            
            # Calcular movimiento del ROI respecto a la referencia
            movement_x = roi_x - ref_roi_x
            movement_y = roi_y - ref_roi_y
            
            # Calcular posición final de la capa para cancelar el movimiento
            # La capa debe moverse en dirección OPUESTA al movimiento detectado
            final_x = comp_center_x - movement_x
            final_y = comp_center_y - movement_y
            
            if args.round:
                final_x = int(round(final_x))
                final_y = int(round(final_y))
            
            f.write(f"   {frame_num}: [{final_x:.2f}, {final_y:.2f}],\n")
        
        f.write("};\n\n")
        
        # Script principal
        f.write(f"""var frameRate = {args.frame_rate};
var compWidth = {args.comp_width};
var compHeight = {args.comp_height};
var sourceWidth = {img_width};
var sourceHeight = {img_height};

var comp = app.project.activeItem;
if (comp && comp instanceof CompItem) {{
    var selectedLayers = comp.selectedLayers;
    if (selectedLayers.length > 0) {{
        app.beginUndoGroup("Perfect Film Stabilization");
        
        var layer = selectedLayers[0];
        var transform = layer.property("Transform");
        var position = transform.property("Position");
        var anchorPoint = transform.property("Anchor Point");
        var scale = transform.property("Scale");
        var rotation = transform.property("Rotation");
        
        try {{
            // Reset layer properties
            layer.startTime = 0;
            if (layer.parent) layer.parent = null;
            layer.threeDLayer = false;
            layer.motionBlur = false;
            
            // Set anchor point to source center
            var srcW = (layer.source && layer.source.width) ? layer.source.width : sourceWidth;
            var srcH = (layer.source && layer.source.height) ? layer.source.height : sourceHeight;
            anchorPoint.setValue([srcW/2, srcH/2]);
            
            // Reset scale and rotation
            scale.setValue([100, 100]);
            rotation.setValue(0);
            
            // Clear existing keyframes
            while (position.numKeys > 0) {{
                position.removeKey(1);
            }}
            
            // Apply stabilization keyframes
            var keyCount = 0;
            for (var frameNum in keyframes) {{
                var time = (parseInt(frameNum) - 1) / frameRate;
                var pos = keyframes[frameNum];
                position.setValueAtTime(time, pos);
                keyCount++;
            }}
            
            // Set all keyframes to HOLD interpolation
            for (var i = 1; i <= position.numKeys; i++) {{
                try {{
                    position.setInterpolationTypeAtKey(i, KeyframeInterpolationType.HOLD, KeyframeInterpolationType.HOLD);
                    position.setSpatialTangentsAtKey(i, [0,0], [0,0]);
                }} catch(e) {{}}
            }}
            
            app.endUndoGroup();
            alert("PERFECT STABILIZATION applied successfully!\\n" +
                  "Keyframes: " + keyCount + " (HOLD interpolation)\\n" +
                  "Composition: " + compWidth + "x" + compHeight + "\\n" +
                  "Source: " + sourceWidth + "x" + sourceHeight + "\\n" +
                  "Reference ROI: {ref_roi_x:.1f}, {ref_roi_y:.1f}");
                  
        }} catch(error) {{
            app.endUndoGroup();
            alert("Error applying stabilization: " + error.toString());
        }}
        
    }} else {{
        alert("Error: Please select the layer with your film sequence.");
    }}
}} else {{
    alert("Error: Please open a composition first.");
}}""")
    
    # Estadísticas
    movements_x = [abs(roi_positions[i][0] - roi_positions[i-1][0]) for i in range(1, len(roi_positions))]
    movements_y = [abs(roi_positions[i][1] - roi_positions[i-1][1]) for i in range(1, len(roi_positions))]
    
    if movements_x and movements_y:
        avg_mov_x = sum(movements_x) / len(movements_x)
        avg_mov_y = sum(movements_y) / len(movements_y)
        max_mov_x = max(movements_x)
        max_mov_y = max(movements_y)
        
        print(f"\n=== STABILIZATION STATISTICS ===")
        print(f"Detected movement in ROI:")
        print(f"  Average: X={avg_mov_x:.2f}px, Y={avg_mov_y:.2f}px")
        print(f"  Maximum: X={max_mov_x:.2f}px, Y={max_mov_y:.2f}px")
        print(f"Keyframes will CANCEL this movement for perfect stabilization.")

print(f"\n=== PROCESS COMPLETED ===")
print(f"Extracted frames: {saved_count}")
print(f"Output folder: {args.output_path}")
print(f"CSV log: {csv_path}")
print(f"AE script: {ae_filename}")
print(f"\nINSTRUCTIONS:")
print(f"1. Import frame sequence into After Effects")
print(f"2. Create composition ({args.comp_width}x{args.comp_height} @ {args.frame_rate}fps)")
print(f"3. Add sequence to composition")  
print(f"4. Select the layer and run the generated .jsx script")
print(f"5. Content will be perfectly stabilized with HOLD keyframes")
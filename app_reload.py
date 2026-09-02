import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import math
from PIL import Image, ImageDraw

# ==========================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ==========================================
st.set_page_config(page_title="Baphte Drop Calculator - Reload", layout="wide")

# ==========================================
# VOS PARAMÈTRES TOPOGRAPHIQUES ET PHYSIQUES (RELOAD)
# ==========================================
IMAGE_WIDTH_METERS = 2647.0 
H_BUS = 400.0  
Z_MAX = 300.0    

# Vitesses exactes
V_BUS = 75.0       
V_H_GLIDE1 = 22.0
V_Z_GLIDE1 = 25.0
V_Z_DIVE1 = 60.0   

V_H_PLAN = 18.5
V_Z_PLAN = 5.0

V_H_DROP = 10.0    
V_Z_DROP = 30.0    

R_MAX_1 = V_H_GLIDE1 / V_Z_GLIDE1
R_MAX_2 = V_H_PLAN / V_Z_PLAN

# Couleurs pour le tracé final
COLOR_FREEFALL_HIGH = (255, 255, 255) # Blanc
COLOR_GLIDER = (0, 255, 0)            # Vert Lime
COLOR_FREEFALL_LOW = (255, 0, 0)      # Rouge
COLOR_BUS = (0, 0, 255)               # Bleu

# ==========================================
# LE CERVEAU DE L'IA (MOTEUR ELLIPTIQUE DYNAMIQUE)
# ==========================================
class DropEngineIA:
    def __init__(self, map_w, map_h, heightmap_array):
        self.img_w = map_w
        self.img_h = map_h
        self.map_width_m = IMAGE_WIDTH_METERS
        self.map_height_m = IMAGE_WIDTH_METERS * (self.img_h / self.img_w)
        self.height_array = heightmap_array

    def get_elevation(self, x, y):
        px = int(round(x / self.map_width_m * self.img_w))
        py = int(round(y / self.map_height_m * self.img_h))
        px = np.clip(px, 0, self.img_w - 1)
        py = np.clip(py, 0, self.img_h - 1)
        return self.height_array[py, px]

    def evaluate_path(self, t_bus, t_deploy, A, bus_vec, bus_length, S):
        J = A + t_bus * bus_vec
        time_bus = (t_bus * bus_length) / V_BUS
        
        dist_J_S = np.linalg.norm(S - J)
        if dist_J_S == 0: 
            Z_open = self.get_elevation(S[0], S[1]) + 50.0
            dZ_A = H_BUS - Z_open
            if dZ_A <= 0: return float('inf'), None
            time_A = dZ_A / V_Z_DIVE1
            time_B = 50.0 / V_Z_DROP
            return time_bus + time_A + time_B, (J, 0, 0, 0, np.array([1, 0]), time_bus, time_A, 0.0, time_B, Z_open, Z_open, 90.0)
        
        dir_vec = (S - J) / dist_J_S
        d_T = t_deploy * dist_J_S
        T = J + d_T * dir_vec
        
        D_A = max(0.0, d_T)
        D_B = max(0.0, dist_J_S - d_T)
        
        # ========================================================
        # 🏔️ 1. ALTITUDE DYNAMIQUE ET SCANNER ANTI-CRASH (PLANEUR)
        # ========================================================
        Z_open = self.get_elevation(T[0], T[1]) + 50.0 # 50m pour Reload
        
        if D_B > 0:
            for step in [0.2, 0.4, 0.6, 0.8, 1.0]:
                P_check = T + step * (S - T)
                Z_ground = self.get_elevation(P_check[0], P_check[1])
                req_Z = Z_ground + (step * D_B) * (V_Z_PLAN / V_H_PLAN)
                if req_Z > Z_open:
                    Z_open = req_Z
                    
        if Z_open > H_BUS:
            return float('inf'), None
        
        dZ_A = H_BUS - Z_open
        if dZ_A <= 0 or D_A > dZ_A * R_MAX_1: 
            return float('inf'), None
            
        # ========================================================
        # 🏔️ 2. SCANNER 3D RÉEL (CHUTE LIBRE)
        # ========================================================
        if D_A > 0:
            for step in [0.3, 0.6, 0.9]:
                P_check = J + step * (T - J)
                Z_ground = self.get_elevation(P_check[0], P_check[1])
                Z_flight = H_BUS - step * dZ_A
                
                # Le planeur s'ouvre de force si on passe sous la barre des 50m
                if Z_flight < Z_ground + 50.0:
                    return float('inf'), None
                    
        # ========================================================
        # 🚀 3. MODÈLE PHYSIQUE : LA PLONGÉE ELLIPTIQUE DYNAMIQUE
        # ========================================================
        if D_A == 0:
            time_A = dZ_A / V_Z_DIVE1
            angle_deg = 90.0
        else:
            H_g = V_H_GLIDE1
            Z_min = V_Z_GLIDE1
            dZ_g = V_Z_DIVE1 - V_Z_GLIDE1
            
            A_eq = (H_g**2) * (dZ_g**2 - Z_min**2)
            B_eq = 2.0 * (H_g**2) * Z_min * dZ_A
            C_eq = - ((D_A**2) * (dZ_g**2) + (H_g**2) * (dZ_A**2))
            
            delta = B_eq**2 - 4 * A_eq * C_eq
            if delta >= 0:
                t_A = (-B_eq + math.sqrt(delta)) / (2 * A_eq)
                if t_A > 0:
                    time_A = t_A
                    v_h = D_A / time_A
                    v_z = dZ_A / time_A
                    angle_deg = math.degrees(math.atan2(v_z, v_h))
                else:
                    return float('inf'), None
            else:
                return float('inf'), None
        # ========================================================
        
        if t_deploy == 1.0:
            dZ_B = 50.0 
        else:
            dZ_B = Z_open - self.get_elevation(S[0], S[1])
            
        if dZ_B <= 0 or D_B > dZ_B * R_MAX_2: 
            return float('inf'), None
            
        # Résolution Phase Planeur/Chute Finale
        det = V_H_PLAN * V_Z_DROP - V_H_DROP * V_Z_PLAN
        if det == 0: return float('inf'), None
        
        t_plan_exact = (D_B * V_Z_DROP - dZ_B * V_H_DROP) / det
        t_drop_exact = (dZ_B * V_H_PLAN - D_B * V_Z_PLAN) / det
        
        if t_plan_exact < 0:
            time_plan = 0.0
            time_drop = dZ_B / V_Z_DROP
            dist_plan = 0.0
            dist_drop = D_B
        elif t_drop_exact < 0:
            return float('inf'), None
        else:
            time_plan = t_plan_exact
            time_drop = t_drop_exact
            dist_plan = time_plan * V_H_PLAN
            dist_drop = time_drop * V_H_DROP
            
        time_B = time_plan + time_drop
        total_time = time_bus + time_A + time_B
        Z_close = Z_open - (time_plan * V_Z_PLAN)
        
        return total_time, (J, D_A, dist_plan, dist_drop, dir_vec, time_bus, time_A, time_plan, time_drop, Z_open, Z_close, angle_deg)

    def run_optimization(self, p_start, p_end, p_spawn):
        A = np.array([p_start[0] / self.img_w * self.map_width_m, p_start[1] / self.img_h * self.map_height_m])
        B = np.array([p_end[0] / self.img_w * self.map_width_m, p_end[1] / self.img_h * self.map_height_m])
        S = np.array([p_spawn[0] / self.img_w * self.map_width_m, p_spawn[1] / self.img_h * self.map_height_m])

        bus_vec = B - A
        bus_length = np.linalg.norm(bus_vec)
        if bus_length == 0: return None

        best_time = float('inf')
        best_splits = None
        paths_tested = 0

        # HYPER-RÉSOLUTION
        t_bus_samples = np.linspace(0, 1, 100) 
        t_deploy_samples = np.linspace(0.01, 1.0, 50)
        
        best_t_bus = 0; best_t_deploy = 0

        for t_bus in t_bus_samples:
            for t_deploy in t_deploy_samples:
                paths_tested += 1
                time_total, splits = self.evaluate_path(t_bus, t_deploy, A, bus_vec, bus_length, S)
                if time_total < best_time:
                    best_time = time_total
                    best_splits = splits
                    best_t_bus = t_bus; best_t_deploy = t_deploy

        if best_time != float('inf'):
            # MICRO-OPTIMISATION
            micro_t_bus = np.linspace(max(0, best_t_bus - 0.03), min(1, best_t_bus + 0.03), 40)
            micro_t_deploy = np.linspace(max(0.01, best_t_deploy - 0.05), min(1, best_t_deploy + 0.05), 40)
            for t_bus in micro_t_bus:
                for t_deploy in micro_t_deploy:
                    paths_tested += 1
                    time_total, splits = self.evaluate_path(t_bus, t_deploy, A, bus_vec, bus_length, S)
                    if time_total < best_time:
                        best_time = time_total
                        best_splits = splits

        if best_time == float('inf'): return None

        J, D_A, dist_plan, dist_drop, dir_vec, time_bus, time_A, time_plan, time_drop, Z_open, Z_close, angle_deg = best_splits
        
        def m_to_px(pt_m):
            return (pt_m[0] / self.map_width_m * self.img_w, pt_m[1] / self.map_height_m * self.img_h)
            
        P0_px = m_to_px(J)
        P1_px = m_to_px(J + dir_vec * D_A)
        P2_px = m_to_px(J + dir_vec * (D_A + dist_plan))
        P3_px = m_to_px(J + dir_vec * (D_A + dist_plan + dist_drop))

        return {
            "time_total": best_time,
            "time_bus": time_bus,
            "time_air": time_A + time_plan + time_drop,
            "time_A": time_A,
            "time_plan": time_plan,
            "time_drop": time_drop,
            "deploy_alt": Z_open,
            "drop_alt": Z_close,
            "angle": angle_deg,
            "paths_tested": paths_tested,
            "P0": P0_px, "P1": P1_px, "P2": P2_px, "P3": P3_px
        }

# ==========================================
# INITIALISATION DES VARIABLES
# ==========================================
if 'points' not in st.session_state: st.session_state.points = []
if 'phase' not in st.session_state: st.session_state.phase = 1
if 'map_key' not in st.session_state: st.session_state.map_key = 0

def reset_all():
    st.session_state.points = []
    st.session_state.phase = 1
    st.session_state.map_key += 1
    if 'result' in st.session_state:
        del st.session_state.result

st.title("🪂 Baphte Drop Calculator - RELOAD")

@st.cache_resource
def load_images():
    # Images pour le mode RELOAD
    img_map_original = Image.open("Map_Reload_Elite_Stronghold.png")
    
    # ==========================================
    # AUTO-ÉTALONNAGE ROBUSTE DE LA HEIGHTMAP (Correction transparence)
    # ==========================================
    img_height_raw = Image.open("Heightmap_Reload_Elite_Stronghold.png")
    if img_height_raw.mode in ('RGBA', 'LA') or (img_height_raw.mode == 'P' and 'transparency' in img_height_raw.info):
        bg = Image.new('RGB', img_height_raw.size, (0, 0, 0))
        bg.paste(img_height_raw, (0, 0), img_height_raw.convert('RGBA'))
        img_height = bg.convert('L')
    else:
        img_height = img_height_raw.convert('L')
        
    img_height = img_height.resize(img_map_original.size)
    raw_arr = np.array(img_height, dtype=np.float32)
    min_val = np.min(raw_arr)
    max_val = np.max(raw_arr)
    
    if max_val > min_val:
        height_arr = (raw_arr - min_val) / (max_val - min_val) * Z_MAX
    else:
        height_arr = np.zeros_like(raw_arr)
    # ==========================================
    
    ui_map = img_map_original.copy()
    ui_map.thumbnail((1000, 1000)) 
    
    scale_x = img_map_original.width / ui_map.width
    scale_y = img_map_original.height / ui_map.height
    
    return img_map_original, height_arr, ui_map, scale_x, scale_y

try:
    map_original, height_array, map_ui, scale_x, scale_y = load_images()
except Exception as e:
    st.error(f"Erreur de chargement. Vérifiez les fichiers sur GitHub ! Erreur : {e}")
    st.stop()

col1, col2 = st.columns([3, 1])

# ==========================================
# COLONNE 2 : INSTRUCTIONS ET METRICS
# ==========================================
with col2:
    st.header("Instructions")
    if st.session_state.phase == 1:
        st.info("📍 **Étape 1 : Le Bus**\nCliquez sur la carte pour placer le **Départ** puis l'**Arrivée**.")
    elif st.session_state.phase == 2:
        st.info("🎯 **Étape 2 : Le Spawn**\nCliquez sur la carte pour indiquer où vous voulez **atterrir**.")
    elif st.session_state.phase == 4:
        st.success("✅ **Calcul terminé !**\nVoici la trajectoire optimale.")

    st.button("🔄 Tout recommencer", on_click=reset_all, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🎨 Légende")
    st.markdown("⚪ **Blanc** : Chute Libre Initiale")
    st.markdown("🟢 **Vert** : Planeur")
    st.markdown("🔴 **Rouge** : Chute Finale")

# ==========================================
# GESTION DU CALCUL IA
# ==========================================
if st.session_state.phase == 3:
    with col1:
        with st.spinner('L\'IA calcule la plongée elliptique (Mode Reload)...'):
            engine = DropEngineIA(map_original.width, map_original.height, height_array)
            
            p_start_real = st.session_state.points[0]
            p_end_real = st.session_state.points[1]
            p_spawn_real = st.session_state.points[2]
            
            result = engine.run_optimization(p_start_real, p_end_real, p_spawn_real)
            st.session_state.result = result
            st.session_state.phase = 4 
    st.rerun()

# ==========================================
# COLONNE 1 : DESSIN CARTE WEB
# ==========================================
img_draw = map_ui.copy()
draw = ImageDraw.Draw(img_draw)

def to_ui(pt_real):
    return (pt_real[0] / scale_x, pt_real[1] / scale_y)

if len(st.session_state.points) >= 1:
    p = to_ui(st.session_state.points[0])
    draw.ellipse((p[0]-6, p[1]-6, p[0]+6, p[1]+6), fill=COLOR_BUS, outline="white", width=2)
    
if len(st.session_state.points) >= 2:
    p1 = to_ui(st.session_state.points[0])
    p2 = to_ui(st.session_state.points[1])
    draw.line([p1, p2], fill=COLOR_BUS, width=3)
    draw.ellipse((p2[0]-6, p2[1]-6, p2[0]+6, p2[1]+6), fill=COLOR_BUS, outline="white", width=2)
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist > 0:
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        ux = dx / dist
        uy = dy / dist
        
        tip = (mid_x + ux * 12, mid_y + uy * 12)
        left = (mid_x - ux * 8 + (-uy) * 8, mid_y - uy * 8 + ux * 8)
        right = (mid_x - ux * 8 - (-uy) * 8, mid_y - uy * 8 - ux * 8)
        
        draw.polygon([tip, left, right], fill=COLOR_BUS)

if len(st.session_state.points) >= 3:
    p3 = to_ui(st.session_state.points[2])
    if st.session_state.phase < 4:
        draw.ellipse((p3[0]-5, p3[1]-5, p3[0]+5, p3[1]+5), fill="lime", outline="black", width=2)

if st.session_state.phase == 4 and hasattr(st.session_state, 'result'):
    res = st.session_state.result
    if res is None:
        st.error("⚠️ Spawn inatteignable depuis ce bus (trop loin ou bloqué par une montagne).")
    else:
        P0 = to_ui(res["P0"])
        P1 = to_ui(res["P1"])
        P2 = to_ui(res["P2"])
        P3 = to_ui(res["P3"])
        
        if res['time_A'] > 0: draw.line([P0, P1], fill=COLOR_FREEFALL_HIGH, width=4)
        if res['time_plan'] > 0: draw.line([P1, P2], fill=COLOR_GLIDER, width=4)
        if res['time_drop'] > 0: draw.line([P2, P3], fill=COLOR_FREEFALL_LOW, width=5)
        
        draw.ellipse((P0[0]-6, P0[1]-6, P0[0]+6, P0[1]+6), fill="yellow", outline="black", width=2)
        draw.ellipse((P1[0]-4, P1[1]-4, P1[0]+4, P1[1]+4), fill="white", outline="black", width=2)
        
        if res['time_plan'] > 0:
            draw.ellipse((P2[0]-4, P2[1]-4, P2[0]+4, P2[1]+4), fill="lime", outline="black", width=2)
            
        draw.ellipse((P3[0]-5, P3[1]-5, P3[0]+5, P3[1]+5), fill="red", outline="white", width=2)
        
        with col2:
            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Temps Total", f"{res['time_total']:.2f} s")
                st.metric("Temps en Bus", f"{res['time_bus']:.1f} s")
            with col_b:
                st.metric("Angle de Chute", f"{res['angle']:.1f}°")
                st.metric("Temps en Vol", f"{res['time_air']:.1f} s")
            
            st.info(f"""
            **⏱️ Altitudes & Changements de Mode :**
            
            🚌 **Saut du Bus** (Altitude : 400 m)
            ⬇️ *Chute Libre pendant {res['time_A']:.1f} s*
            
            🪂 **Ouverture Planeur** (Altitude : **{res['deploy_alt']:.0f} m**)
            ⬇️ *Planeur pendant {res['time_plan']:.1f} s*
            
            🎯 **Chute Finale** (Altitude : **{res['drop_alt']:.0f} m**)
            ⬇️ *Chute vers la cible pendant {res['time_drop']:.1f} s*
            """)
            
            st.caption(f"🤖 L'IA a testé et vérifié {res['paths_tested']:,} chemins.")

with col1:
    value = streamlit_image_coordinates(img_draw, key=f"map_{st.session_state.map_key}")
    
    if value is not None and st.session_state.phase < 3:
        raw_click = (value["x"], value["y"])
        
        true_x = raw_click[0] * scale_x
        true_y = raw_click[1] * scale_y
        click_coords = (true_x, true_y)
        
        if not st.session_state.points or click_coords != st.session_state.points[-1]:
            st.session_state.points.append(click_coords)
            
            if len(st.session_state.points) == 2:
                st.session_state.phase = 2
            elif len(st.session_state.points) == 3:
                st.session_state.phase = 3
            st.rerun()

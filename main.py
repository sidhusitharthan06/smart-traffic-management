import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# 1. PAGE CONFIG & GLOBAL CSS
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Traffic Management System", page_icon="🚦", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }

    .signal-card {
        background: linear-gradient(145deg, #1a1f2e, #141821);
        border: 1px solid #2d3748;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    .signal-card h3 { color: #f0f4ff; margin: 0 0 8px 0; font-size: 1.1rem; }

    .light { width: 28px; height: 28px; border-radius: 50%; margin: 4px auto; }
    .light-on-red    { background: #ff4b4b; box-shadow: 0 0 14px #ff4b4b; }
    .light-on-yellow { background: #ffbd45; box-shadow: 0 0 14px #ffbd45; }
    .light-on-green  { background: #00d26a; box-shadow: 0 0 14px #00d26a; }
    .light-off       { background: #262626; }

    .amb-alert {
        background: linear-gradient(90deg, #dc2626, #b91c1c);
        color: white; padding: 8px 14px; border-radius: 8px;
        font-weight: 700; text-align: center;
        animation: pulse 1.2s infinite; margin-top: 6px;
    }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.6;} }

    .badge-green  { color: #00d26a; font-weight: 600; }
    .badge-red    { color: #ff4b4b; font-weight: 600; }
    .badge-yellow { color: #ffbd45; font-weight: 600; }

    .stMetric { background: #1f2937; padding: 16px; border-radius: 10px; border: 1px solid #374151; }

    .timer-bar {
        background: #374151; border-radius: 6px; height: 8px; margin-top: 6px;
    }
    .timer-fill {
        height: 8px; border-radius: 6px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def render_traffic_light(phase):
    r = "light-on-red"    if phase == "red"    else "light-off"
    y = "light-on-yellow" if phase == "yellow" else "light-off"
    g = "light-on-green"  if phase == "green"  else "light-off"
    return f"""
    <div style="background:#1a1a1a; border-radius:12px; padding:6px 0; width:44px; margin:auto; border:2px solid #333;">
        <div class="light {r}"></div>
        <div class="light {y}"></div>
        <div class="light {g}"></div>
    </div>
    """

def render_signal_card(sig_id, phase, vehicle_count, ambulance_detected, time_left=0):
    phase_badge = f'<span class="badge-{phase}">{phase.upper()}</span>'
    amb_html = '<div class="amb-alert">🚑 AMBULANCE APPROACHING – PRIORITY OVERRIDE</div>' if ambulance_detected else ""
    timer_color = {"red": "#ff4b4b", "yellow": "#ffbd45", "green": "#00d26a"}.get(phase, "#666")
    return f"""
    <div class="signal-card">
        <h3>🚦 Signal {sig_id}</h3>
        <table style="width:100%; color:#d1d5db; font-size:0.9rem;">
            <tr><td>Status</td><td style="text-align:right;">{phase_badge}</td></tr>
            <tr><td>Vehicles</td><td style="text-align:right;">{vehicle_count}</td></tr>
            <tr><td>Time Left</td><td style="text-align:right;">{time_left}s</td></tr>
        </table>
        {amb_html}
    </div>
    """

def display_uploaded_file(uploaded_file, placeholder):
    """Handles both image and video uploads."""
    if uploaded_file is None:
        return False

    file_type = uploaded_file.type
    
    # Handle IMAGE files
    if file_type in ["image/png", "image/jpeg", "image/jpg", "image/webp"]:
        placeholder.image(uploaded_file, use_container_width=True)
        return True
    
    # Handle VIDEO files
    elif file_type in ["video/mp4", "video/avi", "video/x-msvideo", "video/quicktime"]:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps > 0 else 0.03

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame, channels="RGB", use_container_width=True)
            time.sleep(delay)

        cap.release()
        os.unlink(tfile.name)
        return True
    
    return False

# ──────────────────────────────────────────────────────────────
# 3. TIME-BASED SIGNAL CYCLING LOGIC
# ──────────────────────────────────────────────────────────────

def compute_signal_phases(green_duration, yellow_duration):
    """
    Automatically cycles 4 signals based on current time.
    At any moment, exactly 1 signal is GREEN, the rest are RED.
    Before switching, there is a brief YELLOW phase.
    """
    total_cycle = 4 * (green_duration + yellow_duration)
    elapsed = time.time() % total_cycle  # position in the current cycle

    phases = {}
    time_lefts = {}

    for i in range(4):
        sig_id = i + 1
        sig_start = i * (green_duration + yellow_duration)
        sig_end = sig_start + green_duration + yellow_duration

        if sig_start <= elapsed < sig_start + green_duration:
            phases[sig_id] = "green"
            time_lefts[sig_id] = int(sig_start + green_duration - elapsed)
        elif sig_start + green_duration <= elapsed < sig_end:
            phases[sig_id] = "yellow"
            time_lefts[sig_id] = int(sig_end - elapsed)
        else:
            phases[sig_id] = "red"
            # calculate when this signal will next be green
            if elapsed < sig_start:
                time_lefts[sig_id] = int(sig_start - elapsed)
            else:
                time_lefts[sig_id] = int(sig_start + total_cycle - elapsed)

    return phases, time_lefts

# ──────────────────────────────────────────────────────────────
# 4. SESSION STATE DEFAULTS
# ──────────────────────────────────────────────────────────────
if "logs" not in st.session_state:
    st.session_state.logs = [
        f"{datetime.now().strftime('%H:%M:%S')} | System initialised – all signals online"
    ]
if "ambulance" not in st.session_state:
    st.session_state.ambulance = {1: False, 2: False, 3: False, 4: False}

# ──────────────────────────────────────────────────────────────
# 5. SIDEBAR
# ──────────────────────────────────────────────────────────────
st.sidebar.title("🚨 Command Center")
st.sidebar.markdown("---")

view_mode = st.sidebar.selectbox("View", ["Grid Overview", "Signal 1 Focus", "Signal 2 Focus", "Signal 3 Focus", "Signal 4 Focus"])

st.sidebar.markdown("---")
st.sidebar.subheader("Input Source")
input_mode = st.sidebar.radio("Feed Source", ["Upload Image/Video", "Live Camera"], horizontal=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Signal Timing")
green_duration = st.sidebar.slider("Green Duration (seconds)", 10, 120, 30)
yellow_duration = st.sidebar.slider("Yellow Duration (seconds)", 2, 10, 5)
auto_refresh = st.sidebar.toggle("Auto-Refresh Signals", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulate Events")
sim_signal = st.sidebar.selectbox("Choose Signal", [1, 2, 3, 4], key="sim_sig")
if st.sidebar.button("🚑 Simulate Ambulance Detection"):
    st.session_state.ambulance[sim_signal] = True
    st.session_state.logs.insert(0,
        f"{datetime.now().strftime('%H:%M:%S')} | 🚑 AMBULANCE detected 500m from Signal {sim_signal} – PRIORITY OVERRIDE to GREEN"
    )
    st.rerun()

if st.sidebar.button("✅ Clear Ambulance Alert"):
    for k in st.session_state.ambulance:
        st.session_state.ambulance[k] = False
    st.session_state.logs.insert(0,
        f"{datetime.now().strftime('%H:%M:%S')} | Ambulance alerts cleared – resuming normal cycle"
    )
    st.rerun()

# ──────────────────────────────────────────────────────────────
# 6. COMPUTE CURRENT SIGNAL PHASES (TIME-BASED)
# ──────────────────────────────────────────────────────────────
phases, time_lefts = compute_signal_phases(green_duration, yellow_duration)

# Override: if ambulance is detected, force that signal to green
for sig_id in range(1, 5):
    if st.session_state.ambulance[sig_id]:
        phases[sig_id] = "green"
        time_lefts[sig_id] = 0  # stays green until cleared

# Vehicle counts (simulated based on phase)
vehicle_counts = {}
for sig_id in range(1, 5):
    np.random.seed(int(time.time()) // 10 + sig_id)
    vehicle_counts[sig_id] = np.random.randint(5, 45)

# ──────────────────────────────────────────────────────────────
# 7. HEADER & GLOBAL METRICS
# ──────────────────────────────────────────────────────────────
st.title("🚦 AI Traffic Management System")
st.caption("Real-time monitoring with 500m pre-detection for emergency vehicles")

total_vehicles = sum(vehicle_counts.values())
active_amb = sum(1 for v in st.session_state.ambulance.values() if v)
green_signal = [k for k, v in phases.items() if v == "green"]
green_label = f"Signal {green_signal[0]}" if green_signal else "None"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Vehicles", total_vehicles)
k2.metric("Active Ambulance Alerts", active_amb, delta="⚠ PRIORITY" if active_amb > 0 else "✓ Normal")
k3.metric("Current Green", green_label)
k4.metric("Cycle Duration", f"{green_duration + yellow_duration}s per signal")

st.markdown("---")

# ──────────────────────────────────────────────────────────────
# 8. GRID VIEW – all 4 signals
# ──────────────────────────────────────────────────────────────
UPLOAD_TYPES = ["mp4", "avi", "mov", "png", "jpg", "jpeg", "webp"]

if view_mode == "Grid Overview":

    # ── Row 1: Signal 1 & Signal 2 ──
    row1_left, row1_right = st.columns(2)

    for col, sig_id in [(row1_left, 1), (row1_right, 2)]:
        with col:
            st.markdown(render_signal_card(sig_id, phases[sig_id], vehicle_counts[sig_id], st.session_state.ambulance[sig_id], time_lefts[sig_id]), unsafe_allow_html=True)

            feed_a, feed_b, light_col = st.columns([2, 2, 1])

            with feed_a:
                st.markdown(f"<p style='color:#9ca3af; font-size:0.75rem;'>📷 At Signal {sig_id}</p>", unsafe_allow_html=True)
                if input_mode == "Upload Image/Video":
                    uploaded = st.file_uploader(f"Signal {sig_id} – At Signal", type=UPLOAD_TYPES, key=f"sig_{sig_id}_at")
                    ph = st.empty()
                    if uploaded:
                        display_uploaded_file(uploaded, ph)
                    else:
                        st.image(f"https://via.placeholder.com/400x220.png?text=Signal+{sig_id}+No+File", use_container_width=True)
                else:
                    st.image(f"https://via.placeholder.com/400x220.png?text=Signal+{sig_id}+Live+Standby", use_container_width=True)

            with feed_b:
                st.markdown(f"<p style='color:#9ca3af; font-size:0.75rem;'>📷 500m Before Signal {sig_id}</p>", unsafe_allow_html=True)
                if input_mode == "Upload Image/Video":
                    uploaded_500 = st.file_uploader(f"Signal {sig_id} – 500m", type=UPLOAD_TYPES, key=f"sig_{sig_id}_500m")
                    ph2 = st.empty()
                    if uploaded_500:
                        display_uploaded_file(uploaded_500, ph2)
                    else:
                        st.image(f"https://via.placeholder.com/400x220.png?text=500m+Signal+{sig_id}+No+File", use_container_width=True)
                else:
                    st.image(f"https://via.placeholder.com/400x220.png?text=500m+Cam+{sig_id}+Standby", use_container_width=True)

            with light_col:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(render_traffic_light(phases[sig_id]), unsafe_allow_html=True)

    # ── Row 2: Signal 3 & Signal 4 ──
    row2_left, row2_right = st.columns(2)

    for col, sig_id in [(row2_left, 3), (row2_right, 4)]:
        with col:
            st.markdown(render_signal_card(sig_id, phases[sig_id], vehicle_counts[sig_id], st.session_state.ambulance[sig_id], time_lefts[sig_id]), unsafe_allow_html=True)

            feed_a, feed_b, light_col = st.columns([2, 2, 1])

            with feed_a:
                st.markdown(f"<p style='color:#9ca3af; font-size:0.75rem;'>📷 At Signal {sig_id}</p>", unsafe_allow_html=True)
                if input_mode == "Upload Image/Video":
                    uploaded = st.file_uploader(f"Signal {sig_id} – At Signal", type=UPLOAD_TYPES, key=f"sig_{sig_id}_at")
                    ph = st.empty()
                    if uploaded:
                        display_uploaded_file(uploaded, ph)
                    else:
                        st.image(f"https://via.placeholder.com/400x220.png?text=Signal+{sig_id}+No+File", use_container_width=True)
                else:
                    st.image(f"https://via.placeholder.com/400x220.png?text=Signal+{sig_id}+Live+Standby", use_container_width=True)

            with feed_b:
                st.markdown(f"<p style='color:#9ca3af; font-size:0.75rem;'>📷 500m Before Signal {sig_id}</p>", unsafe_allow_html=True)
                if input_mode == "Upload Image/Video":
                    uploaded_500 = st.file_uploader(f"Signal {sig_id} – 500m", type=UPLOAD_TYPES, key=f"sig_{sig_id}_500m")
                    ph2 = st.empty()
                    if uploaded_500:
                        display_uploaded_file(uploaded_500, ph2)
                    else:
                        st.image(f"https://via.placeholder.com/400x220.png?text=500m+Signal+{sig_id}+No+File", use_container_width=True)
                else:
                    st.image(f"https://via.placeholder.com/400x220.png?text=500m+Cam+{sig_id}+Standby", use_container_width=True)

            with light_col:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(render_traffic_light(phases[sig_id]), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 9. SINGLE SIGNAL FOCUS VIEW
# ──────────────────────────────────────────────────────────────
else:
    focus_id = int(view_mode.split()[1])

    st.subheader(f"🔍 Signal {focus_id} – Detailed View")
    st.markdown(render_signal_card(focus_id, phases[focus_id], vehicle_counts[focus_id], st.session_state.ambulance[focus_id], time_lefts[focus_id]), unsafe_allow_html=True)

    cam_left, cam_right = st.columns(2)

    with cam_left:
        st.markdown(f"### 📷 At Signal {focus_id}")
        feed_ph_signal = st.empty()

        if input_mode == "Upload Image/Video":
            vid_at = st.file_uploader(f"Upload Signal {focus_id} – At Signal", type=UPLOAD_TYPES, key=f"focus_{focus_id}_at")
            if vid_at:
                display_uploaded_file(vid_at, feed_ph_signal)
            else:
                st.info("Upload an image or video to start analysis.")
        else:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    feed_ph_signal.image(frame, channels="RGB", use_container_width=True)
                cap.release()
            else:
                st.error("Cannot access webcam (Camera 0).")

    with cam_right:
        st.markdown(f"### 📷 500m Pre-Detection – Signal {focus_id}")
        feed_ph_500m = st.empty()

        if input_mode == "Upload Image/Video":
            vid_500 = st.file_uploader(f"Upload Signal {focus_id} – 500m", type=UPLOAD_TYPES, key=f"focus_{focus_id}_500m")
            if vid_500:
                display_uploaded_file(vid_500, feed_ph_500m)
            else:
                st.info("Upload a 500m pre-detection image or video.")
        else:
            cap2 = cv2.VideoCapture(1)
            if cap2.isOpened():
                while True:
                    ret, frame = cap2.read()
                    if not ret: break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    feed_ph_500m.image(frame, channels="RGB", use_container_width=True)
                cap2.release()
            else:
                st.warning("Cannot access second camera (Camera 1). Use 'Upload' mode instead.")

    # Signal controls
    st.markdown("---")
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        st.markdown(render_traffic_light(phases[focus_id]), unsafe_allow_html=True)
    with ctrl2:
        st.metric("Current Phase", phases[focus_id].upper())
        st.metric("Time Remaining", f"{time_lefts[focus_id]}s")
    with ctrl3:
        st.metric("Queue Length", f"{vehicle_counts[focus_id] * 7}m")
        st.metric("Estimated Wait", f"{vehicle_counts[focus_id] * 3}s")

# ──────────────────────────────────────────────────────────────
# 10. ACTIVITY LOG
# ──────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📜 Activity Log", expanded=False):
    for log in st.session_state.logs[:15]:
        st.text(log)

# ──────────────────────────────────────────────────────────────
# 11. AUTO-REFRESH (updates signal phases every few seconds)
# ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(2)
    st.rerun()

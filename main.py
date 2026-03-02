import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from datetime import datetime
from detection_model import detect_vehicles, VEHICLE_CLASSES

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
    .amb-early-warning {
        background: linear-gradient(90deg, #d97706, #b45309);
        color: white; padding: 8px 14px; border-radius: 8px;
        font-weight: 700; text-align: center;
        animation: pulse 2s infinite; margin-top: 6px;
    }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.6;} }

    .badge-green  { color: #00d26a; font-weight: 600; }
    .badge-red    { color: #ff4b4b; font-weight: 600; }
    .badge-yellow { color: #ffbd45; font-weight: 600; }

    .stMetric { background: #1f2937; padding: 16px; border-radius: 10px; border: 1px solid #374151; }

    .ai-badge {
        display: inline-block; background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white; font-size: 0.7rem; padding: 2px 8px; border-radius: 10px;
        font-weight: 600; margin-left: 6px;
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

def render_signal_card(sig_id, phase, vehicle_count, amb_state, time_left=0, breakdown=None, weight=0):
    """amb_state: None, 'early_warning' (2000m), or 'override' (500m)"""
    phase_colors = {"green": "#00d26a", "red": "#ff4b4b", "yellow": "#ffbd45"}
    phase_color = phase_colors.get(phase, "#999")

    amb_row = ""
    if amb_state == "override":
        amb_row = '<tr><td colspan="2" style="padding-top:8px;"><div style="background:linear-gradient(90deg,#dc2626,#b91c1c);color:white;padding:8px 10px;border-radius:8px;font-weight:700;text-align:center;animation:pulse 1.2s infinite;">AMBULANCE 500m - GREEN OVERRIDE</div></td></tr>'
    elif amb_state == "early_warning":
        amb_row = '<tr><td colspan="2" style="padding-top:8px;"><div style="background:linear-gradient(90deg,#d97706,#b45309);color:white;padding:8px 10px;border-radius:8px;font-weight:700;text-align:center;animation:pulse 2s infinite;">AMBULANCE 2000m - EARLY WARNING</div></td></tr>'

    bd_row = ""
    if breakdown:
        bus = breakdown.get('Bus', 0)
        car = breakdown.get('Car', 0)
        moto = breakdown.get('Motorcycle', 0)
        bd_row = f'<tr><td colspan="2" style="padding-top:6px;"><span style="color:#7dd3fc;font-size:0.75rem;">Bus: {bus} | Car: {car} | Moto: {moto}</span></td></tr>'

    ai_tag = ' <span class="ai-badge">AI Detected</span>' if breakdown else ""
    time_display = "OVERRIDE" if amb_state == "override" else f"{time_left}s"
    weight_row = f'<tr><td>Traffic Weight</td><td style="text-align:right;color:#fbbf24;">{weight}</td></tr>' if weight > 0 else ''

    return f"""<div class="signal-card">
<h3 style="color:#f0f4ff;margin:0 0 8px 0;font-size:1.1rem;">Signal {sig_id}{ai_tag}</h3>
<table style="width:100%;color:#d1d5db;font-size:0.9rem;">
<tr><td>Status</td><td style="text-align:right;color:{phase_color};font-weight:600;">{phase.upper()}</td></tr>
<tr><td>Vehicles</td><td style="text-align:right;">{vehicle_count}</td></tr>
{weight_row}
<tr><td>Time Left</td><td style="text-align:right;">{time_display}</td></tr>
{bd_row}
{amb_row}
</table>
</div>"""


def process_uploaded_image(uploaded_file):
    """Run YOLOv8 detection on an uploaded image."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    uploaded_file.seek(0)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return None, 0, {}
    result = detect_vehicles(frame)
    annotated_rgb = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
    breakdown = {}
    for det in result["detections"]:
        cls_name = det["class"]
        breakdown[cls_name] = breakdown.get(cls_name, 0) + 1
    return annotated_rgb, result["count"], breakdown


def process_uploaded_video(uploaded_file, placeholder):
    """Run YOLOv8 detection on each frame of an uploaded video."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    uploaded_file.seek(0)
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip = max(1, int(fps / 5)) if fps > 0 else 1
    last_count = 0
    last_breakdown = {}
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % skip != 0:
            continue
        result = detect_vehicles(frame)
        annotated_rgb = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
        placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
        last_count = result["count"]
        last_breakdown = {}
        for det in result["detections"]:
            cls_name = det["class"]
            last_breakdown[cls_name] = last_breakdown.get(cls_name, 0) + 1
        time.sleep(0.1)
    cap.release()
    os.unlink(tfile.name)
    return last_count, last_breakdown


def display_and_detect(uploaded_file, placeholder, sig_id):
    """Process an uploaded image or video with AI detection."""
    if uploaded_file is None:
        return False
    file_type = uploaded_file.type
    if file_type in ["image/png", "image/jpeg", "image/jpg", "image/webp"]:
        annotated, count, breakdown = process_uploaded_image(uploaded_file)
        if annotated is not None:
            placeholder.image(annotated, channels="RGB", use_container_width=True)
            st.session_state.detected_counts[sig_id] = count
            st.session_state.detected_breakdowns[sig_id] = breakdown
        return True
    elif file_type in ["video/mp4", "video/avi", "video/x-msvideo", "video/quicktime"]:
        count, breakdown = process_uploaded_video(uploaded_file, placeholder)
        st.session_state.detected_counts[sig_id] = count
        st.session_state.detected_breakdowns[sig_id] = breakdown
        return True
    return False


def display_and_detect_ambulance(uploaded_file, placeholder, sig_id, distance):
    """
    Process upload from 500m or 2000m ambulance camera.
    Automatically triggers ambulance detection in the FIFO queue.
    Only triggers ONCE per unique file upload (not on every rerun).
    """
    if uploaded_file is None:
        return False

    # Create a unique ID for this upload to avoid re-triggering
    upload_id = f"{sig_id}_{distance}_{uploaded_file.name}_{uploaded_file.size}"

    file_type = uploaded_file.type
    processed = False

    # Display the uploaded file with detection
    if file_type in ["image/png", "image/jpeg", "image/jpg", "image/webp"]:
        annotated, count, breakdown = process_uploaded_image(uploaded_file)
        if annotated is not None:
            placeholder.image(annotated, channels="RGB", use_container_width=True)
        processed = True
    elif file_type in ["video/mp4", "video/avi", "video/x-msvideo", "video/quicktime"]:
        count, breakdown = process_uploaded_video(uploaded_file, placeholder)
        processed = True

    if not processed:
        return False

    # Only trigger ambulance alert ONCE per unique upload
    if upload_id in st.session_state.processed_uploads:
        return True  # Already processed, just display the image

    st.session_state.processed_uploads.add(upload_id)

    # Auto-trigger ambulance alert!
    stage = "override" if distance == "500m" else "early_warning"
    already_in_queue = any(e["signal"] == sig_id for e in st.session_state.ambulance_queue)

    if already_in_queue:
        if stage == "override":
            for entry in st.session_state.ambulance_queue:
                if entry["signal"] == sig_id:
                    entry["stage"] = "override"
                    if "override_start" not in entry:
                        entry["override_start"] = time.time()  # start 30s countdown
                    break
    else:
        new_entry = {
            "signal": sig_id,
            "time": datetime.now().strftime('%H:%M:%S'),
            "stage": stage
        }
        if stage == "override":
            new_entry["override_start"] = time.time()  # start 30s countdown
        st.session_state.ambulance_queue.append(new_entry)

    if stage == "override":
        st.session_state.logs.insert(0,
            f"{datetime.now().strftime('%H:%M:%S')} | 🚑 AMBULANCE detected 500m from Signal {sig_id} – IMMEDIATE GREEN OVERRIDE (FIFO)"
        )
    else:
        st.session_state.logs.insert(0,
            f"{datetime.now().strftime('%H:%M:%S')} | 📡 AMBULANCE detected 2000m from Signal {sig_id} – EARLY WARNING – optimizing traffic"
        )

    return True


# ──────────────────────────────────────────────────────────────
# 3. SIGNAL LOGIC — ALL 4 SIGNALS ALWAYS CYCLE
# ──────────────────────────────────────────────────────────────

# Vehicle weight multipliers (heavier = needs more green time to clear)
VEHICLE_WEIGHTS = {
    'Bus': 3,         # Large, slow — needs more time
    'Car': 2,         # Medium
    'Motorcycle': 1   # Small, fast — needs less time
}

def calculate_weight(breakdown):
    """Calculate total traffic weight from vehicle type breakdown."""
    if not breakdown:
        return 0
    total = 0
    for vtype, count in breakdown.items():
        total += count * VEHICLE_WEIGHTS.get(vtype, 1)
    return total


def get_green_duration_for_weight(weight):
    """
    Green time based on traffic WEIGHT (not just count).
    Bus=3, Car=2, Motorcycle=1 per vehicle.
    Formula: 10s base + 2s per weight unit, clamped to 15s–60s.
    """
    if weight == 0:
        return 15  # Default for signals with no detection
    return max(15, min(10 + weight * 2, 60))


def all_weights_similar(weights):
    """Check if all signals have roughly equal weight (within 20%)."""
    active = [w for w in weights.values() if w > 0]
    if len(active) < 2:
        return False
    avg = sum(active) / len(active)
    if avg == 0:
        return True
    return all(abs(w - avg) / avg < 0.2 for w in active)


def compute_ai_signal_phases(detected_counts, ambulance_queue, yellow_dur=3):
    """
    ALL 4 signals always cycle. Priority order = by vehicle count (highest first).
    Signals without detections still get a default 15s green time.
    Ambulance FIFO queue overrides everything.
    """
    phases = {1: "red", 2: "red", 3: "red", 4: "red"}
    time_lefts = {1: 0, 2: 0, 3: 0, 4: 0}

    # ── AMBULANCE FIFO (highest priority) ──
    if ambulance_queue:
        # Auto-clear expired overrides (30 seconds)
        expired = []
        for entry in ambulance_queue:
            if entry.get("stage") == "override" and "override_start" in entry:
                elapsed_override = time.time() - entry["override_start"]
                if elapsed_override >= 30:
                    expired.append(entry)

        # Remove expired entries and reset their file uploaders
        for entry in expired:
            ambulance_queue.remove(entry)
            sig = entry["signal"]
            st.session_state.clear_counter[sig] = st.session_state.clear_counter.get(sig, 0) + 1
            st.session_state.processed_uploads = {
                uid for uid in st.session_state.processed_uploads
                if not uid.startswith(f"{sig}_")
            }
            # Resume from the interrupted signal (saved when override started)
            st.session_state.cycle_resume_time = time.time()
            resume_sig = st.session_state.get("interrupted_signal", 1)
            st.session_state.resume_from = resume_sig
            st.session_state.logs.insert(0,
                f"{datetime.now().strftime('%H:%M:%S')} | AUTO-CLEAR: Ambulance passed Signal {sig} (30s) - resuming from Signal {resume_sig}"
            )

        # Only signals with "override" (500m) status get green
        override_entries = [e for e in ambulance_queue if e.get("stage") == "override"]
        if override_entries:
            priority_entry = override_entries[0]
            priority_signal = priority_entry["signal"]
            phases[priority_signal] = "green"

            # Save which signal was interrupted (only on first override activation)
            if "interrupted_signal" not in st.session_state:
                # Figure out which signal WOULD be green right now in normal cycling
                normal_order = [1, 2, 3, 4]
                bkd = st.session_state.detected_breakdowns
                normal_durations = [(s, get_green_duration_for_weight(calculate_weight(bkd.get(s, {})))) for s in normal_order]
                normal_total = sum(d + yellow_dur for _, d in normal_durations)
                normal_elapsed = time.time() % normal_total
                cum = 0
                interrupted = 1
                for s, gt in normal_durations:
                    if cum <= normal_elapsed < cum + gt:
                        interrupted = s
                        break
                    cum += gt + yellow_dur
                st.session_state.interrupted_signal = interrupted

            # Show live countdown from 30s
            elapsed_so_far = time.time() - priority_entry.get("override_start", time.time())
            remaining = max(0, int(30 - elapsed_so_far))
            time_lefts[priority_signal] = remaining

            # Other override signals get estimated wait times
            for i, entry in enumerate(override_entries[1:], start=1):
                time_lefts[entry["signal"]] = remaining + i * 30

            # Non-ambulance signals show wait estimate
            total_amb_wait = remaining + (len(override_entries) - 1) * 30
            for sid in range(1, 5):
                if sid != priority_signal and not any(e["signal"] == sid for e in override_entries):
                    time_lefts[sid] = total_amb_wait + 5

            return phases, time_lefts
        else:
            # No more overrides — clear the interrupted signal tracker
            if "interrupted_signal" in st.session_state:
                del st.session_state.interrupted_signal

    # ── ALL 4 SIGNALS CYCLE ──
    # If resuming after ambulance, start from the INTERRUPTED signal
    signal_order = [1, 2, 3, 4]
    resume_from = st.session_state.get("resume_from", None)
    resume_time = st.session_state.get("cycle_resume_time", None)

    if resume_from and resume_time:
        # Reorder: start from the interrupted signal
        start_idx = resume_from - 1  # convert 1-based to 0-based
        signal_order = [(start_idx + i) % 4 + 1 for i in range(4)]
        # Use time since resume as the cycle clock
        elapsed_base = time.time() - resume_time
    else:
        elapsed_base = None

    # Build durations in fixed order — timing varies by WEIGHT
    breakdowns = st.session_state.detected_breakdowns
    signal_weights = {}
    for sid in signal_order:
        signal_weights[sid] = calculate_weight(breakdowns.get(sid, {}))

    # If all signals are equally crowded → use default equal timing (30s each)
    if all_weights_similar(signal_weights):
        durations = [(sid, 30) for sid in signal_order]
    else:
        durations = []
        for sid in signal_order:
            w = signal_weights[sid]
            green_time = get_green_duration_for_weight(w)
            durations.append((sid, green_time))

    total_cycle = sum(d + yellow_dur for _, d in durations)

    # ── STABLE CYCLE CLOCK ──
    # Track weight changes: when weights change, restart the cycle clock
    # so the new timing is immediately applied
    current_weight_sig = tuple(signal_weights.get(s, 0) for s in signal_order)
    prev_weight_sig = st.session_state.get("weight_signature", None)

    if prev_weight_sig != current_weight_sig:
        # Weights changed (new detection!) — reset the cycle clock
        st.session_state.weight_signature = current_weight_sig
        st.session_state.ai_cycle_start = time.time()
        st.session_state.completed_cycles = 0

    if "ai_cycle_start" not in st.session_state:
        st.session_state.ai_cycle_start = time.time()
    if "completed_cycles" not in st.session_state:
        st.session_state.completed_cycles = 0

    # Use resume-based elapsed time OR stable clock
    if elapsed_base is not None:
        raw_elapsed = elapsed_base
    else:
        raw_elapsed = time.time() - st.session_state.ai_cycle_start

    # Track cycle completion
    current_cycle_num = int(raw_elapsed // total_cycle) if total_cycle > 0 else 0
    elapsed = raw_elapsed % total_cycle

    # ── CYCLE COMPLETE → RESET DETECTIONS ──
    # After one full rotation of all 4 signals, clear old data
    # so next cycle uses fresh detections
    if current_cycle_num > st.session_state.completed_cycles:
        st.session_state.completed_cycles = current_cycle_num
        st.session_state.detected_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        st.session_state.detected_breakdowns = {1: {}, 2: {}, 3: {}, 4: {}}
        st.session_state.weight_signature = (0, 0, 0, 0)
        st.session_state.ai_cycle_start = time.time()
        st.session_state.completed_cycles = 0
        # Clear ALL uploaded images by incrementing clear counters
        for sid in range(1, 5):
            st.session_state.clear_counter[sid] = st.session_state.clear_counter.get(sid, 0) + 1
        st.session_state.processed_uploads = set()
        st.session_state.logs.insert(0,
            f"{datetime.now().strftime('%H:%M:%S')} | CYCLE COMPLETE - all images & detections reset - upload new images"
        )

    cumulative = 0
    for sid, green_time in durations:
        phase_end = cumulative + green_time + yellow_dur

        if cumulative <= elapsed < cumulative + green_time:
            phases[sid] = "green"
            time_lefts[sid] = int(cumulative + green_time - elapsed)
        elif cumulative + green_time <= elapsed < phase_end:
            phases[sid] = "yellow"
            time_lefts[sid] = int(phase_end - elapsed)
        else:
            phases[sid] = "red"
            if elapsed < cumulative:
                time_lefts[sid] = int(cumulative - elapsed)
            else:
                time_lefts[sid] = int(cumulative + total_cycle - elapsed)

        cumulative = phase_end

    return phases, time_lefts


def get_amb_state(sig_id, ambulance_queue):
    """Get ambulance state for a signal: None, 'early_warning', or 'override'."""
    for entry in ambulance_queue:
        if entry["signal"] == sig_id:
            return entry.get("stage", "override")
    return None


# ──────────────────────────────────────────────────────────────
# 4. SESSION STATE DEFAULTS
# ──────────────────────────────────────────────────────────────
if "logs" not in st.session_state:
    st.session_state.logs = [
        f"{datetime.now().strftime('%H:%M:%S')} | System initialised – AI detection online"
    ]
if "ambulance_queue" not in st.session_state:
    st.session_state.ambulance_queue = []  # [{signal, time, stage: 'early_warning' or 'override'}]
if "detected_counts" not in st.session_state:
    st.session_state.detected_counts = {1: 0, 2: 0, 3: 0, 4: 0}
if "detected_breakdowns" not in st.session_state:
    st.session_state.detected_breakdowns = {1: {}, 2: {}, 3: {}, 4: {}}
if "clear_counter" not in st.session_state:
    st.session_state.clear_counter = {1: 0, 2: 0, 3: 0, 4: 0}  # increments to reset file uploaders
if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = set()  # tracks which uploads already triggered ambulance

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
st.sidebar.subheader("Signal Mode")
signal_mode = st.sidebar.radio("Signal Control", ["🤖 AI-Driven (by vehicle count)", "⏱️ Time-Based (auto cycle)"], horizontal=False)

if signal_mode == "⏱️ Time-Based (auto cycle)":
    green_duration = st.sidebar.slider("Green Duration (seconds)", 10, 120, 30)
    yellow_duration = st.sidebar.slider("Yellow Duration (seconds)", 2, 10, 5)

auto_refresh = st.sidebar.toggle("Auto-Refresh Signals", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulate Events")
sim_signal = st.sidebar.selectbox("Choose Signal", [1, 2, 3, 4], key="sim_sig")

# ── 2000m Early Warning Button ──
if st.sidebar.button("📡 Ambulance at 2000m (Early Warning)"):
    already_in_queue = any(e["signal"] == sim_signal for e in st.session_state.ambulance_queue)
    if not already_in_queue:
        st.session_state.ambulance_queue.append({
            "signal": sim_signal,
            "time": datetime.now().strftime('%H:%M:%S'),
            "stage": "early_warning"
        })
        st.session_state.logs.insert(0,
            f"{datetime.now().strftime('%H:%M:%S')} | 📡 AMBULANCE detected 2000m from Signal {sim_signal} – EARLY WARNING – preparing route"
        )
    st.rerun()

# ── 500m Override Button ──
if st.sidebar.button("🚑 Ambulance at 500m (Override Signal)"):
    # Check if already in queue as early_warning — upgrade to override
    found = False
    for entry in st.session_state.ambulance_queue:
        if entry["signal"] == sim_signal:
            entry["stage"] = "override"
            if "override_start" not in entry:
                entry["override_start"] = time.time()
            found = True
            break
    if not found:
        st.session_state.ambulance_queue.append({
            "signal": sim_signal,
            "time": datetime.now().strftime('%H:%M:%S'),
            "stage": "override",
            "override_start": time.time()
        })
    queue_pos = next((i+1 for i, e in enumerate(st.session_state.ambulance_queue)
                      if e["signal"] == sim_signal and e["stage"] == "override"), "?")
    st.session_state.logs.insert(0,
        f"{datetime.now().strftime('%H:%M:%S')} | 🚑 AMBULANCE at 500m Signal {sim_signal} – OVERRIDE ACTIVE (FIFO #{queue_pos})"
    )
    st.rerun()

# ── Clear First in Queue ──
if st.sidebar.button("✅ Ambulance Passed (Clear First)"):
    # Remove the first override entry (FIFO)
    override_idx = None
    for i, entry in enumerate(st.session_state.ambulance_queue):
        if entry.get("stage") == "override":
            override_idx = i
            break
    if override_idx is not None:
        cleared = st.session_state.ambulance_queue.pop(override_idx)
        cleared_sig = cleared["signal"]
        # Increment clear counter to reset file uploaders (image disappears)
        st.session_state.clear_counter[cleared_sig] = st.session_state.clear_counter.get(cleared_sig, 0) + 1
        # Clear processed uploads for this signal so new uploads can trigger again
        st.session_state.processed_uploads = {
            uid for uid in st.session_state.processed_uploads
            if not uid.startswith(f"{cleared_sig}_")
        }
        # Check if next override exists
        next_override = next((e for e in st.session_state.ambulance_queue if e.get("stage") == "override"), None)
        next_msg = f" → Signal {next_override['signal']} is NEXT" if next_override else " → Resuming NORMAL signal cycling"
        st.session_state.logs.insert(0,
            f"{datetime.now().strftime('%H:%M:%S')} | ✅ Ambulance cleared at Signal {cleared_sig}{next_msg}"
        )
    st.rerun()

# ── Clear ALL ──
if st.sidebar.button("🗑️ Clear ALL Alerts"):
    # Reset all clear counters to remove images
    for sig_id in range(1, 5):
        st.session_state.clear_counter[sig_id] = st.session_state.clear_counter.get(sig_id, 0) + 1
    st.session_state.ambulance_queue = []
    st.session_state.processed_uploads = set()  # Reset so new uploads can trigger
    st.session_state.logs.insert(0,
        f"{datetime.now().strftime('%H:%M:%S')} | 🗑️ All ambulance alerts cleared – NORMAL cycling resumed"
    )
    st.rerun()

# ── Show Ambulance Queue in Sidebar ──
if st.session_state.ambulance_queue:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚑 Ambulance Queue (FIFO)")
    for i, entry in enumerate(st.session_state.ambulance_queue):
        stage = entry.get("stage", "override")
        if stage == "override":
            icon = "🟢" if i == 0 else "🔴"
            label = "OVERRIDE"
        else:
            icon = "🟡"
            label = "2000m WARNING"
        st.sidebar.markdown(f"{icon} **#{i+1}** Signal {entry['signal']} — {label} *(at {entry['time']})*")

if st.sidebar.button("🔄 Reset Detection Counts"):
    st.session_state.detected_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    st.session_state.detected_breakdowns = {1: {}, 2: {}, 3: {}, 4: {}}
    st.session_state.logs.insert(0,
        f"{datetime.now().strftime('%H:%M:%S')} | Detection counts reset – all signals back to default timing"
    )
    st.rerun()

# ──────────────────────────────────────────────────────────────
# 6. COMPUTE CURRENT SIGNAL PHASES
# ──────────────────────────────────────────────────────────────

if signal_mode == "🤖 AI-Driven (by vehicle count)":
    phases, time_lefts = compute_ai_signal_phases(
        st.session_state.detected_counts,
        st.session_state.ambulance_queue
    )
else:
    # Time-based cycling — all 4 signals always rotate
    total_cycle = 4 * (green_duration + yellow_duration)
    elapsed = time.time() % total_cycle

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
            if elapsed < sig_start:
                time_lefts[sig_id] = int(sig_start - elapsed)
            else:
                time_lefts[sig_id] = int(sig_start + total_cycle - elapsed)

    # Ambulance FIFO override in time-based mode too
    override_entries = [e for e in st.session_state.ambulance_queue if e.get("stage") == "override"]
    if override_entries:
        # Override signal gets GREEN, all others RED (keep their time values)
        for sig_id in range(1, 5):
            phases[sig_id] = "red"
        priority = override_entries[0]["signal"]
        phases[priority] = "green"
        time_lefts[priority] = 0

# Vehicle counts
vehicle_counts = st.session_state.detected_counts

# ──────────────────────────────────────────────────────────────
# 7. HEADER & GLOBAL METRICS
# ──────────────────────────────────────────────────────────────
st.title("🚦 AI Traffic Management System")
st.caption("Real-time vehicle detection • AI-driven signal control • 2000m ambulance pre-detection")

total_vehicles = sum(vehicle_counts.values())
active_amb = len(st.session_state.ambulance_queue)
green_signal = [k for k, v in phases.items() if v == "green"]
green_label = f"Signal {green_signal[0]}" if green_signal else "None"
mode_label = "🤖 AI" if signal_mode.startswith("🤖") else "⏱️ Timer"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Vehicles", total_vehicles)
k2.metric("Ambulance Alerts", active_amb, delta="⚠ PRIORITY" if active_amb > 0 else "✓ Normal")
k3.metric("Current Green", green_label)
k4.metric("Signal Mode", mode_label)

st.markdown("---")

# ──────────────────────────────────────────────────────────────
# 8. GRID VIEW – all 4 signals with 3 cameras each
# ──────────────────────────────────────────────────────────────
UPLOAD_TYPES = ["mp4", "avi", "mov", "png", "jpg", "jpeg", "webp"]

def render_signal_grid(col, sig_id):
    """Render one signal block with 3 camera feeds + traffic light."""
    with col:
        amb_state = get_amb_state(sig_id, st.session_state.ambulance_queue)
        breakdown = st.session_state.detected_breakdowns.get(sig_id) or None
        sig_weight = calculate_weight(breakdown or {})
        st.markdown(render_signal_card(
            sig_id, phases[sig_id], vehicle_counts[sig_id],
            amb_state, time_lefts[sig_id], breakdown=breakdown, weight=sig_weight
        ), unsafe_allow_html=True)

        feed_a, feed_b, feed_c, light_col = st.columns([2, 2, 2, 1])

        # Camera 1: At Signal
        with feed_a:
            st.markdown(f"<p style='color:#9ca3af; font-size:0.75rem;'>📷 At Signal {sig_id}</p>", unsafe_allow_html=True)
            if input_mode == "Upload Image/Video":
                cc = st.session_state.clear_counter.get(sig_id, 0)
                uploaded = st.file_uploader(f"Signal {sig_id} – At Signal", type=UPLOAD_TYPES, key=f"sig_{sig_id}_at_{cc}")
                ph = st.empty()
                if uploaded:
                    display_and_detect(uploaded, ph, sig_id)
                else:
                    st.image(f"https://via.placeholder.com/300x180.png?text=Signal+{sig_id}+Camera", use_container_width=True)
            else:
                st.image(f"https://via.placeholder.com/300x180.png?text=Signal+{sig_id}+Live", use_container_width=True)

        # Camera 2: 500m (Ambulance Detection)
        with feed_b:
            st.markdown(f"<p style='color:#9ca3af; font-size:0.75rem;'>📷 500m Before Signal {sig_id}</p>", unsafe_allow_html=True)
            if input_mode == "Upload Image/Video":
                cc = st.session_state.clear_counter.get(sig_id, 0)
                uploaded_500 = st.file_uploader(f"Signal {sig_id} – 500m", type=UPLOAD_TYPES, key=f"sig_{sig_id}_500m_{cc}")
                ph2 = st.empty()
                if uploaded_500:
                    display_and_detect_ambulance(uploaded_500, ph2, sig_id, "500m")
                else:
                    st.image(f"https://via.placeholder.com/300x180.png?text=500m+Signal+{sig_id}", use_container_width=True)
            else:
                st.image(f"https://via.placeholder.com/300x180.png?text=500m+Cam+{sig_id}", use_container_width=True)

        # Camera 3: 2000m (Early Ambulance Detection)
        with feed_c:
            st.markdown(f"<p style='color:#d97706; font-size:0.75rem;'>📡 2000m Before Signal {sig_id}</p>", unsafe_allow_html=True)
            if input_mode == "Upload Image/Video":
                cc = st.session_state.clear_counter.get(sig_id, 0)
                uploaded_2k = st.file_uploader(f"Signal {sig_id} – 2000m", type=UPLOAD_TYPES, key=f"sig_{sig_id}_2000m_{cc}")
                ph3 = st.empty()
                if uploaded_2k:
                    display_and_detect_ambulance(uploaded_2k, ph3, sig_id, "2000m")
                else:
                    st.image(f"https://via.placeholder.com/300x180.png?text=2000m+Signal+{sig_id}", use_container_width=True)
            else:
                st.image(f"https://via.placeholder.com/300x180.png?text=2000m+Cam+{sig_id}", use_container_width=True)

        # Traffic Light
        with light_col:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(render_traffic_light(phases[sig_id]), unsafe_allow_html=True)


if view_mode == "Grid Overview":
    # Row 1: Signal 1 & 2
    row1_left, row1_right = st.columns(2)
    render_signal_grid(row1_left, 1)
    render_signal_grid(row1_right, 2)

    # Row 2: Signal 3 & 4
    row2_left, row2_right = st.columns(2)
    render_signal_grid(row2_left, 3)
    render_signal_grid(row2_right, 4)

# ──────────────────────────────────────────────────────────────
# 9. SINGLE SIGNAL FOCUS VIEW
# ──────────────────────────────────────────────────────────────
else:
    focus_id = int(view_mode.split()[1])
    amb_state = get_amb_state(focus_id, st.session_state.ambulance_queue)
    breakdown = st.session_state.detected_breakdowns.get(focus_id) or None
    focus_weight = calculate_weight(breakdown or {})

    st.subheader(f"Signal {focus_id} – Detailed View")
    st.markdown(render_signal_card(
        focus_id, phases[focus_id], vehicle_counts[focus_id],
        amb_state, time_lefts[focus_id], breakdown=breakdown, weight=focus_weight
    ), unsafe_allow_html=True)

    cam1, cam2, cam3 = st.columns(3)

    with cam1:
        st.markdown(f"### 📷 At Signal {focus_id}")
        feed_ph = st.empty()
        if input_mode == "Upload Image/Video":
            cc = st.session_state.clear_counter.get(focus_id, 0)
            vid_at = st.file_uploader(f"Upload Signal {focus_id} – At Signal", type=UPLOAD_TYPES, key=f"focus_{focus_id}_at_{cc}")
            if vid_at:
                display_and_detect(vid_at, feed_ph, focus_id)
            else:
                st.info("Upload an image or video to start AI analysis.")
        else:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    result = detect_vehicles(frame)
                    annotated = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
                    feed_ph.image(annotated, channels="RGB", use_container_width=True)
                    st.session_state.detected_counts[focus_id] = result["count"]
                cap.release()
            else:
                st.error("Cannot access webcam (Camera 0).")

    with cam2:
        st.markdown(f"### 📷 500m – Signal {focus_id}")
        feed_500 = st.empty()
        if input_mode == "Upload Image/Video":
            cc = st.session_state.clear_counter.get(focus_id, 0)
            vid_500 = st.file_uploader(f"Upload Signal {focus_id} – 500m", type=UPLOAD_TYPES, key=f"focus_{focus_id}_500m_{cc}")
            if vid_500:
                display_and_detect_ambulance(vid_500, feed_500, focus_id, "500m")
            else:
                st.info("Upload a 500m camera feed.")
        else:
            st.warning("Connect 500m camera for ambulance detection.")

    with cam3:
        st.markdown(f"### 📡 2000m – Signal {focus_id}")
        feed_2k = st.empty()
        if input_mode == "Upload Image/Video":
            cc = st.session_state.clear_counter.get(focus_id, 0)
            vid_2k = st.file_uploader(f"Upload Signal {focus_id} – 2000m", type=UPLOAD_TYPES, key=f"focus_{focus_id}_2000m_{cc}")
            if vid_2k:
                display_and_detect_ambulance(vid_2k, feed_2k, focus_id, "2000m")
            else:
                st.info("Upload a 2000m early detection feed.")
        else:
            st.warning("Connect 2000m camera for early ambulance warning.")

    # Signal controls
    st.markdown("---")
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        st.markdown(render_traffic_light(phases[focus_id]), unsafe_allow_html=True)
    with ctrl2:
        st.metric("Current Phase", phases[focus_id].upper())
        st.metric("Time Remaining", f"{time_lefts[focus_id]}s")
    with ctrl3:
        if breakdown:
            st.markdown("**🔍 AI Detection Breakdown:**")
            for cls_name, cnt in breakdown.items():
                st.markdown(f"- **{cls_name}**: {cnt}")
        else:
            st.metric("Queue Length", f"{vehicle_counts[focus_id] * 7}m")

# ──────────────────────────────────────────────────────────────
# 10. AI DETECTION SUMMARY
# ──────────────────────────────────────────────────────────────
has_any_detection = any(v > 0 for v in st.session_state.detected_counts.values())
if has_any_detection:
    st.markdown("---")
    st.subheader("📊 AI Detection Summary")
    summary_cols = st.columns(4)
    for i, col in enumerate(summary_cols):
        sig_id = i + 1
        count = st.session_state.detected_counts[sig_id]
        bd = st.session_state.detected_breakdowns.get(sig_id, {})
        with col:
            phase_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(phases[sig_id], "⚪")
            st.markdown(f"**{phase_emoji} Signal {sig_id}**")
            st.metric(f"Vehicles", count)
            if bd:
                for cls_name, cnt in bd.items():
                    st.caption(f"{cls_name}: {cnt}")
            else:
                st.caption("No detection — default 15s green")

# ──────────────────────────────────────────────────────────────
# 11. ACTIVITY LOG
# ──────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📜 Activity Log", expanded=False):
    for log in st.session_state.logs[:20]:
        st.text(log)

# ──────────────────────────────────────────────────────────────
# 12. AUTO-REFRESH
# ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(2)
    st.rerun()

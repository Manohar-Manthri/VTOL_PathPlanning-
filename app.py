import streamlit as st
import xml.etree.ElementTree as ET
from geopy.distance import geodesic, distance
from geopy import Point
import numpy as np
import requests
import os
import time
import json
import pickle
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# Backend functions
def parse_kml(kml_file):
    """Extract (lat, lon) pairs from the first LineString in a KML, excluding the first coordinate."""
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        for ls in root.findall('.//kml:LineString', ns):
            coords = []
            coord_text = ls.find('kml:coordinates', ns).text.strip().split()
            for c in coord_text[1:]:  # Skip first coordinate
                lon, lat, *_ = map(float, c.split(','))
                coords.append((lat, lon))
            return coords
        return []
    except Exception as e:
        st.error(f"Error parsing KML: {e}")
        return []

def calculate_angle(p1, p2, p3):
    a = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    b = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 180.0
    return np.degrees(np.arccos(np.clip(np.dot(a, b)/(na*nb), -1,1)))

def split_lines_by_turn(coords, thresh=170):
    segments, cur = [], [coords[0]]
    for i in range(1, len(coords)-1):
        cur.append(coords[i])
        if calculate_angle(coords[i-1], coords[i], coords[i+1]) < thresh:
            segments.append(cur)
            cur = [coords[i]]
    cur.append(coords[-1])
    segments.append(cur)
    return segments

def filter_main_lines(lines, min_len=500):
    out = []
    for seg in lines:
        d = sum(geodesic(seg[i], seg[i+1]).meters for i in range(len(seg)-1))
        if d >= min_len and len(seg) >= 2:
            out.append(seg)
    return out

def adjust_line_directions(lines):
    """Adjust each survey line's direction to start near the previous line's end."""
    if len(lines) > 1:
        for i in range(1, len(lines)):
            prev_end = lines[i-1][-1]
            curr_start = lines[i][0]
            curr_end = lines[i][-1]
            if geodesic(prev_end, curr_start).meters > geodesic(prev_end, curr_end).meters:
                lines[i].reverse()
    return lines

def fetch_elevations(coords, cache_file="elevations_cache.pkl"):
    cache = {}
    current_time = time.time()
    expiration_seconds = 24 * 60 * 60  # 1 day in seconds
    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            cache_time = cached_data.get('timestamp', 0)
            if current_time - cache_time <= expiration_seconds:
                cache = cached_data.get('elevations', {})
    except FileNotFoundError:
        pass
    
    uncached_coords = [c for c in coords if c not in cache]
    if uncached_coords:
        elevs = []
        url = 'https://api.open-elevation.com/api/v1/lookup'
        for i in range(0, len(uncached_coords), 100):
            batch = uncached_coords[i:i+100]
            locs = [{'latitude': lat, 'longitude': lon} for lat, lon in batch]
            try:
                r = requests.post(url, json={'locations': locs}, timeout=10)
                r.raise_for_status()
                results = r.json().get('results', [])
                elevs += [pt['elevation'] for pt in results]
                for coord, elev in zip(batch, elevs[-len(batch):]):
                    cache[coord] = elev
            except requests.RequestException as e:
                st.warning(f"Failed to fetch elevations for batch {i//100 + 1}: {e}. Using 0m.")
                elevs += [0] * len(batch)
            time.sleep(0.3)
        with open(cache_file, 'wb') as f:
            pickle.dump({'timestamp': current_time, 'elevations': cache}, f)
    
    return [cache.get(coord, 0) for coord in coords]

def calculate_bearing(p1, p2):
    lat1, lat2 = np.radians(p1[0]), np.radians(p2[0])
    dlon = np.radians(p2[1] - p1[1])
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def move_point_along_bearing(start, bearing, dist_m):
    origin = Point(start[0], start[1])
    dest = distance(meters=dist_m).destination(origin, bearing)
    return (dest.latitude, dest.longitude)

def create_trigger_item(lat, lon, alt, trigger_type, trigger_distance, item_id):
    if trigger_type == "camera":
        return {
            "AMSLAltAboveTerrain": None,
            "Altitude": alt,
            "AltitudeMode": 1,
            "autoContinue": True,
            "command": 206,
            "doJumpId": item_id,
            "frame": 3,
            "params": [
                trigger_distance,
                0,
                1,
                1,
                0, 0, 0
            ],
            "type": "SimpleItem"
        }
    return None

def generate_simplified_path(lines, home_pt, rtl_pt, home_alt, safety_margin=320, turning_length=320, trigger_distance=40, end_trigger_distance=0):
    path = []
    trigger_points = []
    item_id_counter = 1
    trigger_config = {
        "line_start": "speed",
        "line_start_params": {"speed": 15},
        "line_end": "none",
        "line_end_params": {}
    }
    expected_photo_count = len(lines) * 2

    # 1) Survey-start orientation
    survey_start = lines[0][0]
    survey_end = lines[0][-1]
    h5_temp = home_pt
    for dist, _ in [(300,90),(500,180),(700,300),(700,500),(700,800)]:
        brg = calculate_bearing(h5_temp, survey_start)
        h5_temp = move_point_along_bearing(h5_temp, brg, dist)
    if geodesic(h5_temp, survey_end).meters < geodesic(h5_temp, survey_start).meters:
        lines[0].reverse()
        survey_start = lines[0][0]

    # 2) Modified VTOL takeoff ladder with loiter point
    takeoff_ladder = [
        (0,   home_alt),  # Home point with user-defined altitude
        (100,  90),       # Second point
        (200, 180),       # Third point
    ]
    prev_wp = home_pt
    path.append((prev_wp[0], prev_wp[1], takeoff_ladder[0][1]))
    trigger_points.append({
        "lat": prev_wp[0], "lon": prev_wp[1], "alt": takeoff_ladder[0][1],
        "trigger_type": "none",
        "trigger_params": {}
    })
    item_id_counter += 1

    for dist, alt in takeoff_ladder[1:]:
        brg = calculate_bearing(prev_wp, survey_start)
        prev_wp = move_point_along_bearing(prev_wp, brg, dist)
        path.append((prev_wp[0], prev_wp[1], alt))
        trigger_points.append({
            "lat": prev_wp[0], "lon": prev_wp[1], "alt": alt,
            "trigger_type": "none",
            "trigger_params": {}
        })
        item_id_counter += 1

    point3 = prev_wp
    first_seg = lines[0]
    a = first_seg[0]
    b = first_seg[-1]
    brg_seg = calculate_bearing(a, b)
    entry = move_point_along_bearing(a, (brg_seg + 180) % 360, turning_length)

    num_samples = 50
    total_dist = geodesic(point3, entry).meters
    brg_to_entry = calculate_bearing(point3, entry)
    sample_coords = []
    for i in range(num_samples + 1):
        dist = total_dist * (i / num_samples)
        intermediate = move_point_along_bearing(point3, brg_to_entry, dist)
        sample_coords.append(intermediate)
    elevs = fetch_elevations(sample_coords)
    if elevs:
        max_elev_idx = np.argmax(elevs)
        max_elev = elevs[max_elev_idx]
        max_elev_coords = sample_coords[max_elev_idx]
        safe_alt = max_elev + 150
    else:
        max_elev_coords = point3
        safe_alt = 150

    path.append((point3[0], point3[1], safe_alt))
    trigger_points.append({
        "lat": point3[0], "lon": point3[1], "alt": safe_alt,
        "trigger_type": "loiter",
        "trigger_params": {}
    })
    item_id_counter += 1

    path.append((max_elev_coords[0], max_elev_coords[1], safe_alt))
    trigger_points.append({
        "lat": max_elev_coords[0], "lon": max_elev_coords[1], "alt": safe_alt,
        "trigger_type": "none",
        "trigger_params": {}
    })
    item_id_counter += 1

    ev_first = fetch_elevations(first_seg)
    cruise_first = max(max(ev_first) + safety_margin, 180)
    pre_entry_loiter = move_point_along_bearing(entry, (brg_to_entry + 180) % 360, 500)

    path.append((pre_entry_loiter[0], pre_entry_loiter[1], safe_alt))
    trigger_points.append({
        "lat": pre_entry_loiter[0], "lon": pre_entry_loiter[1], "alt": safe_alt,
        "trigger_type": "none",
        "trigger_params": {}
    })
    item_id_counter += 1

    path.append((pre_entry_loiter[0], pre_entry_loiter[1], cruise_first))
    trigger_points.append({
        "lat": pre_entry_loiter[0], "lon": pre_entry_loiter[1], "alt": cruise_first,
        "trigger_type": "loiter",
        "trigger_params": {}
    })
    item_id_counter += 1

    # 3) Survey strips with camera triggers
    prev_exit, prev_alt = None, cruise_first
    for seg in lines:
        ev = fetch_elevations(seg)
        cruise = max(max(ev) + safety_margin, 180)
        a, b = seg[0], seg[-1]
        brg = calculate_bearing(a, b)
        entry = move_point_along_bearing(a, (brg+180)%360, turning_length)
        exitpt = move_point_along_bearing(b, brg, turning_length)

        if prev_exit:
            path.append((prev_exit[0], prev_exit[1], prev_alt))
            trigger_points.append({
                "lat": prev_exit[0], "lon": prev_exit[1], "alt": prev_alt,
                "trigger_type": "none",
                "trigger_params": {}
            })
            item_id_counter += 1
            path.append((entry[0], entry[1], cruise))
            trigger_points.append({
                "lat": entry[0], "lon": entry[1], "alt": cruise,
                "trigger_type": "none",
                "trigger_params": {}
            })
            item_id_counter += 1

        path.extend([
            (entry[0], entry[1], cruise),
            (a[0], a[1], cruise),
            (b[0], b[1], cruise),
            (exitpt[0], exitpt[1], cruise),
        ])
        trigger_points.extend([
            {
                "lat": entry[0], "lon": entry[1], "alt": cruise,
                "trigger_type": "none",
                "trigger_params": {}
            },
            {
                "lat": a[0], "lon": a[1], "alt": cruise,
                "trigger_type": "camera",
                "trigger_params": {"distance": trigger_distance}
            },
            {
                "lat": b[0], "lon": b[1], "alt": cruise,
                "trigger_type": "camera",
                "trigger_params": {"distance": end_trigger_distance}
            },
            {
                "lat": exitpt[0], "lon": exitpt[1], "alt": cruise,
                "trigger_type": "none",
                "trigger_params": {}
            },
        ])
        item_id_counter += 4
        prev_exit, prev_alt = exitpt, cruise

    # 4) RTL descent ladder
    exit_pt = (prev_exit[0], prev_exit[1])
    brg_to_home = calculate_bearing(exit_pt, home_pt)
    direct_dist = geodesic(exit_pt, home_pt).meters

    num_samples = 50
    sample_coords = []
    for i in range(num_samples + 1):
        dist = direct_dist * (i / num_samples)
        intermediate = move_point_along_bearing(exit_pt, brg_to_home, dist)
        sample_coords.append(intermediate)
    elevs = fetch_elevations(sample_coords)
    if elevs:
        max_elev_idx = np.argmax(elevs)
        max_elev = elevs[max_elev_idx]
        max_elev_coords = sample_coords[max_elev_idx]
        new_alt = max_elev + 150
    else:
        max_elev_coords = home_pt
        new_alt = 150

    path.append((exit_pt[0], exit_pt[1], new_alt))
    trigger_points.append({
        "lat": exit_pt[0], "lon": exit_pt[1], "alt": new_alt,
        "trigger_type": "loiter",
        "trigger_params": {}
    })
    item_id_counter += 1

    path.append((max_elev_coords[0], max_elev_coords[1], new_alt))
    trigger_points.append({
        "lat": max_elev_coords[0], "lon": max_elev_coords[1], "alt": new_alt,
        "trigger_type": "none",
        "trigger_params": {}
    })
    item_id_counter += 1

    inverse_brg = (brg_to_home + 180) % 360
    loiter_180m_pos = move_point_along_bearing(home_pt, inverse_brg, 500)

    path.append((loiter_180m_pos[0], loiter_180m_pos[1], new_alt))
    trigger_points.append({
        "lat": loiter_180m_pos[0], "lon": loiter_180m_pos[1], "alt": new_alt,
        "trigger_type": "none",
        "trigger_params": {}
    })
    item_id_counter += 1

    path.append((loiter_180m_pos[0], loiter_180m_pos[1], 180))
    trigger_points.append({
        "lat": loiter_180m_pos[0], "lon": loiter_180m_pos[1], "alt": 180,
        "trigger_type": "loiter",
        "trigger_params": {}
    })
    item_id_counter += 1
    
    landing_ladder = [
        (200, 90),
        (100, 40),
    ]
    for dist_from_home, alt in landing_ladder:
        pos = move_point_along_bearing(home_pt, inverse_brg, dist_from_home)
        path.append((pos[0], pos[1], alt))
        trigger_points.append({
            "lat": pos[0], "lon": pos[1], "alt": alt,
            "trigger_type": "none",
            "trigger_params": {}
        })
        item_id_counter += 1

    path.append((home_pt[0], home_pt[1], 0))
    trigger_points.append({
        "lat": home_pt[0], "lon": home_pt[1], "alt": 0,
        "trigger_type": "none",
        "trigger_params": {}
    })
    item_id_counter += 1

    return path, trigger_points, expected_photo_count

def write_qgc_plan(points, trigger_points, output_file, expected_photo_count, trigger_distance, end_trigger_distance):
    items = []
    item_id = 1

    items.append({
        "AMSLAltAboveTerrain": None,
        "Altitude": points[0][2],
        "AltitudeMode": 1,
        "autoContinue": True,
        "command": 84,
        "doJumpId": item_id,
        "frame": 3,
        "params": [0, 0, 0, None, 0, 0, points[0][2]],
        "type": "SimpleItem"
    })
    item_id += 1

    for i, (lat, lon, alt) in enumerate(points[1:], start=1):
        trigger = trigger_points[i]
        if trigger["trigger_type"] == "loiter":
            items.append({
                "Altitude": alt,
                "AMSLAltAboveTerrain": None,
                "AltitudeMode": 1,
                "autoContinue": True,
                "command": 31,
                "doJumpId": item_id,
                "frame": 3,
                "params": [1, 200, 0, 1, lat, lon, alt],
                "type": "SimpleItem"
            })
            item_id += 1
        else:
            items.append({
                "Altitude": alt,
                "AMSLAltAboveTerrain": alt,
                "AltitudeMode": 1,
                "autoContinue": True,
                "command": 16,
                "doJumpId": item_id,
                "frame": 3,
                "params": [0, 0, 0, None, lat, lon, alt],
                "type": "SimpleItem"
            })
            item_id += 1

            if trigger["trigger_type"] == "camera":
                distance = trigger["trigger_params"].get("distance", trigger_distance)
                trigger_item = create_trigger_item(
                    trigger["lat"], trigger["lon"], trigger["alt"],
                    trigger["trigger_type"], distance, item_id
                )
                if trigger_item:
                    items.append(trigger_item)
                    item_id += 1

    items.append({
        "Altitude": 0,
        "AMSLAltAboveTerrain": 0,
        "AltitudeMode": 1,
        "autoContinue": True,
        "command": 20,
        "doJumpId": item_id,
        "frame": 3,
        "params": [0, 0, 0, 0, 0, 0, 0],
        "type": "SimpleItem"
    })
    item_id += 1

    plan = {
        "fileType": "Plan",
        "version": 1,
        "groundStation": "QGroundControl",
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "rallyPoints": {"points": [], "version": 2},
        "mission": {
            "version": 2,
            "firmwareType": 3,
            "vehicleType": 1,
            "cruiseSpeed": 20,
            "hoverSpeed": 5,
            "plannedHomePosition": [points[0][0], points[0][1], points[0][2]],
            "items": items,
            "surveyStats": {
                "surveyArea": 0,
                "triggerDistance": trigger_distance,
                "photoCount": expected_photo_count
            }
        }
    }
    return plan

# Streamlit UI
st.set_page_config(page_title="Flight Plan Generator", layout="wide")
st.title("Flight Plan Generator")

# Initialize session state
if 'plan' not in st.session_state:
    st.session_state.plan = None
if 'path' not in st.session_state:
    st.session_state.path = None
if 'trigger_points' not in st.session_state:
    st.session_state.trigger_points = None
if 'expected_photo_count' not in st.session_state:
    st.session_state.expected_photo_count = 0
if 'home_lat' not in st.session_state:
    st.session_state.home_lat = 0.0
if 'home_lon' not in st.session_state:
    st.session_state.home_lon = 0.0
if 'rtl_lat' not in st.session_state:
    st.session_state.rtl_lat = 0.0
if 'rtl_lon' not in st.session_state:
    st.session_state.rtl_lon = 0.0
if 'start_trigger' not in st.session_state:
    st.session_state.start_trigger = 40.0
if 'end_trigger' not in st.session_state:
    st.session_state.end_trigger = 0.0
if 'safety_margin' not in st.session_state:
    st.session_state.safety_margin = 320.0
if 'home_alt' not in st.session_state:
    st.session_state.home_alt = 40.0
if 'kml_coords' not in st.session_state:
    st.session_state.kml_coords = None

# Tabs
tab1, tab2 = st.tabs(["Input & Settings", "Statistics & Visualization"])

# Tab 1: Input & Settings
with tab1:
    st.header("Input & Settings")
    st.subheader("KML Import")
    kml_file = st.file_uploader("Upload a KML file", type=["kml"], key="kml_uploader")
    
    st.subheader("Home Position")
    col1, col2 = st.columns(2)
    with col1:
        home_lat_num = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=st.session_state.home_lat, format="%.6f", key="home_lat_num")
        st.session_state.home_lat = home_lat_num
    with col2:
        home_lon_num = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=st.session_state.home_lon, format="%.6f", key="home_lon_num")
        st.session_state.home_lon = home_lon_num
    
    st.subheader("RTL Position")
    col5, col6 = st.columns(2)
    with col5:
        rtl_lat_num = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=st.session_state.rtl_lat, format="%.6f", key="rtl_lat_num")
        st.session_state.rtl_lat = rtl_lat_num
    with col6:
        rtl_lon_num = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=st.session_state.rtl_lon, format="%.6f", key="rtl_lon_num")
        st.session_state.rtl_lon = rtl_lon_num
    
    st.subheader("VTOL Takeoff Altitude")
    col7, col8 = st.columns(2)
    with col7:
        home_alt_num = st.number_input("Takeoff Altitude (m)", min_value=20.0, max_value=100.0, value=st.session_state.home_alt, key="home_alt_num")
        st.session_state.home_alt = home_alt_num
    with col8:
        home_alt_slider = st.slider("Takeoff Altitude (slider)", min_value=20.0, max_value=100.0, value=st.session_state.home_alt, key="home_alt_slider")
        if home_alt_slider != st.session_state.home_alt:
            st.session_state.home_alt = home_alt_slider
    
    st.subheader("Camera Trigger Settings")
    col9, col10 = st.columns(2)
    with col9:
        start_trigger_num = st.number_input("Start Trigger Distance (m)", min_value=0.0, max_value=100.0, value=st.session_state.start_trigger, key="start_trigger_num")
        st.session_state.start_trigger = start_trigger_num
    with col10:
        start_trigger_slider = st.slider("Start Trigger (slider)", min_value=0.0, max_value=100.0, value=st.session_state.start_trigger, key="start_trigger_slider")
        if start_trigger_slider != st.session_state.start_trigger:
            st.session_state.start_trigger = start_trigger_slider
    
    col11, col12 = st.columns(2)
    with col11:
        end_trigger_num = st.number_input("End Trigger Distance (m)", min_value=0.0, max_value=100.0, value=st.session_state.end_trigger, key="end_trigger_num")
        st.session_state.end_trigger = end_trigger_num
    with col12:
        end_trigger_slider = st.slider("End Trigger (slider)", min_value=0.0, max_value=100.0, value=st.session_state.end_trigger, key="end_trigger_slider")
        if end_trigger_slider != st.session_state.end_trigger:
            st.session_state.end_trigger = end_trigger_slider
    
    st.subheader("Safety Margin")
    col13, col14 = st.columns(2)
    with col13:
        safety_margin_num = st.number_input("Safety Margin (m)", min_value=100.0, max_value=500.0, value=st.session_state.safety_margin, key="safety_margin_num")
        st.session_state.safety_margin = safety_margin_num
    with col14:
        safety_margin_slider = st.slider("Safety Margin (slider)", min_value=100.0, max_value=500.0, value=st.session_state.safety_margin, key="safety_margin_slider")
        if safety_margin_slider != st.session_state.safety_margin:
            st.session_state.safety_margin = safety_margin_slider
    
    reverse_kml = st.checkbox("Reverse KML Direction", value=False, key="reverse_kml")
    
    if st.button("Generate Flight Plan", key="generate_button"):
        if kml_file is None and st.session_state.kml_coords is None:
            st.error("Please upload a KML file.")
        elif st.session_state.home_lat == 0.0 or st.session_state.home_lon == 0.0 or st.session_state.rtl_lat == 0.0 or st.session_state.rtl_lon == 0.0:
            st.error("Please provide valid Home and RTL coordinates.")
        else:
            with st.spinner("Generating flight plan..."):
                home_pt = (st.session_state.home_lat, st.session_state.home_lon)
                rtl_pt = (st.session_state.rtl_lat, st.session_state.rtl_lon)
                if kml_file is not None:
                    coords = parse_kml(kml_file)
                    st.session_state.kml_coords = coords
                else:
                    coords = st.session_state.kml_coords
                if reverse_kml:
                    coords = coords.copy()
                    coords.reverse()
                if not coords:
                    st.error("No valid survey lines found in KML.")
                else:
                    segs = split_lines_by_turn(coords)
                    mains = filter_main_lines(segs)
                    if not mains:
                        st.error("No survey lines meet the minimum length requirement.")
                    else:
                        mains = adjust_line_directions(mains)
                        path, trigger_points, expected_photo_count = generate_simplified_path(
                            mains, home_pt, rtl_pt,
                            home_alt=st.session_state.home_alt,
                            safety_margin=st.session_state.safety_margin,
                            trigger_distance=st.session_state.start_trigger,
                            end_trigger_distance=st.session_state.end_trigger
                        )
                        plan = write_qgc_plan(path, trigger_points, "flight_plan.plan", expected_photo_count, st.session_state.start_trigger, st.session_state.end_trigger)
                        st.session_state.plan = plan
                        st.session_state.path = path
                        st.session_state.trigger_points = trigger_points
                        st.session_state.expected_photo_count = expected_photo_count
                        st.success("Flight plan generated successfully! Check the Statistics & Visualization tab.")

# Tab 2: Statistics & Visualization
with tab2:
    st.header("Mission Statistics")
    if st.session_state.path:
        total_distance = sum(geodesic(st.session_state.path[i][:2], st.session_state.path[i+1][:2]).meters 
                            for i in range(len(st.session_state.path)-1))
        cruise_speed = 15
        flight_time = total_distance / cruise_speed / 60
        st.write(f"**Flight Time**: {flight_time:.2f} minutes")
        st.write(f"**Distance**: {total_distance/1000:.2f} km")
    else:
        st.write("Generate a flight plan to view statistics.")
    
    st.header("Plan View (Satellite)")
    if st.session_state.path:
        try:
            invalid_coords = [(i, p) for i, p in enumerate(st.session_state.path) if not (-90 <= p[0] <= 90 and -180 <= p[1] <= 180)]
            if invalid_coords:
                st.error(f"Invalid coordinates detected: {invalid_coords}")
            elif len(st.session_state.path) != len(st.session_state.trigger_points):
                st.error("Mismatch between path and trigger points lengths.")
            else:
                lats = [p[0] for p in st.session_state.path]
                lons = [p[1] for p in st.session_state.path]
                center_lat = (min(lats) + max(lats)) / 2
                center_lon = (min(lons) + max(lons)) / 2
                
                map_types = {
                    "OpenStreetMap": ("OpenStreetMap", None),
                    "Satellite": ("https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", "Google Satellite"),
                    "Dark": ("CartoDB dark_matter", None),
                    "Light": ("CartoDB positron", None),
                    "Satellite Streets": ("https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", "Google Satellite Streets")
                }
                selected_map = st.selectbox("Select Map Type", list(map_types.keys()))
                
                tiles, attr = map_types[selected_map]
                if attr:
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=13,
                        tiles=tiles,
                        attr=attr,
                        max_zoom=19,
                        control_scale=True
                    )
                else:
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=13,
                        tiles=tiles,
                        max_zoom=19,
                        control_scale=True
                    )
                
                path_color = 'yellow' if selected_map in ["Satellite", "Dark", "Satellite Streets"] else 'blue'
                coords = [[p[0], p[1]] for p in st.session_state.path]
                folium.PolyLine(
                    locations=coords,
                    color=path_color,
                    weight=2.5,
                    opacity=1,
                    tooltip='Flight Path'
                ).add_to(m)
                
                for i, p in enumerate(st.session_state.path):
                    trigger = st.session_state.trigger_points[i]
                    is_takeoff_landing = i == 0 or i == len(st.session_state.path)-1
                    marker_color = 'red' if is_takeoff_landing else 'lightblue'
                    popup_content = f"WP {i+1}<br>Alt: {p[2]:.1f}m"
                    if trigger["trigger_type"] == "camera":
                        distance = trigger["trigger_params"].get("distance", 0)
                        popup_content += f"<br>Camera Trigger: {distance}m"
                    elif trigger["trigger_type"] == "loiter":
                        popup_content += "<br>Loiter"
                    
                    folium.Marker(
                        location=[p[0], p[1]],
                        popup=popup_content,
                        icon=folium.Icon(color=marker_color, icon='circle', prefix='fa'),
                        tooltip=f"WP {i+1}"
                    ).add_to(m)
                
                bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
                m.fit_bounds(bounds, padding=[50, 50])
                
                st_folium(m, width=700, height=500)
                
        except Exception as e:
            st.error(f"Error rendering map: {e}")
            st.write("Possible causes: Invalid coordinates, network issues, or Folium rendering error.")
            st.write(f"Sample path coords: {st.session_state.path[:3] if st.session_state.path else 'None'}")
    else:
        st.write("Generate a flight plan to view the satellite map.")
    
    st.header("Graph View (Flight Path & Terrain)")
    if st.session_state.path:
        coords = [(p[0], p[1]) for p in st.session_state.path]
        terrain_elevs = fetch_elevations(coords)
        altitudes = [p[2] for p in st.session_state.path]
        distances = [0]
        for i in range(1, len(st.session_state.path)):
            dist = geodesic(st.session_state.path[i-1][:2], st.session_state.path[i][:2]).meters
            distances.append(distances[-1] + dist)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=distances,
            y=terrain_elevs,
            mode='lines',
            name='Terrain',
            line=dict(color='brown', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=distances,
            y=altitudes,
            mode='lines+markers',
            name='Flight Path',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        fig.update_layout(
            title='Flight Path and Terrain Profile',
            xaxis_title='Distance (m)',
            yaxis_title='Elevation (m)',
            legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.5)'),
            margin=dict(l=40, r=40, t=40, b=40),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Generate a flight plan to view the flight path and terrain profile.")
    
    st.header("Download Flight Plan")
    if st.session_state.plan:
        plan_json = json.dumps(st.session_state.plan, indent=2)
        st.download_button(
            label="Download Flight Plan (.plan)",
            data=plan_json,
            file_name="flight_plan.plan",
            mime="application/json",
            key="download_button"
        )
    else:
        st.write("Generate a flight plan to download.")
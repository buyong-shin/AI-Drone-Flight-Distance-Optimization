#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-Drone — Flight Range & Airspace Overlay (MOBILE)
- Open-Meteo 실시간 풍속/풍향/기온 연동
- 소비율(Wh/km) 기반 도달 반경 계산
- 출발지/목적지 클릭 → 거리·SOC·가능/불가 판정
- 바람 부는 방향(to)을 기준으로 반경 원의 중심 이동
- AIP KMZ + GeoJSON 공역 레이어 표시
- 📱 모바일 화면 최적화 레이아웃
"""

import io
import re
import json
import zipfile
import copy
from pathlib import Path
from typing import List, Tuple, Optional

import requests
import numpy as np
import pandas as pd
import folium
from folium import FeatureGroup

from shapely.geometry import Point, shape
from shapely.affinity import scale
from lxml import etree

import streamlit as st
from streamlit_folium import st_folium

# ============================================================
# 0. 기본 설정 (모바일 UI용)
# ============================================================

st.set_page_config(
    page_title="AI-Drone (Mobile)",
    layout="centered",                 # ✅ 모바일: centered
    initial_sidebar_state="collapsed"  # ✅ 모바일: 처음엔 사이드바 접기
)

st.markdown(
    """
    <div style='text-align:center; margin-top:8px; margin-bottom:10px; line-height:1.4;'>
      <div style='font-size:22px; font-weight:800; color:#0F172A;'>
        AI-Drone Flight Distance (Mobile)
      </div>
      <div style='font-size:12px; color:#6b7280; margin-top:2px;'>
        출발지·목적지 클릭만으로 비행 가능 여부 확인
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 기준 좌표 (세종)
LAT, LON = 36.6108, 127.2869
CENTER_LAT, CENTER_LON = LAT, LON

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
FILE_DIR = Path(__file__).resolve().parent

DEFAULT_KMZ       = FILE_DIR / "AIP 25년 5차 기준.kmz"
DEFAULT_FORBIDDEN = FILE_DIR / "forbidden_outline.geojson"
DEFAULT_ALLOWED   = FILE_DIR / "allowed_area.geojson"

M_PER_DEG   = 111_000               # 위도 1도 ≈ 111 km (111,000 m)
DEG_PER_KM  = 1000.0 / M_PER_DEG    # 1 km를 위/경도(도)로 바꾸는 계수
AIR3_WH     = 62.0                  # DJI Air 3 기준 배터리 Wh (가정)
USABLE_SOC_FRAC = 0.8               # DJI Air3: RTH 20% 남기고 80%까지만 사용

# Open-Meteo 실패 시 기본값
DEFAULT_WIND_SPEED_MS = 10.0
DEFAULT_TEMP_C        = 20.0

# 실측/근사 Base 반경 표 (풍속 set / 페이로드)
BASE_MAX_KM = pd.DataFrame([
    {"wind_set": 5,  "payload_g": 0,   "max_km":  9.692},
    {"wind_set": 10, "payload_g": 0,   "max_km": 19.835},
    {"wind_set": 15, "payload_g": 0,   "max_km": 22.000},
    {"wind_set": 20, "payload_g": 0,   "max_km": 24.000},
    {"wind_set": 5,  "payload_g": 168, "max_km":  7.726},
    {"wind_set": 10, "payload_g": 168, "max_km": 15.139},
    {"wind_set": 15, "payload_g": 168, "max_km": 17.000},
    {"wind_set": 20, "payload_g": 168, "max_km": 18.000},
    {"wind_set": 5,  "payload_g": 336, "max_km":  6.500},
    {"wind_set": 10, "payload_g": 336, "max_km": 12.500},
    {"wind_set": 15, "payload_g": 336, "max_km": 14.000},
    {"wind_set": 20, "payload_g": 336, "max_km": 15.000},
    {"wind_set": 5,  "payload_g": 504, "max_km":  5.500},
    {"wind_set": 10, "payload_g": 504, "max_km": 10.500},
    {"wind_set": 15, "payload_g": 504, "max_km": 12.000},
    {"wind_set": 20, "payload_g": 504, "max_km": 13.000},
])

WIND_SETS    = [5, 10, 15, 20]
PAYLOAD_SETS = [0, 168, 336, 504]

# ------------------------------------------------------------
# 금지구역(보라색 영역) 폴리곤 로딩 + 클릭 차단용 함수
# ------------------------------------------------------------

def load_forbidden_polygons(path: Path):
    polys = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for feat in data.get("features", []):
            geom = feat.get("geometry")
            if geom:
                try:
                    polys.append(shape(geom))
                except Exception:
                    pass
    except Exception:
        pass
    return polys

if DEFAULT_FORBIDDEN.exists():
    FORBIDDEN_POLYGONS = load_forbidden_polygons(DEFAULT_FORBIDDEN)
else:
    FORBIDDEN_POLYGONS = []

def is_in_forbidden(lat: float, lon: float) -> bool:
    if not FORBIDDEN_POLYGONS:
        return False
    p = Point(lon, lat)
    for poly in FORBIDDEN_POLYGONS:
        try:
            if poly.contains(p):
                return True
        except Exception:
            continue
    return False

# ============================================================
# 1. 풍향/기상 유틸
# ============================================================

def deg_to_cardinal(deg: float) -> str:
    if deg is None:
        return "N"
    try:
        d = float(deg)
    except (TypeError, ValueError):
        return "N"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ix = int((d % 360) / 45.0 + 0.5) % 8
    return dirs[ix]

def arrow_from_cardinal(card: str) -> str:
    arrows = {
        "N": "↑",  "NE": "↗", "E": "→",  "SE": "↘",
        "S": "↓",  "SW": "↙", "W": "←",  "NW": "↖",
    }
    return arrows.get(card, "↑")

@st.cache_data(ttl=60, show_spinner=False)
def fetch_wind_temp_current(lat: float, lon: float, unit: str = "ms"):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": unit,
        "timezone": "Asia/Seoul",
    }
    r = requests.get(OPEN_METEO, params=params, timeout=10)
    r.raise_for_status()
    cur = r.json().get("current", {})
    return (
        cur.get("wind_speed_10m"),
        cur.get("wind_direction_10m"),
        cur.get("temperature_2m"),
        cur.get("time"),
    )

# ============================================================
# 2. 세션 상태 초기화
# ============================================================

ss = st.session_state
ss.setdefault("click_count", 0)
ss.setdefault("origin", None)
ss.setdefault("dest", None)

ss.setdefault("wind_speed_mps", None)
ss.setdefault("wind_deg_from", None)  # API에서 온 from-direction
ss.setdefault("wind_deg_to", None)    # 우리가 쓰는 to-direction
ss.setdefault("temp_c_now", None)
ss.setdefault("meteo_ts", None)

ss.setdefault("ui_cap_wh", AIR3_WH)
ss.setdefault("ui_cruise_ms", 40.0 / 3.6)  # 기본 40 km/h
ss.setdefault("ui_payload_g", 0)

ss.setdefault("airspace_visible", True)
ss.setdefault("show_coords", True)
ss.setdefault("show_range", True)
ss.setdefault("last_R", None)
ss.setdefault("circle_center", None)  # (lat, lon)

# ============================================================
# 3. 실시간 기상 데이터 호출
# ============================================================

origin_for_weather = ss.origin
if origin_for_weather:
    meteo_lat, meteo_lon = origin_for_weather[0], origin_for_weather[1]
else:
    meteo_lat, meteo_lon = LAT, LON

wind_spd_mps = None
wind_deg_from = None
temp_c_now   = None
meteo_ts     = None

try:
    wind_spd_mps, wind_deg_from, temp_c_now, meteo_ts = fetch_wind_temp_current(
        meteo_lat, meteo_lon, unit="ms"
    )
except Exception as e:
    st.error(f"실시간 날씨 데이터를 불러오는 중 오류가 발생했습니다: {e}")

# fallback + 세션 저장
if wind_spd_mps is None:
    wind_speed = float(DEFAULT_WIND_SPEED_MS)
else:
    wind_speed = float(wind_spd_mps)

if temp_c_now is None:
    temp_c = float(DEFAULT_TEMP_C)
else:
    temp_c = float(temp_c_now)

ss["wind_speed_mps"] = wind_speed
ss["temp_c_now"]     = temp_c
ss["meteo_ts"]       = meteo_ts

if wind_deg_from is not None:
    wind_deg_from = float(wind_deg_from)
    wind_deg_to = (wind_deg_from + 180.0) % 360.0
else:
    wind_deg_from = None
    wind_deg_to   = None

ss["wind_deg_from"] = wind_deg_from
ss["wind_deg_to"]   = wind_deg_to

# ============================================================
# 4. 상단 날씨 블록 (모바일용: 한 줄에 2~3개씩)
# ============================================================

if (wind_spd_mps is None) or (wind_deg_from is None) or (temp_c_now is None):
    st.info("실시간 바람/기온 데이터를 불러올 수 없습니다. (기본값: 10 m/s, 20°C 사용)")
else:
    card_to = deg_to_cardinal(float(wind_deg_to))
    arr_to  = arrow_from_cardinal(card_to)

    st.markdown(
        """
        <style>
        .wx-wrap-m {
            display:flex; justify-content:center; gap:20px;
            text-align:center; margin-top:4px; margin-bottom:6px;
            flex-wrap:wrap;
        }
        .wx-box-m  {
            display:flex; flex-direction:column; align-items:center;
            min-width:110px;
        }
        .wx-title-m {
            font-weight:600; font-size:11px; color:#374151;
            margin-bottom:2px; line-height:1.2;
        }
        .wx-val-m {
            font-weight:700; font-size:15px; color:#111827;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="wx-wrap-m">
            <div class="wx-box-m">
                <div class="wx-title-m">풍속</div>
                <div class="wx-val-m">{float(wind_spd_mps):.2f} m/s</div>
            </div>
            <div class="wx-box-m">
                <div class="wx-title-m">풍향(도)</div>
                <div class="wx-val-m">{float(wind_deg_to):.0f}°</div>
            </div>
            <div class="wx-box-m">
                <div class="wx-title-m">바람 방향</div>
                <div class="wx-val-m">{card_to} {arr_to}</div>
            </div>
            <div class="wx-box-m">
                <div class="wx-title-m">기온</div>
                <div class="wx-val-m">{float(temp_c_now):.1f} °C</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# 5. 비행 모델 유틸
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    dφ = np.radians(lat2 - lat1)
    dλ = np.radians(lon2 - lon1)
    a = np.sin(dφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2
    return float(2 * R * np.arcsin(np.sqrt(a)))

def add_circle_km(m, lat, lon, km, color_hex="#22c55e", weight=4, alpha=0.25):
    shp = Point(lon, lat).buffer(1, resolution=256)
    a_y = km * 1000.0 / M_PER_DEG
    a_x = a_y / max(np.cos(np.deg2rad(lat)), 1e-9)
    shp = scale(shp, xfact=a_x, yfact=a_y)
    folium.GeoJson(
        shp.__geo_interface__,
        style_function=lambda _: {
            "fillColor": color_hex,
            "color": color_hex,
            "weight": weight,
            "fillOpacity": alpha,
        },
        tooltip=f"{km:.2f} km",
        name="Range",
    ).add_to(m)

def predict_whkm(speed_kmh, wspd, temp_c_in, payload_g,
                 wind_from_deg=None, heading_deg=None):

    v_opt = 40.0

    wind_term = np.clip((wspd - 5.0) / 10.0, -2.0, 3.0)
    v_opt -= 2.0 * wind_term

    payload_term = np.clip(payload_g / 504.0, 0.0, 1.0)
    v_opt -= 3.0 * payload_term

    temp_term = np.clip((20.0 - temp_c_in) / 20.0, -2.0, 2.0)
    v_opt -= 1.5 * temp_term

    v_opt = float(np.clip(v_opt, 26.0, 46.0))

    base_wh = 2.4
    delta = (speed_kmh - v_opt) / 10.0
    speed_factor = 1.0 + 0.09 * (delta ** 2)

    wind_factor = 1.0 + 0.03 * np.clip(wspd, 0.0, 20.0)
    temp_factor = 1.0 + 0.25 * np.clip((20.0 - temp_c_in) / 40.0, -0.8, 1.0)
    weight_factor = 1.0 + 0.4 * np.clip(payload_g / 504.0, 0.0, 1.0)

    eff = base_wh * speed_factor * wind_factor * temp_factor * weight_factor
    return float(np.clip(eff, 1.5, 10.0))

def best_speed_and_whkm(wspd, temp_c_in, payload_g):
    speeds_kmh = np.arange(24.0, 57.0, 1.0)
    effs = np.array([
        predict_whkm(v, wspd, temp_c_in, payload_g)
        for v in speeds_kmh
    ])

    if len(effs) == 0 or np.all(np.isnan(effs)):
        return 40.0, 3.0

    i = int(np.nanargmin(effs))
    best_v = float(speeds_kmh[i])
    best_v = float(np.clip(best_v, 26.0, 46.0))
    return best_v, float(effs[i])

def lookup_base_range_from_table(wind_ms: float, payload_g: float) -> Optional[float]:
    if BASE_MAX_KM.empty:
        return None

    df = BASE_MAX_KM

    w = float(np.clip(wind_ms,  min(WIND_SETS),    max(WIND_SETS)))
    p = float(np.clip(payload_g, min(PAYLOAD_SETS), max(PAYLOAD_SETS)))

    w_list = sorted(WIND_SETS)
    w1, w2 = w_list[0], w_list[-1]
    for i in range(len(w_list) - 1):
        if w_list[i] <= w <= w_list[i + 1]:
            w1, w2 = w_list[i], w_list[i + 1]
            break

    p_list = sorted(PAYLOAD_SETS)
    p1, p2 = p_list[0], p_list[-1]
    for i in range(len(p_list) - 1):
        if p_list[i] <= p <= p_list[i + 1]:
            p1, p2 = p_list[i], p_list[i + 1]
            break

    def _get_km(wind_set, payload_set):
        row = df[(df["wind_set"] == wind_set) & (df["payload_g"] == payload_set)]
        if row.empty:
            return None
        return float(row.iloc[0]["max_km"])

    z11 = _get_km(w1, p1)
    z12 = _get_km(w1, p2)
    z21 = _get_km(w2, p1)
    z22 = _get_km(w2, p2)

    if any(v is None for v in [z11, z12, z21, z22]):
        return None

    if abs(w2 - w1) < 1e-9:
        t_w = 0.0
    else:
        t_w = (w - w1) / (w2 - w1)

    if abs(p2 - p1) < 1e-9:
        t_p = 0.0
    else:
        t_p = (p - p1) / (p2 - p1)

    z_w1 = z11 + (z12 - z11) * t_p
    z_w2 = z21 + (z22 - z21) * t_p

    z_final = z_w1 + (z_w2 - z_w1) * t_w
    return float(z_final)

def compute_shifted_center(origin_lat, origin_lon,
                           wind_deg_to, R_display_km,
                           wind_speed_ms, cruise_ms, trip_type):

    circle_center_lat, circle_center_lon = origin_lat, origin_lon

    if (wind_deg_to is None) or (R_display_km is None) or (R_display_km <= 0):
        return circle_center_lat, circle_center_lon
    if cruise_ms <= 0 or wind_speed_ms <= 0:
        return circle_center_lat, circle_center_lon

    ratio = float(wind_speed_ms) / float(cruise_ms)

    if trip_type == "편도":
        shift_km = R_display_km * ratio
    else:
        shift_km = 2.0 * R_display_km * ratio

    if shift_km <= 0:
        return circle_center_lat, circle_center_lon

    theta = np.deg2rad(wind_deg_to)
    dy_km = shift_km * np.cos(theta)
    dx_km = shift_km * np.sin(theta)

    km_to_deg = 1000.0 / M_PER_DEG
    dlat = dy_km * km_to_deg
    dlon = dx_km * km_to_deg / max(np.cos(np.deg2rad(origin_lat)), 1e-6)

    circle_center_lat = origin_lat + dlat
    circle_center_lon = origin_lon + dlon
    return circle_center_lat, circle_center_lon

# ============================================================
# 6. 공역 파서 (GeoJSON / KMZ)
# ============================================================

NS = {
    "kml": "http://www.opengis.net/kml/2.2",
    "gx": "http://www.google.com/kml/ext/2.2",
}

def _parse_coords_text(txt: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    if not txt:
        return pts
    for token in txt.replace("\n", " ").replace("\t", " ").split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                pts.append((lat, lon))
            except ValueError:
                pass
    return pts

def _classify(name: str) -> str:
    if not name:
        return "other"
    n = name.lower()

    if re.search(r"\bp[-\s]?\d+\b", n) or "prohibited" in n or "금지" in n:
        return "prohibited"
    if re.search(r"\br[-\s]?\d+\b", n) or "restricted" in n or "제한" in n:
        return "restricted"
    if re.search(r"\bd[-\s]?\d+\b", n) or "danger" in n or "위험" in n:
        return "danger"
    if "ctr" in n or "관제" in n or "atz" in n:
        return "ctr"
    if "tma" in n:
        return "tma"
    if "cta" in n:
        return "cta"
    if "adiz" in n or "방공식별" in n:
        return "adiz"
    if "fir" in n:
        return "fir"
    if "moa" in n or "훈련" in n or "mtr" in n or "군사" in n or "oparea" in n:
        return "training"
    if (
        "airway" in n
        or "awy" in n
        or "ats route" in n
        or "항로" in n
        or "비행로" in n
        or re.search(r"\b[vnagl]\s*-\s*\d+", n)
    ):
        return "airway"
    if "boundary" in n or "경계" in n or "구분" in n:
        return "boundary"
    return "other"

def add_airspace_geojson(m, geojson_bytes: bytes, selected_cats, layer_name="Airspace (GeoJSON)"):
    try:
        gj = json.loads(geojson_bytes.decode("utf-8"))
        feats = gj.get("features", [])
        gj["features"] = [
            f for f in feats
            if _classify((f.get("properties") or {}).get("name") or "airspace") in selected_cats
        ]

        def _style(f):
            name = (f.get("properties") or {}).get("name") or "airspace"
            cat = _classify(name)

            COLOR = {
                "prohibited": "#ff0033", "restricted": "#ff8800", "danger": "#ff3d00",
                "ctr": "#0066ff", "tma": "#3377ff", "cta": "#8a2be2",
                "training": "#00b050", "adiz": "#aa00aa", "fir": "#5555aa",
                "airway": "#00ccff", "boundary": "#999999",
            }
            FILL_OPACITY = {
                "adiz": 0.0, "fir": 0.0, "cta": 0.02, "tma": 0.03,
                "ctr": 0.05, "prohibited": 0.10, "restricted": 0.08,
                "danger": 0.08, "training": 0.06,
                "airway": 0.0, "boundary": 0.0,
            }
            DASH = {"airway": "6,6", "boundary": "4,6", "adiz": "8,6", "fir": "8,6"}

            return {
                "fillColor": COLOR.get(cat, "#999"),
                "color": COLOR.get(cat, "#999"),
                "weight": 1.2,
                "fillOpacity": FILL_OPACITY.get(cat, 0.05),
                "dashArray": DASH.get(cat),
            }

        folium.GeoJson(gj, name=layer_name, style_function=_style).add_to(m)
        return True

    except Exception as e:
        st.warning(f"GeoJSON 로드 실패: {e}")
        return False

def add_airspace_kmz_from_bytes(m, raw_bytes: bytes, selected_cats,
                                layer_name_prefix="Airspace (AIP KMZ)"):

    try:
        tmp_dir = FILE_DIR / "_tmp_kmz_unpack"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for p in tmp_dir.rglob("*"):
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass

        with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as zf:
            zf.extractall(tmp_dir)

        kml_files = list(tmp_dir.rglob("*.kml"))
        if not kml_files:
            st.warning("KMZ 내부에 KML이 없습니다.")
            return False

        parser = etree.XMLParser(recover=True, huge_tree=True)
        added = 0

        for kml_path in kml_files:
            doc = etree.fromstring(kml_path.read_bytes(), parser=parser)

            for poly in doc.xpath(".//kml:Polygon", namespaces=NS):
                name_node = poly.xpath(
                    "ancestor-or-self::kml:Placemark/kml:name/text()",
                    namespaces=NS,
                )
                pname = name_node[0] if name_node else ""
                cat = _classify(pname)
                if cat not in selected_cats:
                    continue

                rings: List[List[Tuple[float, float]]] = []

                outer = poly.xpath(
                    ".//kml:outerBoundaryIs//kml:LinearRing/kml:coordinates/text()",
                    namespaces=NS,
                )
                if outer:
                    rings.append(_parse_coords_text(" ".join(outer)))

                inners = poly.xpath(
                    ".//kml:innerBoundaryIs//kml:LinearRing/kml:coordinates/text()",
                    namespaces=NS,
                )
                for txt in inners:
                    pts = _parse_coords_text(txt)
                    if pts:
                        rings.append(pts)

                if rings and rings[0]:
                    folium.Polygon(
                        locations=rings[0],
                        holes=rings[1:] if len(rings) > 1 else None,
                        weight=1.2,
                        color="#8b5cf6",
                        opacity=0.7,
                        fill=True,
                        fill_color="#8b5cf6",
                        fill_opacity=0.03,
                        name=f"{layer_name_prefix} - Poly",
                    ).add_to(m)
                    added += 1

            for ln in doc.xpath(".//kml:LineString", namespaces=NS):
                name_node = ln.xpath(
                    "ancestor-or-self::kml:Placemark/kml:name/text()",
                    namespaces=NS,
                )
                pname = name_node[0] if name_node else ""
                cat = _classify(pname)
                if cat not in selected_cats:
                    continue

                coords = ln.xpath(".//kml:coordinates/text()", namespaces=NS)
                pts = _parse_coords_text(" ".join(coords)) if coords else []
                if pts:
                    folium.PolyLine(
                        locations=pts,
                        weight=2,
                        color="#8b5cf6",
                        opacity=0.7,
                        name=f"{layer_name_prefix} - Line",
                    ).add_to(m)
                    added += 1

        return added > 0

    except Exception as e:
        st.warning(f"KMZ 파싱 실패: {e}")
        return False

# ============================================================
# 7. 사이드바 UI (모바일용: 더 압축된 버전)
# ============================================================

with st.sidebar:

    default_cap = float(ss.get("ui_cap_wh", AIR3_WH))
    default_cap = min(max(default_cap, 20.0), 300.0)

    st.markdown(
        "<div style='text-align:center; font-size:16px; font-weight:600; margin-bottom:4px;'>DJI Air3 기준</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='text-align:center; font-size:11px; margin-top:2px;'>배터리 용량 (Wh)</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style='text-align:center; color:#6b7280; font-size:11px;
                    margin-top:8px; margin-bottom:6px;'>
            임무는 배터리 <b>80%</b>까지만 사용합니다.
        </div>
        """,
        unsafe_allow_html=True
    )

    ss['ui_cap_wh'] = st.number_input(
        label=" ",               
        min_value=20.0,
        max_value=300.0,
        value=default_cap,
        step=1.0,
    )

    if (wind_spd_mps is None) or (wind_deg_from is None) or (temp_c_now is None):
        st.info("실시간 데이터를 불러올 수 없습니다. (기본값 10 m/s, 20°C 사용)")
    else:
        card_sb = deg_to_cardinal(float(wind_deg_to))
        arr_sb  = arrow_from_cardinal(card_sb)

        st.markdown(
            """
<style>
.sb-wrap-m { display:flex; flex-direction:column; gap:4px; font-size:11px; }
.sb-row-m  { display:flex; justify-content:space-between; }
.sb-key-m  { color:#374151; font-weight:600; }
.sb-val-m  { color:#111827; font-weight:700; }
</style>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
<div class="sb-wrap-m">
  <div class="sb-row-m"><span class="sb-key-m">좌표 기준</span>
      <span class="sb-val-m">{meteo_lat:.4f}, {meteo_lon:.4f}</span></div>
  <div class="sb-row-m"><span class="sb-key-m">풍속</span>
      <span class="sb-val-m">{float(wind_spd_mps):.2f} m/s</span></div>
  <div class="sb-row-m"><span class="sb-key-m">풍향(도)</span>
      <span class="sb-val-m">{float(wind_deg_to):.0f}°</span></div>
  <div class="sb-row-m"><span class="sb-key-m">바람 방향</span>
      <span class="sb-val-m">{card_sb} {arr_sb}</span></div>
  <div class="sb-row-m"><span class="sb-key-m">기온</span>
      <span class="sb-val-m">{float(temp_c_now):.1f} °C</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        if meteo_ts:
            meteo_date = meteo_ts.split("T")[0]

            st.markdown(
                f"""
<div style='text-align:center; color:#6b7280; font-size:10px; margin-top:4px;'>
    Open-Meteo 기준<br>
    <b>{meteo_date}</b>
</div>
""",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # 좌표 표시
    st.subheader("클릭 좌표", anchor=False)
    if ss.get("show_coords", True):
        st.write("출발지:",
                 f"{ss.origin[0]:.6f}, {ss.origin[1]:.6f}" if ss.origin else "(미지정)")
        st.write("목적지:",
                 f"{ss.dest[0]:.6f}, {ss.dest[1]:.6f}" if ss.dest else "(미지정)")
        if st.button("좌표 초기화"):
            ss.click_count = 0
            ss.origin = None
            ss.dest = None
            ss.circle_center = None
            st.rerun()

    st.markdown("---")

    # 비행 설정
    st.subheader("비행 설정", anchor=False)

    speed_options_kmh = [20, 30, 40, 50]
    prev_ms  = float(ss.get("ui_cruise_ms", 10.0))
    prev_kmh = round(prev_ms * 3.6 / 10) * 10
    if prev_kmh not in speed_options_kmh:
        prev_kmh = 30

    selected_kmh = st.select_slider(
        "비행 속도 (km/h)",
        options=speed_options_kmh,
        value=prev_kmh,
    )

    st.markdown(
        """
        <div style="text-align:center; color:#6b7280; font-size:11px; margin-top:-6px; margin-bottom:4px;">
            40 km/h는 기본 순항 속도입니다.
        </div>
        """,
        unsafe_allow_html=True
    )

    ss["ui_cruise_ms"] = selected_kmh / 3.6

    ss["ui_payload_g"] = st.select_slider(
        "적재 중량 (g)",
        [0, 168, 336, 504],
        value=ss["ui_payload_g"]
        if ss["ui_payload_g"] in [0, 168, 336, 504]
        else 0,
    )
    trip_type = st.radio("비행 유형", ["편도", "왕복"], horizontal=True, index=0)

    st.markdown("---")

    # 공역 레이어 설정 (모바일: 토글만)
    st.subheader("공역 레이어", anchor=False)
    ss["airspace_visible"] = st.checkbox("공역 표시", value=ss.get("airspace_visible", True))

    use_forbidden = DEFAULT_FORBIDDEN.exists()
    use_allowed   = DEFAULT_ALLOWED.exists()
    use_kmz       = DEFAULT_KMZ.exists()
    st.caption(
        f"금지/제한 GeoJSON: {'있음' if use_forbidden else '없음'} · "
        f"허용 GeoJSON: {'있음' if use_allowed else '없음'} · "
        f"AIP KMZ: {'있음' if use_kmz else '없음'}"
    )

    AIRSPACE_COLOR = {
        "prohibited": "#ff0033", "restricted": "#ff8800", "danger": "#ff3d00",
        "ctr": "#0066ff", "tma": "#3377ff", "cta": "#8a2be2",
        "training": "#00b050", "adiz": "#aa00aa", "fir": "#5555aa",
        "airway": "#00ccff", "boundary": "#999999",
    }
    CAT_ORDER = [
        "prohibited", "restricted", "danger",
        "ctr", "tma", "cta",
        "training", "adiz", "fir",
        "airway", "boundary",
    ]
    DEFAULT_ON = set([
        "prohibited", "restricted", "danger",
        "ctr", "tma", "cta", "training",
    ])

    if "airspace_on" not in ss:
        ss.airspace_on = {k: (k in DEFAULT_ON) for k in CAT_ORDER}

    if ss.get("airspace_visible", True):
        for key in CAT_ORDER:
            ss.airspace_on[key] = st.checkbox(
                f"{key}",
                value=ss.airspace_on[key],
                key=f"tog_{key}_m",
            )

selected_cats = [k for k, v in ss.get("airspace_on", {}).items() if v]

# ============================================================
# 8. 도달 반경 / SOC / 판정 계산 (원 중심 이동 포함)
# ============================================================

cap_Wh = float(ss["ui_cap_wh"])
cap_usable_Wh = cap_Wh * USABLE_SOC_FRAC

wind_speed = float(ss.get("wind_speed_mps", DEFAULT_WIND_SPEED_MS))
temp_c     = float(ss.get("temp_c_now",  DEFAULT_TEMP_C))

cruise_ms  = float(ss["ui_cruise_ms"])
cruise_kmh = cruise_ms * 3.6
payload_g  = int(ss["ui_payload_g"])

origin = ss.origin
dest   = ss.dest

best_now, _ = best_speed_and_whkm(wind_speed, temp_c, payload_g)
best_now = float(np.clip(best_now, 26.0, 46.0))

wh_now = predict_whkm(cruise_kmh, wind_speed, temp_c, payload_g)

base_km = lookup_base_range_from_table(wind_speed, payload_g)

if base_km is not None:
    wh_ref = predict_whkm(36.0, wind_speed, 20.0, payload_g)
    ratio = wh_ref / max(wh_now, 1e-6)
    R_full = base_km * float(np.clip(ratio, 0.4, 1.6))
    R_estimate_base = R_full * USABLE_SOC_FRAC
else:
    R_estimate_base = (cap_usable_Wh / wh_now) if (wh_now and cap_usable_Wh > 0) else None

R_max_base = R_estimate_base
wh_leg     = wh_now

R_max_display = R_max_base
if R_max_display is not None and trip_type == "왕복":
    R_max_display /= 2.0

if (R_max_display is not None) and (R_max_display > 0):
    ss.last_R = float(R_max_display)

circle_center_lat = None
circle_center_lon = None
if origin and R_max_display is not None and R_max_display > 0:
    circle_center_lat, circle_center_lon = compute_shifted_center(
        origin[0], origin[1],
        ss.get("wind_deg_to"),
        R_max_display,
        wind_speed,
        cruise_ms,
        trip_type,
    )
    ss.circle_center = (circle_center_lat, circle_center_lon)
else:
    ss.circle_center = None

soc_display = None
verdict_display = None

full_reason_html = ""
reason_html = ""

if origin and dest:
    dist_geom_km = haversine_km(origin[0], origin[1], dest[0], dest[1])

    dist_for_range  = dist_geom_km
    dist_for_energy = dist_geom_km

    if (R_max_display is not None) and (R_max_display > 0.0) \
       and (wind_speed is not None) and (cruise_ms > 0.0):

        wind_deg_to_loc = ss.get("wind_deg_to", None)
        if wind_deg_to_loc is None:
            wind_deg_to_loc = ss.get("wind_deg_from", 0.0)

        try:
            theta = np.deg2rad(float(wind_deg_to_loc))
        except Exception:
            theta = 0.0

        ratio = float(wind_speed) / max(float(cruise_ms), 1e-6)

        if trip_type == "왕복":
            shift_km = 2.0 * R_max_display * ratio
        else:
            shift_km = 1.0 * R_max_display * ratio

        lat0, lon0 = origin
        dy_km = shift_km * np.cos(theta)
        dx_km = shift_km * np.sin(theta)

        lat_shift = lat0 + dy_km * DEG_PER_KM
        lon_shift = lon0 + dx_km * DEG_PER_KM / max(np.cos(np.deg2rad(lat0)), 1e-9)

        dist_eff_km = haversine_km(lat_shift, lon_shift, dest[0], dest[1])

        dist_for_range  = dist_eff_km
        dist_for_energy = dist_eff_km

    if wh_leg:
        trip_factor = 2.0 if trip_type == "왕복" else 1.0
        E_need = wh_leg * dist_for_energy * trip_factor
    else:
        E_need = None

    if E_need and cap_Wh > 0:
        soc_full = 100.0 * E_need / cap_Wh
    else:
        soc_full = None

    soc_display = soc_full

    ok = (
        R_max_display is not None
        and dist_for_range <= R_max_display
        and soc_full is not None
        and soc_full <= 80.0
    )

    verdict_display = "가능" if ok else "불가"

    reason_lines = []

    if R_max_display is not None:
        if dist_for_range > R_max_display:
            reason_lines.append(
                f"<span style='color:#ef4444; font-weight:600;'>거리 초과</span> "
                f"(가능 {R_max_display:.2f} km / 유효 {dist_for_range:.2f} km / 직선 {dist_geom_km:.2f} km)"
            )
        else:
            reason_lines.append(
                f"<span style='color:#16a34a; font-weight:600;'>거리 여유</span> "
                f"(가능 {R_max_display:.2f} km / 유효 {dist_for_range:.2f} km / 직선 {dist_geom_km:.2f} km)"
            )

    if soc_display is not None:
        if soc_display > 80.0:
            reason_lines.append(
                f"<span style='color:#ef4444; font-weight:600;'>배터리 부족</span> "
                f"(필요 {soc_display:.1f}% / 80% 한계 초과)"
            )
        else:
            reason_lines.append(
                f"<span style='color:#16a34a; font-weight:600;'>배터리 여유</span> "
                f"(필요 {soc_display:.1f}% / 80% 이내)"
            )

    reason_html = "<br>".join(reason_lines)

    if verdict_display == "불가":
        solution_html = """
<div style='margin-top:4px; font-size:11px; color:#4b5563;'>
  <b>해결 방법</b><br>
  • <b>속도</b>를 조정해 에너지 효율을 높여보세요.<br>
  • <b>목적지 위치</b>를 더 가깝게 조정해 보세요.<br>
  • <b>적재 중량(g)</b>을 줄여 페이로드를 가볍게 해 보세요.<br>
  • 비행 유형을 <b>왕복 → 편도</b>로 바꾸면 필요한 배터리가 줄어듭니다.
</div>
"""
    else:
        solution_html = ""

    if verdict_display == "불가":
        full_reason_html = (
            "<b>비행 불가 사유</b><br>"
            f"{reason_html}"
            f"{solution_html}"
        )
    else:
        full_reason_html = ""

# ============================================================
# 9. 상단 메트릭 바 (모바일용)
# ============================================================

speed_txt  = f"{best_now:.1f} km/h"
whkm_txt   = f"{(wh_leg if wh_leg is not None else float('nan')):.2f} Wh/km"
rmax_txt   = "-" if R_max_display is None else f"{R_max_display:.2f} km"

if soc_display is None:
    batt_txt = "-"
else:
    batt_txt = f"{soc_display:.1f}%"

verdict_txt   = "-" if verdict_display is None else verdict_display
verdict_color = "#16a34a" if verdict_display == "가능" else ("#ef4444" if verdict_display == "불가" else "#6b7280")

metrics_bar_html = f"""
<style>
.metrics-wrap-m {{
  margin: 6px auto 8px;
  max-width: 520px;
  display: flex; gap: 10px; flex-wrap: wrap;
  justify-content: space-evenly;
  background: rgba(255,255,255,.98);
  border: 1px solid #e5e7eb; border-radius: 12px;
  padding: 8px 10px; box-shadow: 0 2px 8px rgba(0,0,0,.05);
}}
.mbox-m {{ min-width: 45%; text-align: center; margin-bottom:2px; }}
.mtitle-m {{ font-size: 11px; color:#64748b; font-weight: 700; margin-bottom:1px; }}
.mvalue-m {{ font-size: 16px; font-weight: 800; color:#0f172a; }}
</style>
<div class="metrics-wrap-m">
  <div class="mbox-m">
    <div class="mtitle-m">추천 속도</div>
    <div class="mvalue-m">{speed_txt}</div>
  </div>
  <div class="mbox-m">
    <div class="mtitle-m">Wh/km</div>
    <div class="mvalue-m">{whkm_txt}</div>
  </div>
  <div class="mbox-m">
    <div class="mtitle-m">비행 거리</div>
    <div class="mvalue-m">{rmax_txt}</div>
  </div>
  <div class="mbox-m">
    <div class="mtitle-m">필요 배터리</div>
    <div class="mvalue-m">{batt_txt}</div>
  </div>
  <div class="mbox-m">
    <div class="mtitle-m">가능 여부</div>
    <div class="mvalue-m" style="color:{verdict_color}">{verdict_txt}</div>
  </div>
</div>
"""
st.markdown(metrics_bar_html, unsafe_allow_html=True)

if full_reason_html:
    reason_box_html = f"""
<div style="
    margin: 4px auto 6px;
    max-width: 520px;
    padding: 8px 10px;
    border-radius: 10px;
    background: rgba(255,255,255,0.95);
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    font-size: 11px;
    color: #1f2937;
">
  {full_reason_html}
</div>
"""
    st.markdown(reason_box_html, unsafe_allow_html=True)

# ============================================================
# 10. 지도 생성 및 표시 요소 (모바일: width=100%, height↓)
# ============================================================

base_map = folium.Map(
    location=[CENTER_LAT, CENTER_LON],
    zoom_start=10,
    control_scale=True,
)

if ss.get("airspace_visible", True):
    if DEFAULT_FORBIDDEN.exists():
        add_airspace_geojson(
            base_map,
            DEFAULT_FORBIDDEN.read_bytes(),
            selected_cats,
            "Airspace (Forbidden GeoJSON)",
        )
    if DEFAULT_ALLOWED.exists():
        add_airspace_geojson(
            base_map,
            DEFAULT_ALLOWED.read_bytes(),
            selected_cats,
            "Airspace (Allowed GeoJSON)",
        )
    if DEFAULT_KMZ.exists():
        add_airspace_kmz_from_bytes(
            base_map,
            DEFAULT_KMZ.read_bytes(),
            selected_cats,
            "Airspace (AIP KMZ)",
        )

fg = FeatureGroup(name="Dynamic", show=True)

if ss.origin:
    folium.Marker(ss.origin, tooltip="출발지").add_to(fg)
    R_draw = (
        R_max_display
        if (R_max_display is not None and R_max_display > 0)
        else ss.get("last_R", None)
    )
    if R_draw is not None and R_draw > 0:
        if ss.circle_center is not None:
            circle_center_lat, circle_center_lon = ss.circle_center
        else:
            circle_center_lat, circle_center_lon = ss.origin
        add_circle_km(
            base_map,
            circle_center_lat,
            circle_center_lon,
            R_draw,
            "#22c55e",
            4,
            0.25,
        )

if ss.origin and ss.dest:
    dist_km = haversine_km(
        ss.origin[0], ss.origin[1], ss.dest[0], ss.dest[1]
    )
    ok = (verdict_display == "가능")
    line_col = "#16a34a" if ok else "#ef4444"
    dot_col  = "green"   if ok else "red"

    folium.PolyLine(
        [ss.origin, ss.dest],
        color=line_col,
        weight=4,
        opacity=0.95,
    ).add_to(fg)
    folium.Marker(
        ss.dest,
        tooltip=f"목적지 — {dist_km:.2f} km · {'가능' if ok else '불가'}",
        icon=folium.Icon(color=dot_col),
    ).add_to(fg)

ui_sig = (
    f"{ss['ui_cruise_ms']}-{payload_g}-{cap_Wh}-"
    f"{wind_speed:.2f}-{temp_c}-"
    f"{trip_type}-"
    f"{1 if ss.origin else 0}-{1 if ss.dest else 0}"
)

ret = st_folium(
    copy.deepcopy(base_map),
    feature_group_to_add=fg,
    width="100%",      # ✅ 모바일: 화면 폭에 맞추기
    height=520,        # ✅ 약간 줄인 높이
    returned_objects=["last_clicked"],
    key=ui_sig,
)

clicked = ret.get("last_clicked")
if clicked:
    lat = float(clicked["lat"])
    lon = float(clicked["lng"])

    if is_in_forbidden(lat, lon):
        alert_html = """
        <div style="
            margin-top:6px;
            padding:10px 12px;
            border-radius:10px;
            background:#f3e8ff;
            color:#1f2937;
            font-size:11px;
            line-height:1.5;
        ">
            <b><span style="color:#dc2626;">비행 금지 공역</span></b> (보라색 영역)입니다.
            <span style="color:#1d4ed8; font-weight:600;">다른 위치를 선택해주세요.</span>
        </div>
        """
        st.markdown(alert_html, unsafe_allow_html=True)

    else:
        ss.click_count += 1
        if ss.click_count % 2 == 1:
            ss.origin = (lat, lon)
            ss.dest = None
        else:
            ss.dest = (lat, lon)

        ss.circle_center = None
        st.rerun()

# ============================================================
# 11. 하단 설명 블록 (모바일용 간단 Legend)
# ============================================================

legend_html_mobile = """
<style>
.legend-box-m{
  margin:8px auto 12px;
  padding:10px 12px 8px;
  border:1px solid #e5e7eb;
  border-radius:12px;
  background:#ffffff;
  color:#111827;
  font-size:11px;
  max-width:520px;
  line-height:1.5;
}
.legend-title-m{
  font-size:12px;
  font-weight:800;
  margin-bottom:3px;
}
.legend-section-m{margin-top:10px;}
.legend-box-m ul{
  margin:3px 0 0 0;
  padding-left:1rem;
}
.legend-box-m li{margin:2px 0;}
.legend-footer-m{
  text-align:center;
  margin-top:10px;
  font-size:11px;
  color:#4b5563;
  font-weight:600;
}
</style>

<div class="legend-box-m">

  <!-- 지도 사용 -->
  <div class="legend-section-m">
    <div class="legend-title-m">지도 사용 방법</div>
    <ul>
      <li>지도를 터치해 <b>출발지 → 목적지</b> 순서로 선택합니다.</li>
      <li>경로선이 <span style="color:#16a34a;font-weight:700;">초록색</span>이면 <b>비행 가능</b>,
          <span style="color:#ef4444;font-weight:700;">빨간색</span>이면 <b>비행 불가</b>입니다.</li>
    </ul>
  </div>

  <!-- 공역 안내 -->
  <div class="legend-section-m">
    <div class="legend-title-m">공역 안내</div>
    <ul>
      <li><span style="color:#8b5cf6;font-weight:700;">보라색 윤곽선</span>은 AIP 공역(P/R/D·CTR·TMA 등)을
          단순 통합한 <b>주의 구역</b>입니다.</li>
      <li>실제 비행 전에는 반드시 <b>드론원스톱·AIP 원문</b>에서 고도 제한과 상세 공역을 확인하세요.</li>
    </ul>
  </div>

  <!-- 핵심 지표 -->
  <div class="legend-section-m">
    <div class="legend-title-m">핵심 지표</div>
    <ul>
      <li><b>추천 속도</b> : 현재 조건에서 가장 효율적인 비행 속도</li>
      <li><b>Wh/km</b> : 1km 이동 시 예상 배터리 사용량</li>
      <li><b>비행 가능 거리</b> : 현 조건에서 이동 가능한 최대 거리</li>
      <li><b>필요 배터리</b> : RTH 확보 위해 <b>20%</b>는 여유로 남기고 계산</li>
    </ul>
  </div>

  <!-- 비행 알고리즘 -->
  <div class="legend-section-m">
    <div class="legend-title-m">비행 알고리즘 이해하기</div>

    <div style="margin-top:5px; font-size:11px; line-height:1.55;">

      <div style="margin-bottom:4px;">
        <b>① 너무 느리거나 빠르면 효율이 떨어집니다.</b><br>
        전기모터 특성상 <b>35~45km/h</b> 구간에서 배터리 소모가 가장 낮습니다.
      </div>

      <div style="margin-bottom:4px;">
        <b>② 역풍에서는 배터리 소모가 크게 늘어납니다.</b><br>
        지상 속도가 줄어 <b>비행 시간↑ → 비행 가능 거리↓</b>가 됩니다.
      </div>

      <div style="margin-bottom:4px;">
        <b>③ 순풍일 때는 더 멀리 비행할 수 있습니다.</b><br>
        지상 속도가 증가해 <b>적은 에너지로 더 먼 거리</b>를 이동합니다.
      </div>

      <div style="margin-bottom:4px;">
        <b>④ 비행 가능 거리 = AI 모델 × 배터리 80% 사용 전략</b><br>
        DJI Air 3 기준으로 <b>20%는 RTH 여유</b>로 남기고 계산합니다.
      </div>

      <div style="margin-bottom:0;">
        <b>⑤ 무게·온도는 효율에 직접 영향을 줍니다.</b><br>
        적재 중량 증가·저온일수록 <b>Wh/km↑ → 비행 거리↓</b>가 됩니다.
      </div>

    </div>
  </div>

  <!-- 모델 기반 안내 -->
  <div class="legend-section-m">
    <div class="legend-title-m">모델 기반 안내</div>

    <div style="margin-top:5px; font-size:11px; line-height:1.55;">

      <div style="margin-bottom:4px;">
        <b>① 모델 구성 기반</b><br>
        <b>DJI Air 3 실제 비행 로그</b>와 <b>실시간 날씨 데이터</b>를 결합해,
        속도·풍속·기온·중량에 따른 <b>배터리 소비율(Wh/km)</b>을 예측합니다.
      </div>

      <div style="margin-bottom:4px;">
        <b>② 비행 전 필수 체크</b><br>
        다른 기체에도 참고용으로 쓸 수 있지만, 비행 전 <b>현장 바람·지형·GPS 상태</b>를 확인하고
        <b>SOC 20~30%</b>는 항상 여유로 남겨야 합니다.
      </div>

      <div style="margin-bottom:0;">
        <b>③ 실제 비행 거리 오차</b><br>
        온도·난류·배터리 노후도 등 환경에 따라
        <b>예측값보다 실제 비행 거리가 더 짧을 수 있습니다.</b>
      </div>

    </div>
  </div>

  <div class="legend-footer-m">
    AI-Drone Flight Distance Optimization (Mobile)
  </div>

</div>
"""

st.markdown(legend_html_mobile, unsafe_allow_html=True)


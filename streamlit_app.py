#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-Drone — Flight Range & Airspace Overlay (wind-to-direction + shifted circle)
- Open-Meteo 실시간 풍속/풍향/기온 연동
- 소비율(Wh/km) 기반 도달 반경 계산
- 출발지/목적지 클릭 → 거리·SOC·가능/불가 판정
- 바람 부는 방향(to)을 기준으로 반경 원의 중심 이동
- AIP KMZ + GeoJSON 공역 레이어 표시
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
# 0. 기본 설정
# ============================================================

st.set_page_config(page_title="AI-Drone", layout="wide")

st.markdown(
    """
    <div style='text-align:center; margin-top:12px; margin-bottom:18px; line-height:1.4;'>

      <div style='font-size:28px; font-weight:800; color:#0F172A;'>
        AI-Drone Flight Distance Optimization
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
    """0°=N, 90°=E, 180°=S, 270°=W 기준 8방향 텍스트"""
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
    # ✅ API: FROM 방향 → 우리가 보는 건 바람이 "불어가는 방향(to)"
    wind_deg_to = (wind_deg_from + 180.0) % 360.0
else:
    wind_deg_from = None
    wind_deg_to   = None

ss["wind_deg_from"] = wind_deg_from
ss["wind_deg_to"]   = wind_deg_to

# ============================================================
# 4. 상단 날씨 블록 (바람 부는 방향 기준 표시)
# ============================================================

if (wind_spd_mps is None) or (wind_deg_from is None) or (temp_c_now is None):
    st.info("실시간 바람/기온 데이터를 불러올 수 없습니다. (기본값: 10 m/s, 20°C 사용)")
else:
    card_to = deg_to_cardinal(float(wind_deg_to))
    arr_to  = arrow_from_cardinal(card_to)

    st.markdown(
        """
        <style>
        .wx-wrap {
            display:flex; justify-content:center; gap:60px;
            text-align:center; margin-top:6px; margin-bottom:6px;
            flex-wrap:wrap;
        }
        .wx-box  {
            display:flex; flex-direction:column; align-items:center;
            min-width:130px;
        }
        .wx-title {
            font-weight:600; font-size:15px; color:#222;
            margin-bottom:4px; line-height:1.2;
        }
        .wx-val {
            font-weight:700; font-size:20px; color:#111;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="wx-wrap">
            <div class="wx-box">
                <div class="wx-title">풍속</div>
                <div class="wx-val">{float(wind_spd_mps):.2f} m/s</div>
            </div>
            <div class="wx-box">
                <div class="wx-title">풍향(도)</div>
                <div class="wx-val">{float(wind_deg_to):.0f}°</div>
            </div>
            <div class="wx-box">
                <div class="wx-title">바람 부는 방향</div>
                <div class="wx-val">{card_to} {arr_to}</div>
            </div>
            <div class="wx-box">
                <div class="wx-title">기온</div>
                <div class="wx-val">{float(temp_c_now):.1f} °C</div>
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

def add_circle_km(m, lat, lon, km, color_hex="#22c55e", weight=6, alpha=0.25):
    """위도/경도 기준으로 '원처럼 보이는' 폴리곤 추가"""
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
    """
    Km당 에너지 소비량(Wh/km) 추정
    - 기준: 무풍, 20°C, 페이로드 0g, 40 km/h에서 최소 소비율
    - 풍속/기온/페이로드에 따라 '최적 속도'가 약간씩 이동하도록 설계
    """

    # -------------------------------
    # 1) 기준 최적 속도 v_opt 계산
    # -------------------------------
    # 기본 최적 속도: 40 km/h
    v_opt = 40.0

    # (1) 풍속이 강할수록 약간 속도를 낮추는 방향
    #     - wspd 5 m/s 기준, 15 m/s 정도에서 최대 -4 km/h 정도
    wind_term = np.clip((wspd - 5.0) / 10.0, -2.0, 3.0)
    v_opt -= 2.0 * wind_term

    # (2) 페이로드가 무거울수록 약간 보수적으로 (속도 ↓)
    #     - 0g ~ 504g 기준, 최대 -3 km/h 정도
    payload_term = np.clip(payload_g / 504.0, 0.0, 1.0)
    v_opt -= 3.0 * payload_term

    # (3) 기온이 낮을수록 배터리 효율이 떨어지니 약간 속도 ↓
    #     - 20°C 기준, 0°C에서 -3 km/h 정도
    temp_term = np.clip((20.0 - temp_c_in) / 20.0, -2.0, 2.0)
    v_opt -= 1.5 * temp_term

    # 최적 속도 범위 클램핑 (너무 느리거나 빠르지 않게)
    v_opt = float(np.clip(v_opt, 26.0, 46.0))

    # -------------------------------
    # 2) 속도에 따른 기본 소비율 (v_opt에서 최소)
    # -------------------------------
    base_wh = 2.4  # 대략적인 기준 값 (상대값이라 정확할 필요는 없음)

    # v_opt에서 최소가 되도록 U자형 곡선
    # delta: v_opt에서 ±10 km/h 벗어날수록 소비율 증가
    delta = (speed_kmh - v_opt) / 10.0
    speed_factor = 1.0 + 0.09 * (delta ** 2)  # 곡률 조정 (0.09 정도면 적당히 변화)

    # -------------------------------
    # 3) 풍속 / 온도 / 페이로드 보정
    # -------------------------------
    # 풍속이 강할수록 저항 ↑
    wind_factor = 1.0 + 0.03 * np.clip(wspd, 0.0, 20.0)

    # 기온 낮을수록 효율 ↓ (20°C 기준)
    temp_factor = 1.0 + 0.25 * np.clip((20.0 - temp_c_in) / 40.0, -0.8, 1.0)

    # 페이로드 무거울수록 소비 ↑
    weight_factor = 1.0 + 0.4 * np.clip(payload_g / 504.0, 0.0, 1.0)

    eff = base_wh * speed_factor * wind_factor * temp_factor * weight_factor

    return float(np.clip(eff, 1.5, 10.0))

def best_speed_and_whkm(wspd, temp_c_in, payload_g):
    """
    주어진 환경에서 Wh/km가 최소가 되는 속도와 그때의 Wh/km 반환
    - 속도 후보 범위: 24 ~ 56 km/h (Air3 운용 범위 안에서)
    """
    speeds_kmh = np.arange(24.0, 57.0, 1.0)  # 24, 25, ..., 56 km/h

    effs = np.array([
        predict_whkm(v, wspd, temp_c_in, payload_g)
        for v in speeds_kmh
    ])

    if len(effs) == 0 or np.all(np.isnan(effs)):
        # 문제가 생기면 기본값: 40 km/h, 3.0 Wh/km 정도로 리턴
        return 40.0, 3.0

    i = int(np.nanargmin(effs))
    best_v = float(speeds_kmh[i])

    # 추천 속도 범위 최종 클램핑
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
    """
    [편도]  shift = R * a/v
    [왕복]  shift = 2R * a/v   (R = '왕복 반영된 R', 즉 표시 반경)
    바람이 '불어가는 방향(to)'으로 shift.
    """
    circle_center_lat, circle_center_lon = origin_lat, origin_lon

    if (wind_deg_to is None) or (R_display_km is None) or (R_display_km <= 0):
        return circle_center_lat, circle_center_lon
    if cruise_ms <= 0 or wind_speed_ms <= 0:
        return circle_center_lat, circle_center_lon

    ratio = float(wind_speed_ms) / float(cruise_ms)  # a/v

    if trip_type == "편도":
        shift_km = R_display_km * ratio
    else:
        shift_km = 2.0 * R_display_km * ratio

    if shift_km <= 0:
        return circle_center_lat, circle_center_lon

    theta = np.deg2rad(wind_deg_to)   # 0=N, 90=E
    dy_km = shift_km * np.cos(theta)  # 북(+)/남(-)
    dx_km = shift_km * np.sin(theta)  # 동(+)/서(-)

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
# 7. 사이드바 UI
# ============================================================

with st.sidebar:

    # 배터리 기본값 범위 클램핑
    default_cap = float(ss.get("ui_cap_wh", AIR3_WH))
    default_cap = min(max(default_cap, 20.0), 300.0)

    st.markdown(
        "<div style='text-align:center; font-size:20px; font-weight:600; margin-bottom:6px;'>DJI Air3 기준</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='text-align:center; font-size:14px; margin-top:6px;'>배터리 용량(Wh)</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style='text-align:center; color:#6b7280; font-size:12px;
                    margin-top:15px; margin-bottom:10px;'>
            임무는 배터리 <b>80%</b>까지만 사용합니다.
        </div>
        """,
        unsafe_allow_html=True
    )

    ss['ui_cap_wh'] = st.number_input(
        label=" ",               # 실제 레이블은 숨김
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
.sb-wrap { display:flex; flex-direction:column; gap:8px; }
.sb-row  { display:flex; justify-content:space-between; }
.sb-key  { color:#374151; font-weight:600; }
.sb-val  { color:#111827; font-weight:700; }
</style>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
<div class="sb-wrap">
  <div class="sb-row"><span class="sb-key">좌표 기준</span>
      <span class="sb-val">{meteo_lat:.4f}, {meteo_lon:.4f}</span></div>
  <div class="sb-row"><span class="sb-key">풍속</span>
      <span class="sb-val">{float(wind_spd_mps):.2f} m/s</span></div>
  <div class="sb-row"><span class="sb-key">풍향(도)</span>
      <span class="sb-val">{float(wind_deg_to):.0f}°</span></div>
  <div class="sb-row"><span class="sb-key">바람 부는 방향</span>
      <span class="sb-val">{card_sb} {arr_sb}</span></div>
  <div class="sb-row"><span class="sb-key">기온</span>
      <span class="sb-val">{float(temp_c_now):.1f} °C</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        if meteo_ts:
            meteo_date = meteo_ts.split("T")[0]

            st.markdown(
                f"""
<div style='text-align:center; color:#6b7280; font-size:12px; margin-top:4px;'>
    Open-Meteo 기준<br>
    <b>{meteo_date}</b>
</div>
""",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # 좌표 표시
    c1, c2 = st.columns([1, 0.45])
    with c1:
        st.subheader("클릭 시 좌표")
    with c2:
        st.toggle("", value=ss.get("show_coords", True), key="show_coords")

    if ss.get("show_coords", True):
        st.write("출발지:",
                 f"{ss.origin[0]:.6f}, {ss.origin[1]:.6f}" if ss.origin else "(미지정)")
        st.write("목적지:",
                 f"{ss.dest[0]:.6f}, {ss.dest[1]:.6f}" if ss.dest else "(미지정)")
        if st.button("출발/목적지 초기화"):
            ss.click_count = 0
            ss.origin = None
            ss.dest = None
            ss.circle_center = None
            st.rerun()

    st.markdown("---")

    # 비행 설정
    c1, c2 = st.columns([1, 0.45])
    with c1:
        st.subheader("비행")
    with c2:
        st.toggle("", value=ss.get("show_range", True), key="show_range")

    if ss.get("show_range", True):
        # 속도 옵션을 km/h 기준으로 정의
        speed_options_kmh = [20, 30, 40, 50]

        prev_ms  = float(ss.get("ui_cruise_ms", 10.0))
        prev_kmh = round(prev_ms * 3.6 / 10) * 10
        if prev_kmh not in speed_options_kmh:
            prev_kmh = 30  # 기본값

        selected_kmh = st.select_slider(
            "비행 속도 (km/h)",
            options=speed_options_kmh,
            value=prev_kmh,
        )

        # 슬라이더 아래 회색 한줄 설명 (중앙 정렬)
        st.markdown(
            """
            <div style="text-align:center; color:#6b7280; font-size:12px; margin-top:-8px; margin-bottom:6px;">
                40 km/h는 드론의 기본 순항 속도입니다.
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
    else:
        trip_type = "편도"

    st.markdown("---")

    # 공역 레이어 설정
    c1, c2 = st.columns([1, 0.45])
    with c1:
        st.subheader("공역 레이어")
    with c2:
        st.toggle("", value=ss.get("airspace_visible", True), key="airspace_visible")

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
    AIRSPACE_LABEL_KO = {
        "prohibited": "금지구역 (P)", "restricted": "제한구역 (R)", "danger": "위험구역 (D)",
        "ctr": "관제권 (CTR/ATZ)", "tma": "접근관제구역 (TMA)", "cta": "통제공역 (CTA)",
        "training": "훈련공역 (MOA)", "adiz": "방공식별구역 (ADIZ)", "fir": "비행정보구역 (FIR)",
        "airway": "항로 (AWY)", "boundary": "경계/기타",
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
        grid = st.columns(2)
        for i, key in enumerate(CAT_ORDER):
            with grid[i % 2]:
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin-top:6px'>"
                    f"<span style='display:inline-block;width:14px;height:14px;"
                    f"background:{AIRSPACE_COLOR[key]};border-radius:3px;opacity:.9'></span>"
                    f"<span style='font-size:13px;color:#0f172a'>{AIRSPACE_LABEL_KO[key]}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                ss.airspace_on[key] = st.toggle(
                    "",
                    value=ss.airspace_on[key],
                    key=f"tog_{key}",
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

# 추천 속도 (현재 조건 최소 Wh/km)
best_now, _ = best_speed_and_whkm(wind_speed, temp_c, payload_g)
best_now = float(np.clip(best_now, 26.0, 46.0))

# 현재 설정 속도에서의 Wh/km
wh_now = predict_whkm(cruise_kmh, wind_speed, temp_c, payload_g)

# 실측 테이블 기반 base range
base_km = lookup_base_range_from_table(wind_speed, payload_g)

if base_km is not None:
    wh_ref = predict_whkm(36.0, wind_speed, 20.0, payload_g)  # 기준 속도/온도
    ratio = wh_ref / max(wh_now, 1e-6)
    R_full = base_km * float(np.clip(ratio, 0.4, 1.6))        # 100% 기준 거리
    R_estimate_base = R_full * USABLE_SOC_FRAC                # 80% 사용
else:
    R_estimate_base = (cap_usable_Wh / wh_now) if (wh_now and cap_usable_Wh > 0) else None

R_max_base = R_estimate_base      # 왕복 전, 에너지 기준 전체 거리
wh_leg     = wh_now               # 풍향 보정은 여기선 안 넣음

# 반경 (표시용, 편도/왕복 반영)
R_max_display = R_max_base
if R_max_display is not None and trip_type == "왕복":
    R_max_display /= 2.0

# 세션에 반경 저장
if (R_max_display is not None) and (R_max_display > 0):
    ss.last_R = float(R_max_display)

# 원 중심 계산 (원 shift 로직)
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

# ---------------- SOC/판정 ----------------
soc_display = None
verdict_display = None

# 설명용 메시지 변수 미리 초기화
full_reason_html = ""
reason_html = ""

if origin and dest:
    # 1) 기본 직선거리 (비교용 / 안내용)
    dist_geom_km = haversine_km(origin[0], origin[1], dest[0], dest[1])

    # 2) 바람 방향을 반영한 "유효 거리" 계산 -----------------------
    dist_for_range  = dist_geom_km   # 기본값
    dist_for_energy = dist_geom_km

    if (R_max_display is not None) and (R_max_display > 0.0) \
       and (wind_speed is not None) and (cruise_ms > 0.0):

        # 바람 "부는 방향(To)" 각도 (세션에 저장된 값 활용)
        wind_deg_to_loc = ss.get("wind_deg_to", None)
        if wind_deg_to_loc is None:
            # 정의 안 되어 있으면 FROM 방향을 fallback 으로 사용
            wind_deg_to_loc = ss.get("wind_deg_from", 0.0)

        try:
            theta = np.deg2rad(float(wind_deg_to_loc))
        except Exception:
            theta = 0.0  # 에러 나면 북쪽 기준

        # a/v 비율 (a: 바람속도, v: 드론속도)
        ratio = float(wind_speed) / max(float(cruise_ms), 1e-6)

        # 편도 : shift = R * a/v
        # 왕복 : shift = 2R * a/v
        if trip_type == "왕복":
            shift_km = 2.0 * R_max_display * ratio
        else:
            shift_km = 1.0 * R_max_display * ratio

        # 바람이 부는 방향으로 원 중심 이동 (km 기준 → 위/경도)
        lat0, lon0 = origin
        dy_km = shift_km * np.cos(theta)   # 북(+)/남(-)
        dx_km = shift_km * np.sin(theta)   # 동(+)/서(-)

        lat_shift = lat0 + dy_km * DEG_PER_KM
        lon_shift = lon0 + dx_km * DEG_PER_KM / max(np.cos(np.deg2rad(lat0)), 1e-9)

        # 이동된 중심 기준 목적지까지 거리 = 바람 반영 유효 거리
        dist_eff_km = haversine_km(lat_shift, lon_shift, dest[0], dest[1])

        dist_for_range  = dist_eff_km
        dist_for_energy = dist_eff_km

    # 3) 필요 에너지 / SOC 계산 (유효 거리 기준) -------------------
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

    # 4) 가능/불가 판정 -------------------------------------------
    ok = (
        R_max_display is not None
        and dist_for_range <= R_max_display
        and soc_full is not None
        and soc_full <= 80.0          # 80%까지만 사용
    )

    verdict_display = "가능" if ok else "불가"

    # 5) 사유 텍스트 ----------------------------------------------
    reason_lines = []

    # 거리 관련 문구 (유효 거리 vs 반경, 괄호에 실제 직선거리)
    if R_max_display is not None:
        if dist_for_range > R_max_display:
            reason_lines.append(
                f"<span style='color:#ef4444; font-weight:600;'>거리 초과</span> "
                f"(비행 가능 거리 {R_max_display:.2f} km / "
                f"유효 {dist_for_range:.2f} km / 직선 {dist_geom_km:.2f} km)"
            )
        else:
            reason_lines.append(
                f"<span style='color:#16a34a; font-weight:600;'>거리 여유 있음</span> "
                f"(비행 가능 거리 {R_max_display:.2f} km / "
                f"유효 {dist_for_range:.2f} km / 직선 {dist_geom_km:.2f} km)"
            )

    # 배터리 관련 문구 (SOC도 유효 거리 기준)
    if soc_display is not None:
        if soc_display > 80.0:
            reason_lines.append(
                f"<span style='color:#ef4444; font-weight:600;'>배터리 부족</span> "
                f"(필요 {soc_display:.1f}% / 80% 한계 초과)"
            )
        else:
            reason_lines.append(
                f"<span style='color:#16a34a; font-weight:600;'>배터리는 충분함</span> "
                f"(필요 {soc_display:.1f}% / 80% 이내)"
            )

    reason_html = "<br>".join(reason_lines)

    # 6) 해결 방법 박스 --------------------------------------------
    if verdict_display == "불가":
        solution_html = """
<div style='margin-top:4px; font-size:13px; color:#4b5563;'>
  <b>해결 방법</b><br>
  • <b>속도를 조정</b>해 에너지 효율을 높여보세요.<br>
  • <b>목적지 위치를 더 가깝게</b> 조정해 보세요.<br>
  • 사이드바의 <b>적재 중량(g)</b>을 줄여 페이로드를 가볍게 해 보세요.<br>
  • 비행 유형을 <b>왕복 → 편도</b>로 바꾸면 돌아오는 비행이 없어 필요한 배터리가 줄어듭니다.
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
# 9. 상단 메트릭 바
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
.metrics-bar {{
  position: sticky; top: 8px; z-index: 50;
  margin: 0 auto 10px; max-width: 1100px;
  display: flex; justify-content: center;
}}
.metrics-wrap {{
  display: flex; gap: 16px; flex-wrap: wrap;
  justify-content: space-evenly;
  background: rgba(255,255,255,.98);
  border: 1px solid #e5e7eb; border-radius: 14px;
  padding: 10px 14px; box-shadow: 0 4px 14px rgba(0,0,0,.06);
  width: 100%;
}}
.mbox {{ min-width: 150px; text-align: center; }}
.mtitle {{ font-size: 12px; color:#64748b; font-weight: 700; margin-bottom: 2px; }}
.mvalue {{ font-size: 20px; font-weight: 900; color:#0f172a; }}
@media (max-width: 760px) {{ .mbox {{ min-width: 46%; }} }}
</style>
<div class="metrics-bar">
  <div class="metrics-wrap">
    <div class="mbox">
      <div class="mtitle">추천 비행 속도</div>
      <div class="mvalue">{speed_txt}</div>
    </div>
    <div class="mbox">
      <div class="mtitle">Km당 배터리 소모량</div>
      <div class="mvalue">{whkm_txt}</div>
    </div>
    <div class="mbox">
      <div class="mtitle">비행 가능 거리</div>
      <div class="mvalue">{rmax_txt}</div>
    </div>
    <div class="mbox">
      <div class="mtitle">예상 배터리 사용량</div>
      <div class="mvalue">{batt_txt}</div>
    </div>
    <div class="mbox">
      <div class="mtitle">비행 가능 여부</div>
      <div class="mvalue" style="color:{verdict_color}">{verdict_txt}</div>
    </div>
  </div>
</div>
"""
st.markdown(metrics_bar_html, unsafe_allow_html=True)

if full_reason_html:
    reason_box_html = f"""
<div style="
    margin: 10px auto 6px;
    max-width: 860px;
    padding: 10px 14px;
    border-radius: 10px;
    background: rgba(255,255,255,0.95);
    border: 1px solid #e5e7eb;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    font-size: 14px;
    color: #1f2937;
">
  {full_reason_html}
</div>
"""
    st.markdown(reason_box_html, unsafe_allow_html=True)

# ============================================================
# 10. 지도 생성 및 표시 요소
# ============================================================

base_map = folium.Map(
    location=[CENTER_LAT, CENTER_LON],
    zoom_start=10,
    control_scale=True,
)

# 공역 오버레이
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

# 출발지 + 반경
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
            6,
            0.25,
        )

# 목적지 경로/판정
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
        weight=5,
        opacity=0.95,
    ).add_to(fg)
    folium.Marker(
        ss.dest,
        tooltip=f"목적지 — {dist_km:.2f} km · {'가능' if ok else '불가'}",
        icon=folium.Icon(color=dot_col),
    ).add_to(fg)

# 리렌더 키
ui_sig = (
    f"{ss['ui_cruise_ms']}-{payload_g}-{cap_Wh}-"
    f"{wind_speed:.2f}-{temp_c}-"
    f"{trip_type}-"
    f"{1 if ss.origin else 0}-{1 if ss.dest else 0}"
)

ret = st_folium(
    copy.deepcopy(base_map),
    feature_group_to_add=fg,
    width=1100,
    height=640,
    returned_objects=["last_clicked"],
    key=ui_sig,
)

# 클릭 처리
clicked = ret.get("last_clicked")
if clicked:
    lat = float(clicked["lat"])
    lon = float(clicked["lng"])

    if is_in_forbidden(lat, lon):
        alert_html = """
        <div style="
            margin-top:8px;
            padding:12px 15px;
            border-radius:10px;
            background:#f3e8ff;
            color:#1f2937;
            font-size:14px;
            line-height:1.5;
        ">
            &nbsp;
            <b><span style="color:#dc2626;">비행 금지 공역</span></b>
            (보라색 영역)입니다.&nbsp;
            <span style="color:#1d4ed8; font-weight:600;">다른 위치를 선택해주세요.</span>
            &nbsp;
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
# 11. 하단 설명 블록 (Legend) — 공역 + 지표 설명
# ============================================================

legend_html_web = """
<style>
.legend-box-web{
  margin:12px auto 18px;
  padding:18px 20px 14px;
  border:1px solid #e5e7eb;
  border-radius:14px;
  background:#ffffff;
  color:#111827;
  font-size:14px;
  max-width:1100px;
  line-height:1.6;
}
.legend-title{
  font-size:15px;
  font-weight:800;
  color:#111827;
  margin-bottom:4px;
}
.legend-section{margin-top:10px;}
.legend-box-web ul{
  margin:4px 0 0 0;
  padding-left:1.1rem;
  list-style-position:outside;
}
.legend-box-web li{margin:2px 0;}
.legend-colors{
  margin-top:6px;
  font-size:13px;
  line-height:1.5;
}
.legend-colors span{font-weight:600;}
.legend-note{
  font-size:12px;
  color:#6b7280;
  margin-top:6px;
}
.legend-footer{
  text-align:center;
  margin-top:14px;
  font-size:13px;
  color:#4b5563;
  font-weight:600;
}
</style>

<div class="legend-box-web">

<div class="legend-section">
  <div class="legend-title">지도를 이렇게 사용해보세요</div>
  <ul>
    <li>지도를 클릭해 <b>출발지 → 목적지</b>를 순으로 선택해 주세요.</li>
    <li>경로선이 <span style="color:#16a34a; font-weight:700;">초록색</span>이면 <b><span style="color:#16a34a; font-weight:700;">비행 가능</span></b>,
        <span style="color:#ef4444; font-weight:700;">빨간색</span>이면
        <b>배터리 부족·거리 초과로 <span style="color:#ef4444; font-weight:700;">비행 불가</span></b>입니다.</li>
  </ul>
</div>

<div class="legend-section">
  <div class="legend-title">공역 안내</div>
  <ul>
    <li>지도에 보이는 <span style="color:#8b5cf6; font-weight:700;">보라색 윤곽선</span>은
        AIP 공역(P/R/D·CTR·TMA 등)을 하나로 합친 <b>“주의 구역” 테두리</b>입니다.</li>
    <li>한눈에 보기 쉽게 하려고,
        <b>세부 구분(금지/제한/위험, 고도 정보 등)은 단순화</b>되어 있습니다.</li>
    <li>실제 비행 전에 반드시 <b>국토부 드론원스톱·AIP 원문</b>에서
        공역 종류와 고도 제한을 다시 확인해 주세요.</li>
  </ul>

<div class="legend-section">
  <div class="legend-title" style="font-size:15px; font-weight:800; margin-bottom:6px;">
    공역 색상 예시
  </div>

  <div class="legend-colors" style="margin-top:6px; font-size:14px; line-height:1.6;">
    <span style="color:#ff0033; font-weight:600;">■</span> 금지(P)&nbsp;&nbsp;
    <span style="color:#ff8800; font-weight:600;">■</span> 제한(R)&nbsp;&nbsp;
    <span style="color:#ff3d00; font-weight:600;">■</span> 위험(D)&nbsp;&nbsp;
    <span style="color:#0066ff; font-weight:600;">■</span> CTR&nbsp;&nbsp;
    <span style="color:#3377ff; font-weight:600;">■</span> TMA&nbsp;&nbsp;
    <span style="color:#8a2be2; font-weight:600;">■</span> CTA&nbsp;&nbsp;
    <span style="color:#00b050; font-weight:600;">■</span> 훈련(MOA)&nbsp;&nbsp;
    <span style="color:#00ccff; font-weight:600;">▭</span> 항로(AWY)&nbsp;&nbsp;
    <span style="color:#9ca3af; font-weight:600;">▭</span> 경계/기타
  </div>

  <div class="legend-note" style="font-size:12px; color:#6b7280; margin-top:6px;">
    ※ 위 색상은 데이터 원본 기준 구분이며, 지도에서는 가독성을 위해 대부분 
    <b><span style="color:#8b5cf6; font-weight:700;">보라색 윤곽선</span> 하나로 단순화</b>하여 표시됩니다.
  </div>
</div>

<!-- 핵심 지표 설명 -->
<div class="legend-section">
<div class="legend-title">비행 핵심 지표 한눈에 보기</div>
<ul>
<div style="margin-top:6px; font-size:13px; line-height:1.8;">
  <div style="margin-bottom:6px;">
    <b>추천 비행 속도</b><br>
    → 현재 <b>풍속·풍향·기온·비행 속도·적재 중량</b>을 기준으로 가장 효율적인 속도입니다.
  </div>

  <div style="margin-bottom:6px;">
    <b>Wh/km (Km당 배터리 소모량)</b><br>
    → 지금 속도로 <b>1km 비행 시 소모되는 배터리 양</b>입니다.
  </div>

  <div style="margin-bottom:6px;">
    <b>비행 가능 거리</b><br>
    → 현재 조건에서 드론이 <b>안전하게 이동할 수 있는 최대 거리(km)</b>입니다.
  </div>

  <div style="margin-bottom:6px;">
    <b>필요 배터리 (80% 기준)</b><br>
    → <b>20%는 안전 여유(RTH 용)</b>으로 남기고 실제 임무에 사용하는 배터리 비율입니다.
  </div>

  <div style="margin-bottom:2px;">
    <b>비행 가능 여부</b><br>
    → 거리·배터리 조건을 모두 충족하면 “<span style="color:#16a34a; font-weight:700;">비행 가능</span>”, 하나라도 초과하면 <b>“<span style="color:#ef4444; font-weight:700;">비행 불가</span>”</b>로 표시됩니다.
  </div>

</div>

<div style="text-align:center; margin-top:16px; font-size:17px; font-weight:700; color:#1f2937;">
  AI-Drone Flight Distance Optimization
</div>

"""

st.markdown(legend_html_web, unsafe_allow_html=True)

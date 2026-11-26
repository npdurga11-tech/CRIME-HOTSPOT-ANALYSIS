"""Interactive maps.

``build_map`` writes a single self-contained HTML file using Leaflet + leaflet.heat (the same
libraries Folium wraps), so it needs no Python mapping package. ``build_folium_map`` produces the
equivalent map with Folium if you have it installed (``pip install folium``).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEQ_STOPS = ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
HEAT_GRADIENT = {0.15: "#cde2fb", 0.4: "#6da7ec", 0.65: "#2a78d6", 0.85: "#1c5cab", 1.0: "#0d366b"}
ACCENT = "#eb6834"


def grid_points(lat, lon, proj, weights=None, cell_m: float = 150.0, clip_pct: float = 98.0) -> list:
    """Aggregate incidents to a metric grid -> [[lat, lon, intensity 0-1], ...] (small HTML, no raw points)."""
    x, y = proj.forward(np.asarray(lat), np.asarray(lon))
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    df = pd.DataFrame({"ix": np.floor(x / cell_m).astype(int), "iy": np.floor(y / cell_m).astype(int), "w": w})
    g = df.groupby(["ix", "iy"], as_index=False)["w"].sum()
    la, lo = proj.inverse((g["ix"] + 0.5) * cell_m, (g["iy"] + 0.5) * cell_m)
    inten = np.clip(g["w"] / np.percentile(g["w"], clip_pct), 0, 1)
    return [[round(float(a), 5), round(float(b), 5), round(float(c), 3)] for a, b, c in zip(la, lo, inten)]


def hotspot_popup(r: pd.Series) -> str:
    yoy = "n/a" if pd.isna(r["yoy_change_pct"]) else f"{r['yoy_change_pct']:+.0f}%"
    return (f"<b>Hotspot #{int(r['rank'])}</b><br>{int(r['incidents']):,} incidents "
            f"({r['share_of_all_risk_pct']:.1f}% of citywide severity-weighted risk)<br>"
            f"Violent share: {r['violent_share_pct']:.0f}%<br>"
            f"Peak window: {html.escape(r['peak_window'])} ({r['peak_window_share_pct']:.0f}% of risk), "
            f"busiest day: {r['peak_dow']}<br>Top types: {html.escape(r['top_types'])}<br>"
            f"Last 12 months vs previous 12: {yoy}")


def build_payload(df_geo, rank, hotspots, polygons, zones, proj, banner: str, title: str) -> dict:
    lat, lon = df_geo["latitude"].to_numpy(), df_geo["longitude"].to_numpy()
    hour = df_geo["hour"].to_numpy()
    night = (hour >= 20) | (hour < 4)
    day = (hour >= 8) & (hour < 18)
    violent = (df_geo["group"] == "violent").to_numpy()
    heat = {
        "All incidents": grid_points(lat, lon, proj),
        "Violent crime": grid_points(lat[violent], lon[violent], proj),
        "Night (20:00-04:00)": grid_points(lat[night], lon[night], proj),
        "Daytime (08:00-18:00)": grid_points(lat[day], lon[day], proj),
    }
    hs = []
    for _, r in hotspots.iterrows():
        poly = polygons.get(int(r["rank"]))
        hs.append({
            "rank": int(r["rank"]), "centroid": [round(r["centroid_lat"], 5), round(r["centroid_lon"], 5)],
            "polygon": None if poly is None else [[round(float(a), 5), round(float(b), 5)]
                                                  for a, b in zip(poly[0], poly[1])],
            "popup": hotspot_popup(r),
        })
    zs = []
    if zones is not None and {"latitude", "longitude"} <= set(zones.columns):
        outcome = "rate_per_1000" if "rate_per_1000" in zones else "incidents_per_year"
        for _, z in zones.iterrows():
            if pd.isna(z["latitude"]):
                continue
            zs.append({"name": str(z["zone_name"]), "lat": round(float(z["latitude"]), 5),
                       "lon": round(float(z["longitude"]), 5), "value": float(z[outcome]),
                       "hardship": float(z.get("hardship_index", np.nan)) if "hardship_index" in z else None,
                       "popup": (f"<b>{html.escape(str(z['zone_name']))}</b><br>"
                                 f"{z[outcome]:.1f} {'per 1,000 residents/yr' if outcome == 'rate_per_1000' else 'incidents/yr'}"
                                 + (f"<br>Hardship index: {z['hardship_index']:.0f}" if "hardship_index" in z else "")
                                 + (f"<br>Poverty: {z['poverty_pct']:.1f}%" if "poverty_pct" in z else ""))})
    return {"title": title, "banner": banner, "center": [round(float(lat.mean()), 4), round(float(lon.mean()), 4)],
            "heat": heat, "hotspots": hs, "zones": zs, "heat_radius": 14, "heat_blur": 20}


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
 html,body,#map{height:100%;margin:0}
 body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
 .panel{background:rgba(252,252,251,.95);padding:10px 12px;border-radius:8px;box-shadow:0 1px 5px rgba(0,0,0,.3);
        color:#0b0b0b;font-size:12px;line-height:1.45;max-width:280px}
 .panel h4{margin:0 0 4px;font-size:13px}
 .panel .warn{background:#fff3cd;border-left:3px solid #eda100;padding:4px 6px;margin:6px 0;color:#0b0b0b}
 .ramp{height:10px;border-radius:3px;margin:4px 0 2px;background:linear-gradient(90deg,#cde2fb,#6da7ec,#2a78d6,#1c5cab,#0d366b)}
 .ramp-labels{display:flex;justify-content:space-between;color:#52514e;font-size:11px}
 .sw{display:inline-block;width:12px;height:12px;border:2px solid #eb6834;background:rgba(235,104,52,.15);
     vertical-align:-2px;margin-right:6px;border-radius:2px}
 .badge{background:#eb6834;color:#fff;border-radius:50%;width:22px;height:22px;line-height:22px;text-align:center;
        font-weight:700;font-size:12px;border:2px solid #fcfcfb;box-shadow:0 0 2px rgba(0,0,0,.5)}
</style></head><body><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<script>
const D = /*__DATA__*/null;
const STOPS = ["#cde2fb","#6da7ec","#2a78d6","#1c5cab","#0d366b"];
function ramp(t){
  t = Math.max(0, Math.min(1, t)); const p = t*(STOPS.length-1), i = Math.min(Math.floor(p), STOPS.length-2), f = p-i;
  const a = STOPS[i], b = STOPS[i+1], c = k => parseInt(a.slice(k,k+2),16)*(1-f)+parseInt(b.slice(k,k+2),16)*f;
  return "rgb("+Math.round(c(1))+","+Math.round(c(3))+","+Math.round(c(5))+")";
}
const map = L.map('map', {preferCanvas:true}).setView(D.center, 11);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {attribution:'&copy; OpenStreetMap contributors &copy; CARTO', maxZoom:19}).addTo(map);

const overlays = {};
const grad = {}; for (const k in __GRADIENT__) grad[k] = __GRADIENT__[k];
for (const name in D.heat) {
  overlays["Heatmap: " + name] = L.heatLayer(D.heat[name],
    {radius:D.heat_radius, blur:D.heat_blur, maxZoom:15, minOpacity:0.25, gradient:grad});
}
const hot = L.layerGroup(), badges = L.layerGroup();
D.hotspots.forEach(h => {
  if (h.polygon && h.polygon.length >= 3) {
    L.polygon(h.polygon, {color:"#eb6834", weight:2, fillColor:"#eb6834", fillOpacity:0.12}).bindPopup(h.popup).addTo(hot);
  } else {
    L.circle(h.centroid, {radius:200, color:"#eb6834", weight:2, fillOpacity:0.12}).bindPopup(h.popup).addTo(hot);
  }
  L.marker(h.centroid, {icon:L.divIcon({className:"", html:'<div class="badge">'+h.rank+'</div>', iconSize:[22,22]})})
    .bindPopup(h.popup).addTo(badges);
});
const hotGroup = L.layerGroup([hot, badges]);
overlays["Hotspots (DBSCAN clusters)"] = hotGroup;

const zoneLayer = L.layerGroup();
const vmax = Math.max.apply(null, D.zones.map(z => z.value).concat([1]));
D.zones.forEach(z => {
  L.circle([z.lat, z.lon], {radius: 250 + 1400*Math.sqrt(z.value/vmax), color:"#0b0b0b", weight:1,
    fillColor: ramp(z.hardship == null ? 0.5 : z.hardship/100), fillOpacity:0.55}).bindPopup(z.popup).addTo(zoneLayer);
});
if (D.zones.length) overlays["Zone crime rate (size) and hardship (colour)"] = zoneLayer;

overlays["Heatmap: All incidents"].addTo(map);
hotGroup.addTo(map);
L.control.layers(null, overlays, {collapsed:false, position:"topright"}).addTo(map);

const info = L.control({position:"bottomleft"});
info.onAdd = function(){
  const d = L.DomUtil.create("div","panel");
  d.innerHTML = "<h4>"+D.title+"</h4>" +
    (D.banner ? '<div class="warn">'+D.banner+'</div>' : "") +
    '<div>Heatmap intensity (incidents per 150 m cell)</div><div class="ramp"></div>' +
    '<div class="ramp-labels"><span>fewer</span><span>more</span></div>' +
    '<div style="margin-top:6px"><span class="sw"></span>DBSCAN hotspot (numbered by severity-weighted risk)</div>' +
    '<div style="margin-top:4px;color:#52514e">Click a hotspot for details. Toggle layers at top right.</div>';
  L.DomEvent.disableClickPropagation(d); return d;
};
info.addTo(map);
</script></body></html>
"""


def build_map(payload: dict, out_path: str | Path) -> Path:
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    page = (_HTML.replace("__TITLE__", html.escape(payload["title"]))
            .replace("/*__DATA__*/null", data)
            .replace("__GRADIENT__", json.dumps({str(k): v for k, v in HEAT_GRADIENT.items()})))
    out = Path(out_path)
    out.write_text(page, encoding="utf-8")
    return out


def build_folium_map(payload: dict, out_path: str | Path) -> Path:
    """Same layers via Folium. NOTE: written without Folium installed in the build sandbox, so this
    function has only been syntax-checked, not run. ``pip install folium`` first."""
    import folium
    from folium.plugins import HeatMap

    m = folium.Map(location=payload["center"], zoom_start=11, tiles="CartoDB positron")
    for name, pts in payload["heat"].items():
        fg = folium.FeatureGroup(name=f"Heatmap: {name}", show=(name == "All incidents"))
        HeatMap(pts, radius=payload["heat_radius"], blur=payload["heat_blur"], min_opacity=0.25,
                gradient={str(k): v for k, v in HEAT_GRADIENT.items()}).add_to(fg)
        fg.add_to(m)
    hot = folium.FeatureGroup(name="Hotspots (DBSCAN clusters)")
    for h in payload["hotspots"]:
        popup = folium.Popup(h["popup"], max_width=320)
        if h["polygon"] and len(h["polygon"]) >= 3:
            folium.Polygon(h["polygon"], color=ACCENT, weight=2, fill=True, fill_color=ACCENT,
                           fill_opacity=0.12, popup=popup).add_to(hot)
        else:
            folium.Circle(h["centroid"], radius=200, color=ACCENT, weight=2, popup=popup).add_to(hot)
        folium.Marker(h["centroid"], icon=folium.DivIcon(
            html=f'<div style="background:{ACCENT};color:#fff;border-radius:50%;width:22px;height:22px;'
                 f'line-height:22px;text-align:center;font-weight:700">{h["rank"]}</div>')).add_to(hot)
    hot.add_to(m)
    if payload["zones"]:
        zl = folium.FeatureGroup(name="Zone crime rate", show=False)
        vmax = max(z["value"] for z in payload["zones"]) or 1
        for z in payload["zones"]:
            folium.Circle([z["lat"], z["lon"]], radius=250 + 1400 * (z["value"] / vmax) ** 0.5, color="#0b0b0b",
                          weight=1, fill=True, fill_opacity=0.4, popup=folium.Popup(z["popup"], max_width=260)).add_to(zl)
        zl.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    out = Path(out_path)
    m.save(str(out))
    return out

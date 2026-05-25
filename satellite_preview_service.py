#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WebView-only satellite preview service for Mustatil Qt Workspace.

Important behaviour requested by user:
- The WebView always uses ESRI World Imagery internally.
- The ESRI URL is NOT written into the GUI URL input field.
- URLs from the GUI are NOT read/accepted by the WebView preview.
- Shift+Drag or right mouse drag selects an extent and writes only
  South/West/North/East into the existing GUI input fields.
- Zoom is not written to GUI fields and not passed through the selection callback.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

WEB_TILE_SIZE = 256
ESRI_WORLD_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def sat_clamp_lat(lat: float) -> float:
    return max(min(float(lat), 85.05112878), -85.05112878)


def sat_world_px(lon: float, lat: float, z: int) -> Tuple[float, float]:
    lat = sat_clamp_lat(lat)
    n = (2 ** int(z)) * WEB_TILE_SIZE
    x = (float(lon) + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def sat_blank_tile():
    return None


def sat_decode_tile(data):
    return None


def _log(app: Any, msg: str) -> None:
    try:
        app.log(msg)
    except Exception:
        pass


def _status(app: Any, msg: str) -> None:
    # Prefer the GUI's Qt signal so worker-thread calls do not touch QLabel directly.
    try:
        sigs = getattr(app, "signals", None)
        sig = getattr(sigs, "sat_status", None) if sigs is not None else None
        if sig is not None:
            sig.emit(str(msg))
            return
    except Exception:
        pass
    try:
        if hasattr(app, "sat_view_status_label"):
            app.sat_view_status_label.setText(str(msg))
    except Exception:
        pass


def _qt_modules():
    try:
        from PySide6.QtCore import QUrl, QObject, Slot, Signal, QThread  # type: ignore
        from PySide6.QtWidgets import QSizePolicy  # type: ignore
        from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore
        from PySide6.QtWebChannel import QWebChannel  # type: ignore
        return QUrl, QObject, Slot, Signal, QThread, QSizePolicy, QWebEngineView, QWebChannel
    except Exception as exc:
        raise RuntimeError(
            "PySide6-WebEngine fehlt. Installiere in der venv: "
            ".venv\\Scripts\\python.exe -m pip install PySide6-WebEngine"
        ) from exc


def _set_any_field(widget: Any, value: Any) -> bool:
    text = f"{float(value):.8f}" if isinstance(value, (float, int)) else str(value)
    for method in ("setText", "setPlainText"):
        try:
            if hasattr(widget, method):
                getattr(widget, method)(text)
                return True
        except Exception:
            pass
    try:
        if hasattr(widget, "setValue"):
            getattr(widget, "setValue")(float(value))
            return True
    except Exception:
        pass
    try:
        if hasattr(widget, "set"):
            widget.set(text)
            return True
    except Exception:
        pass
    return False


def _set_first_existing(app: Any, names: Iterable[str], value: Any) -> bool:
    ok = False
    for name in names:
        if not hasattr(app, name):
            continue
        try:
            if _set_any_field(getattr(app, name), value):
                ok = True
        except Exception:
            pass
    return ok


def _write_selection_to_gui(app: Any, west: float, south: float, east: float, north: float) -> bool:
    west, south, east, north = float(west), float(south), float(east), float(north)
    if east <= west or north <= south:
        _status(app, "Auswahl ignoriert: Feld ist leer/ungültig")
        _log(app, "Satellite WebMap selection ignored: invalid/empty extent.")
        return False
    if abs(east - west) < 1e-9 or abs(north - south) < 1e-9:
        _status(app, "Auswahl ignoriert: Rechteck größer ziehen")
        _log(app, "Satellite WebMap selection ignored: rectangle too small.")
        return False

    app.sat_selected_bbox_lonlat = (west, south, east, north)
    app.sat_selection_bbox_lonlat = app.sat_selected_bbox_lonlat

    # Exact fields from mustatil_qt_workspace.py. These Var objects are bound to
    # the visible QLineEdit inputs in the "Selected map extent" group.
    _set_first_existing(app, ("sat_min_lat", "sat_south_var", "sat_min_lat_var", "sat_lat_min_var"), south)
    _set_first_existing(app, ("sat_min_lon", "sat_west_var", "sat_min_lon_var", "sat_lon_min_var"), west)
    _set_first_existing(app, ("sat_max_lat", "sat_north_var", "sat_max_lat_var", "sat_lat_max_var"), north)
    _set_first_existing(app, ("sat_max_lon", "sat_east_var", "sat_max_lon_var", "sat_lon_max_var"), east)

    # Do NOT write zoom anywhere. Main GUI uses self.sat_zoom for detection zoom.
    # Do NOT write/read self.sat_url_template. WebView uses internal ESRI only.

    calc = getattr(app, "satellite_calculate_selection", None)
    if callable(calc):
        try:
            calc()
        except Exception as exc:
            _log(app, "Satellite calculate selection after WebMap selection failed: " + str(exc))

    cb = getattr(app, "on_satellite_preview_selection_changed", None)
    if callable(cb):
        try:
            cb(west, south, east, north)
        except TypeError:
            # Backward compatibility only. Do not pass actual zoom from WebView.
            try:
                cb(west, south, east, north, None)
            except Exception as exc:
                _log(app, "Satellite selection callback failed: " + str(exc))
        except Exception as exc:
            _log(app, "Satellite selection callback failed: " + str(exc))
    return True


def _extract_number(d: Any, names: Iterable[str]) -> Optional[float]:
    for name in names:
        try:
            if isinstance(d, dict) and name in d:
                return float(d[name])
            if hasattr(d, name):
                return float(getattr(d, name))
        except Exception:
            continue
    return None


def _detection_to_bounds(item: Any) -> Optional[List[List[float]]]:
    """Return Leaflet rectangle bounds [[south, west], [north, east]].

    Mustatil writes satellite detections in two possible forms:
    1) in-memory bbox fields: bbox_lon_min/bbox_lat_min/bbox_lon_max/bbox_lat_max
    2) read-back GeoPackage polygons: polygon_lonlat = [(lon, lat), ...]
    The WebMap overlay must handle both, otherwise detections disappear after
    they are read back from the output file.
    """
    west = _extract_number(item, ("west", "lon_min", "min_lon", "xmin_lon", "left_lon", "bbox_lon_min"))
    east = _extract_number(item, ("east", "lon_max", "max_lon", "xmax_lon", "right_lon", "bbox_lon_max"))
    south = _extract_number(item, ("south", "lat_min", "min_lat", "ymin_lat", "bottom_lat", "bbox_lat_min"))
    north = _extract_number(item, ("north", "lat_max", "max_lat", "ymax_lat", "top_lat", "bbox_lat_max"))
    if None not in (west, east, south, north):
        w, e = sorted((float(west), float(east)))
        s, n = sorted((float(south), float(north)))
        if e > w and n > s:
            return [[s, w], [n, e]]

    for key in ("bbox_lonlat", "lonlat_bbox", "bounds_lonlat", "geo_bbox"):
        try:
            bbox = item.get(key) if isinstance(item, dict) else getattr(item, key)
            if bbox and len(bbox) == 4:
                w, s, e, n = map(float, bbox)
                w, e = sorted((w, e)); s, n = sorted((s, n))
                if e > w and n > s:
                    return [[s, w], [n, e]]
        except Exception:
            pass

    for key in ("polygon_lonlat", "poly_lonlat", "points_lonlat", "geometry_lonlat"):
        try:
            pts = item.get(key) if isinstance(item, dict) else getattr(item, key)
            if pts:
                lons, lats = [], []
                for p in pts:
                    if isinstance(p, dict):
                        lon = p.get("lon", p.get("x"))
                        lat = p.get("lat", p.get("y"))
                    else:
                        lon, lat = p[0], p[1]
                    lons.append(float(lon)); lats.append(float(lat))
                if lons and lats:
                    w, e = min(lons), max(lons)
                    s, n = min(lats), max(lats)
                    if e > w and n > s:
                        return [[s, w], [n, e]]
        except Exception:
            pass
    return None


def _collect_detections(app: Any) -> List[Dict[str, Any]]:
    # Prefer the GUI's own filtered record list so sliders affect the WebMap.
    try:
        if hasattr(app, "_satellite_visible_records") and callable(app._satellite_visible_records):
            items = list(app._satellite_visible_records())
        else:
            items = []
    except Exception:
        items = []

    if not items:
        src = None
        for name in ("satellite_detections", "sat_last_records", "sat_detections", "satellite_detection_results", "sat_detection_results", "detection_results", "detections"):
            val = getattr(app, name, None)
            if val:
                src = val
                break
        if src is None:
            return []
        try:
            items = list(src.values()) if isinstance(src, dict) else list(src)
        except Exception:
            return []

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(items[:10000]):
        bounds = _detection_to_bounds(item)
        if not bounds:
            continue
        conf = _extract_number(item, ("conf", "confidence", "score", "yolo_conf"))
        form = _extract_number(item, ("formscore", "form_score", "shape_score"))
        txt = f"Detection {i + 1}"
        if conf is not None:
            txt += f" | conf {conf:.3f}"
        if form is not None:
            txt += f" | form {form:.3f}"
        out.append({"bounds": bounds, "text": txt})
    return out


def _leaflet_html(lon: float, lat: float, zoom: int, detections: List[Dict[str, Any]]) -> str:
    return f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
<link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\"/>
<script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
<script src=\"qrc:///qtwebchannel/qwebchannel.js\"></script>
<style>
html,body,#map{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#111}}
.leaflet-container{{background:#111;cursor:grab}}.leaflet-container.selecting{{cursor:crosshair}}
.hint{{position:absolute;left:10px;bottom:10px;z-index:1000;color:#eee;background:rgba(0,0,0,.62);font:12px/1.35 Arial,sans-serif;padding:6px 8px;border-radius:4px;user-select:none}}
.crosshair{{position:absolute;left:50%;top:50%;width:18px;height:18px;margin-left:-9px;margin-top:-9px;pointer-events:none;z-index:1000}}
.crosshair:before,.crosshair:after{{content:\"\";position:absolute;background:rgba(255,255,255,.85);box-shadow:0 0 2px #000}}
.crosshair:before{{left:8px;top:0;width:2px;height:18px}}.crosshair:after{{left:0;top:8px;width:18px;height:2px}}
</style></head><body><div id=\"map\"></div><div class=\"crosshair\"></div><div id=\"hint\" class=\"hint\">Shift+Drag oder Rechts-Drag: Feld markieren</div>
<script>
(function(){{
const ESRI={json.dumps(ESRI_WORLD_IMAGERY)};
const detections={json.dumps(detections)};
let bridge=null;
const map=L.map('map',{{zoomControl:true,attributionControl:false,preferCanvas:true,inertia:true,zoomAnimation:true,fadeAnimation:true,updateWhenIdle:false,updateWhenZooming:false,wheelPxPerZoomLevel:96}}).setView([{float(sat_clamp_lat(lat))},{float(lon)}],{int(zoom)});
let layer=L.tileLayer(ESRI,{{tileSize:256,minZoom:0,maxZoom:22,maxNativeZoom:22,keepBuffer:5,updateWhenIdle:false,updateWhenZooming:false,detectRetina:false,crossOrigin:false}}).addTo(map);
const detLayer=L.layerGroup().addTo(map);
function drawDetections(list){{detLayer.clearLayers();(list||[]).forEach(function(d){{if(!d.bounds)return;let r=L.rectangle(d.bounds,{{color:'#ff0000',weight:3,opacity:1,fill:false,interactive:true}}).addTo(detLayer);if(d.text)r.bindTooltip(d.text,{{sticky:true}});}});}}
drawDetections(detections);
let selectionRect=null, selecting=false, startLatLng=null;
function hint(t){{document.getElementById('hint').textContent=t;}}
function notifyMove(){{const c=map.getCenter();hint(`Zoom ${{map.getZoom()}} | lon ${{c.lng.toFixed(7)}} lat ${{c.lat.toFixed(7)}} | detections ${{detLayer.getLayers().length}} | Shift+Drag/Rechts-Drag: Feld markieren`);if(bridge&&bridge.mapMoved)bridge.mapMoved(c.lng,c.lat,map.getZoom());}}
map.on('moveend zoomend',notifyMove);
map.getContainer().addEventListener('contextmenu',function(e){{e.preventDefault();}});
map.on('mousedown',function(e){{const oe=e.originalEvent||{{}};if(!(oe.shiftKey||oe.button===2))return;selecting=true;startLatLng=e.latlng;map.dragging.disable();map.getContainer().classList.add('selecting');if(selectionRect)map.removeLayer(selectionRect);selectionRect=L.rectangle([startLatLng,startLatLng],{{color:'#00ffff',weight:2,fill:true,fillOpacity:.12,dashArray:'5,4'}}).addTo(map);}});
map.on('mousemove',function(e){{if(selecting&&selectionRect&&startLatLng)selectionRect.setBounds(L.latLngBounds(startLatLng,e.latlng));}});
function finishSelection(e){{if(!selecting||!selectionRect)return;selecting=false;map.dragging.enable();map.getContainer().classList.remove('selecting');const b=selectionRect.getBounds();const west=b.getWest(),south=b.getSouth(),east=b.getEast(),north=b.getNorth();if(east<=west||north<=south||Math.abs(east-west)<1e-9||Math.abs(north-south)<1e-9){{hint('Auswahl ignoriert: Rechteck größer ziehen');return;}}hint(`Auswahl eingetragen: W ${{west.toFixed(8)}} S ${{south.toFixed(8)}} E ${{east.toFixed(8)}} N ${{north.toFixed(8)}}`);if(bridge&&bridge.selectionChanged)bridge.selectionChanged(west,south,east,north);}}
map.on('mouseup',finishSelection);map.on('mouseout',function(e){{if(selecting)finishSelection(e);}});
window.mustatilSetView=function(lon,lat,zoom,ignoredTemplate,newDetections){{layer.setUrl(ESRI);if(newDetections)drawDetections(newDetections);map.setView([lat,lon],zoom,{{animate:false}});setTimeout(function(){{map.invalidateSize(true);notifyMove();}},30);}};
window.mustatilSetDetections=function(newDetections){{drawDetections(newDetections||[]);}};
if(window.qt&&window.qt.webChannelTransport){{new QWebChannel(qt.webChannelTransport,function(channel){{bridge=channel.objects.mustatilBridge;notifyMove();}});}}else{{notifyMove();}}
setTimeout(function(){{map.invalidateSize(true);notifyMove();}},100);
}})();
</script></body></html>"""



def _ensure_js_bridge(app: Any, webview: Any):
    """Create a queued main-thread JavaScript bridge.

    Worker threads in the main app may call satellite_redraw_detection_overlay()
    after detection. Calling QWebEnginePage.runJavaScript() directly from those
    threads can crash Qt with: QObject::startTimer: Timers cannot be started
    from another thread. This bridge receives JS strings through a Qt signal and
    executes them on the WebEngine object's thread.
    """
    existing = getattr(app, "satellite_web_js_bridge", None)
    if existing is not None:
        return existing
    QUrl, QObject, Slot, Signal, QThread, QSizePolicy, QWebEngineView, QWebChannel = _qt_modules()

    class JsBridge(QObject):  # type: ignore[misc, valid-type]
        jsRequested = Signal(str)

        def __init__(self, view):
            super().__init__(view)
            self._view = view
            self.jsRequested.connect(self.run_js)

        @Slot(str)
        def run_js(self, code: str) -> None:
            try:
                if self._view is not None and self._view.page() is not None:
                    self._view.page().runJavaScript(str(code))
            except Exception as exc:
                _log(app, "Satellite WebMap JS execution failed: " + str(exc))

    bridge = JsBridge(webview)
    app.satellite_web_js_bridge = bridge
    return bridge


def _run_js_on_web_thread(app: Any, js: str) -> None:
    """Thread-safe runJavaScript wrapper."""
    webview = getattr(app, "satellite_web_view", None)
    if webview is None:
        return
    try:
        QUrl, QObject, Slot, Signal, QThread, QSizePolicy, QWebEngineView, QWebChannel = _qt_modules()
        bridge = _ensure_js_bridge(app, webview)
        try:
            same_thread = QThread.currentThread() == webview.thread()
        except Exception:
            same_thread = False
        if same_thread:
            bridge.run_js(str(js))
        else:
            bridge.jsRequested.emit(str(js))
    except Exception as exc:
        _log(app, "Satellite WebMap JS scheduling failed: " + str(exc))


def _make_bridge(app: Any):
    QUrl, QObject, Slot, Signal, QThread, QSizePolicy, QWebEngineView, QWebChannel = _qt_modules()

    class Bridge(QObject):  # type: ignore[misc, valid-type]
        @Slot(float, float, int)
        def mapMoved(self, lon: float, lat: float, zoom: int) -> None:
            try:
                app.sat_center_lon = float(lon)
                app.sat_center_lat = float(lat)
                app.sat_preview_z = int(zoom)
            except Exception:
                pass

        @Slot(float, float, float, float)
        def selectionChanged(self, west: float, south: float, east: float, north: float) -> None:
            ok = _write_selection_to_gui(app, float(west), float(south), float(east), float(north))
            if not ok:
                return
            _status(app, f"Auswahl eingetragen: W {west:.8f} S {south:.8f} E {east:.8f} N {north:.8f}")
            _log(app, f"Satellite WebMap selection entered into GUI fields: South={south:.8f}, West={west:.8f}, North={north:.8f}, East={east:.8f}")

    return Bridge()


def _replace_old_view_with_webview(app: Any, webview: Any) -> bool:
    old = getattr(app, "satellite_view", None)
    if old is None:
        return False
    try:
        parent = old.parentWidget()
        layout = parent.layout() if parent is not None else None
        if layout is None:
            return False
        index = layout.indexOf(old)
        if index < 0:
            return False
        layout.removeWidget(old)
        old.hide()
        old.setParent(None)
        layout.insertWidget(index, webview)
        app.satellite_view = webview
        return True
    except Exception as exc:
        _log(app, "Satellite web map replace failed: " + str(exc))
        return False


def ensure_satellite_web_view(app: Any) -> Optional[Any]:
    if getattr(app, "satellite_web_view", None) is not None:
        return app.satellite_web_view
    try:
        QUrl, QObject, Slot, Signal, QThread, QSizePolicy, QWebEngineView, QWebChannel = _qt_modules()
    except Exception as exc:
        _log(app, "Satellite web map unavailable: " + str(exc))
        _status(app, "WebMap unavailable - install PySide6-WebEngine")
        return None

    old_view = getattr(app, "satellite_view", None)
    parent_widget = old_view.parentWidget() if old_view is not None else None
    if old_view is None or parent_widget is None:
        _status(app, "WebMap waiting for satellite preview widget")
        return None

    webview = QWebEngineView(parent_widget)
    try:
        webview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    except Exception:
        pass
    bridge = _make_bridge(app)
    channel = QWebChannel(webview.page())
    channel.registerObject("mustatilBridge", bridge)
    webview.page().setWebChannel(channel)
    app.satellite_web_view = webview
    app.satellite_web_bridge = bridge
    app.satellite_web_channel = channel
    _ensure_js_bridge(app, webview)

    # The main GUI still has an old QGraphicsView overlay method with the same
    # name. Since this module replaces the preview with QWebEngineView, that old
    # method cannot draw on the WebMap. Override the instance method so calls
    # after detection and slider changes refresh Leaflet rectangles instead.
    try:
        import types
        app.satellite_redraw_detection_overlay = types.MethodType(lambda self: refresh_satellite_detections_overlay(self), app)
    except Exception:
        pass

    if not _replace_old_view_with_webview(app, webview):
        return None
    _log(app, "Satellite WebMap enabled: internal ESRI only, no GUI URL read/write, GUI extent fields, red detection overlay.")
    return webview


def refresh_satellite_preview(app: Any) -> None:
    webview = ensure_satellite_web_view(app)
    if webview is None:
        return

    # Do NOT read app.sat_url_template. Do NOT write ESRI into the GUI.
    lon = float(getattr(app, "sat_center_lon", 0.0))
    lat = float(getattr(app, "sat_center_lat", 0.0))
    zoom = int(getattr(app, "sat_preview_z", 18))
    detections = _collect_detections(app)
    QUrl, QObject, Slot, Signal, QThread, QSizePolicy, QWebEngineView, QWebChannel = _qt_modules()
    html_text = _leaflet_html(lon, lat, zoom, detections)
    if not getattr(app, "satellite_web_loaded", False):
        app.satellite_web_loaded = True
        webview.setHtml(html_text, QUrl("https://mustatil.local/"))
        _status(app, f"Zoom {zoom} | WebMap internal ESRI | detections {len(detections)}")
        return
    js = "window.mustatilSetView(%s,%s,%s,%s,%s);" % (
        json.dumps(lon), json.dumps(sat_clamp_lat(lat)), json.dumps(zoom), json.dumps(""), json.dumps(detections)
    )
    try:
        _run_js_on_web_thread(app, js)
    except Exception:
        app.satellite_web_loaded = False
        webview.setHtml(html_text, QUrl("https://mustatil.local/"))
    _status(app, f"Zoom {zoom} | WebMap internal ESRI | detections {len(detections)}")


def refresh_satellite_detections_overlay(app: Any) -> None:
    webview = getattr(app, "satellite_web_view", None)
    if webview is None:
        # Do not create QWebEngineView from a worker thread. The map will be
        # created by the normal GUI refresh path.
        return
    detections = _collect_detections(app)
    try:
        _run_js_on_web_thread(app, "window.mustatilSetDetections(%s);" % json.dumps(detections))
        _status(app, f"WebMap red detection outlines: {len(detections)}")
        try:
            app.log(f"Satellite WebMap red detection outlines: {len(detections)}")
        except Exception:
            pass
    except Exception as exc:
        _log(app, "Satellite detection overlay refresh failed: " + str(exc))


def satellite_redraw_detection_overlay(app):
    """Refresh WebMap detection rectangles from current visible satellite detections."""
    try:
        refresh_satellite_detections_overlay(app)
    except Exception as exc:
        try:
            app.log("Satellite detection overlay refresh failed: " + str(exc))
        except Exception:
            pass

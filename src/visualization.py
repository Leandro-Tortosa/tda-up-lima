"""
Módulo de visualización cartográfica (Pestaña 1) y renderizado multicapa
del Complejo Simplicial con Centralidades Topológicas (Pestaña 2).
"""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import folium
from folium import plugins
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from src.simplicial import SimplicialComplex
from src.hodge_centrality import SimplicialCentralities, HodgeOperators


# Mapas base disponibles
BASEMAP_TILES = {
    "Esri Claro (Light Gray Canvas)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ"
    },
    "Esri Oscuro (Dark Gray Canvas)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ"
    },
    "CartoDB Positron (Fondo claro)": {
        "tiles": "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_2r6b_1_91a5b5d056fb6597047e0812",
        "attr": '&copy; <a href="https://carto.com/">CARTO</a>'
    },
    "CartoDB Dark Matter (Fondo oscuro)": {
        "tiles": "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_2r6b_1_91a5b5d056fb6597047e0812",
        "attr": '&copy; <a href="https://carto.com/">CARTO</a>'
    },
    "OpenStreetMap Estándar": {
        "tiles": "OpenStreetMap",
        "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    },
    "Esri World Imagery (Satélite)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
    },
    "Esri World Topo Map (Topográfico)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community"
    },
    "Lienzo en Blanco (Sin teselas base)": {
        "tiles": None,
        "attr": "White Canvas"
    }
}


def _get_colormap_rgba(palette_name: str, values: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Genera lista de colores hexadecimales normalizados a partir de una paleta matplotlib."""
    cmap_str = palette_name.lower() if palette_name else "plasma"
    try:
        import matplotlib as mpl
        if hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps.get_cmap(cmap_str)
        else:
            cmap = cm.get_cmap(cmap_str)
    except Exception:
        import matplotlib as mpl
        if hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps.get_cmap("plasma")
        else:
            cmap = cm.get_cmap("plasma")
        
    val_clean = np.nan_to_num(values, nan=0.0)
    vmin = float(np.min(val_clean)) if len(val_clean) > 0 else 0.0
    vmax = float(np.max(val_clean)) if len(val_clean) > 0 else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
        
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = [mcolors.to_hex(cmap(norm(v))) for v in val_clean]
    return colors, np.array([vmin, vmax])


def _generate_colormap_gradient_css(palette_name: str, num_stops: int = 10) -> str:
    """Genera la cadena CSS linear-gradient(to right, ...) para una paleta matplotlib."""
    cmap_str = palette_name.lower() if palette_name else "plasma"
    try:
        import matplotlib as mpl
        if hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps.get_cmap(cmap_str)
        else:
            cmap = cm.get_cmap(cmap_str)
    except Exception:
        import matplotlib as mpl
        if hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps.get_cmap("plasma")
        else:
            cmap = cm.get_cmap("plasma")
            
    stops = []
    for i in range(num_stops):
        frac = i / (num_stops - 1)
        hex_col = mcolors.to_hex(cmap(frac))
        stops.append(f"{hex_col} {int(frac * 100)}%")
    return f"linear-gradient(to right, {', '.join(stops)})"


def inject_compact_map_styles(m: folium.Map) -> folium.Map:
    """Inyecta reglas CSS para hacer el cuadro de capas/mapas más pequeño y compacto."""
    compact_css = """
    <style>
    /* Cuadro de capas y mapas más compacto y pequeño */
    .leaflet-control-layers {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 11px !important;
        padding: 5px 8px !important;
        border-radius: 8px !important;
        background: rgba(255, 255, 255, 0.92) !important;
        backdrop-filter: blur(6px) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.14) !important;
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        max-width: 200px !important;
        line-height: 1.25 !important;
    }
    .leaflet-control-layers-base, .leaflet-control-layers-overlays {
        max-height: 200px !important;
        overflow-y: auto !important;
    }
    .leaflet-control-layers-base label, 
    .leaflet-control-layers-overlays label {
        margin-bottom: 2px !important;
        font-size: 10.5px !important;
        font-weight: 500 !important;
        color: #1e293b !important;
        display: flex !important;
        align-items: center !important;
        gap: 4px !important;
        cursor: pointer !important;
    }
    .leaflet-control-layers input[type="radio"],
    .leaflet-control-layers input[type="checkbox"] {
        margin: 0 3px 0 0 !important;
        transform: scale(0.8) !important;
        accent-color: #0284c7 !important;
        cursor: pointer !important;
    }
    .leaflet-control-layers-separator {
        margin: 3px 0 !important;
        border-top: 1px solid #e2e8f0 !important;
    }
    </style>
    """
    m.get_root().header.add_child(folium.Element(compact_css))
    return m


def add_chromatic_scale_legend(
    m: folium.Map,
    colormap_name: str = "Plasma",
    title: str = "Escala de Centralidad",
    vmin_label: str = "0.0 (Baja)",
    vmax_label: str = "1.0 (Alta)"
) -> folium.Map:
    """Añade una leyenda flotante compacta con la escala cromática al mapa Folium."""
    gradient_css = _generate_colormap_gradient_css(colormap_name)
    legend_html = f"""
    <div id="chromatic-scale-legend" style="
        position: absolute;
        bottom: 24px;
        left: 14px;
        z-index: 999;
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(6px);
        padding: 6px 10px;
        border-radius: 8px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
        width: 175px;
        pointer-events: auto;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-size: 10px; font-weight: 700; color: #1e293b;">📊 {title}</span>
            <span style="font-size: 9px; font-weight: 600; color: #64748b; text-transform: uppercase;">{colormap_name}</span>
        </div>
        <div style="
            width: 100%;
            height: 9px;
            border-radius: 4px;
            background: {gradient_css};
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.25);
            margin-bottom: 3px;
        "></div>
        <div style="display: flex; justify-content: space-between; font-size: 9px; font-weight: 600; color: #475569;">
            <span>{vmin_label}</span>
            <span>{vmax_label}</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def create_folium_geographic_map(
    sc: SimplicialComplex,
    centralities: SimplicialCentralities,
    basemap_name: str = "Esri Claro (Light Gray Canvas)",
    colormap_name: str = "Plasma",
    metric_name: str = "degree"
) -> folium.Map:
    """
    Construye el mapa interactivo de Folium para la Pestaña 1 (Geographic Explorer).
    """
    if len(sc.s0_nodes) == 0:
        return folium.Map(location=[38.346254, -0.490899], zoom_start=12)
        
    center_lat = float(np.mean(sc.coords_geographic[:, 0]))
    center_lon = float(np.mean(sc.coords_geographic[:, 1]))
    
    cfg = BASEMAP_TILES.get(basemap_name, BASEMAP_TILES["Esri Claro (Light Gray Canvas)"])
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles=None,
        prefer_canvas=True
    )
    
    # Inyección de CSS para fondo blanco puro cuando no hay teselas o lienzo blanco
    m.get_root().header.add_child(
        folium.Element("<style>.leaflet-container { background-color: #ffffff !important; background: #ffffff !important; }</style>")
    )
    
    if cfg["tiles"] is not None:
        folium.TileLayer(
            tiles=cfg["tiles"],
            attr=cfg["attr"],
            name=basemap_name,
            overlay=False,
            control=False
        ).add_to(m)

    # Valores para colorear nodos
    df_s0 = centralities.df_centrality_s0
    if metric_name == "degree":
        metric_vals = df_s0['degree_k1_edges'].values
        val_label = "Grado (Aristas)"
    elif metric_name == "betweenness":
        metric_vals = df_s0['betweenness_score'].values
        val_label = "Betweenness"
    elif metric_name == "eigenvector":
        metric_vals = df_s0['eigenvector_node'].values
        val_label = "Eigenvector"
    else:
        metric_vals = df_s0['degree_k1_edges'].values
        val_label = "Grado"
        
    colors_hex, (vmin, vmax) = _get_colormap_rgba(colormap_name, metric_vals)
    
    # 1. Dibujar aristas viales (Multi-PolyLine consolidada para alta velocidad de renderizado)
    fg_edges = folium.FeatureGroup(name="Red Vial (Aristas)", show=True)
    edge_locations = [[sc.coords_geographic[u].tolist(), sc.coords_geographic[v].tolist()] for u, v in sc.s1_edges]
    folium.PolyLine(
        locations=edge_locations,
        color="#64748b",
        weight=2.0,
        opacity=0.7
    ).add_to(fg_edges)
    fg_edges.add_to(m)
    
    # 2. Dibujar nodos con popups interactivos
    fg_nodes = folium.FeatureGroup(name="Nodos Urbanos (0-Símplices)", show=True)
    for i, r in df_s0.iterrows():
        lat, lon = float(r['lat']), float(r['lon'])
        color = colors_hex[int(r['simplex_id'])]
        osmid = r['osmid']
        deg_k1 = int(r['degree_k1_edges'])
        deg_k2 = int(r['degree_k2_triangles'])
        m_val = float(metric_vals[int(r['simplex_id'])])
        
        popup_html = f"""
        <div style="font-family: 'Inter', sans-serif; font-size: 13px; width: 220px; line-height: 1.4;">
            <b style="color: #0f172a; font-size: 14px;">Nodo OSM #{osmid}</b><hr style="margin: 4px 0; border: 0.5px solid #cbd5e1;"/>
            <b>ID Canónico:</b> {int(r['simplex_id'])}<br/>
            <b>Coordenadas:</b> ({lat:.5f}, {lon:.5f})<br/>
            <b>Grado k=1 (Calles):</b> {deg_k1}<br/>
            <b>Grado k=2 (Triángulos):</b> {deg_k2}<br/>
            <b style="color: #6366f1;">{val_label}:</b> {m_val:.4f}
        </div>
        """
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="#1e293b",
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            weight=1,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Nodo #{osmid} | {val_label}: {m_val:.4f}"
        ).add_to(fg_nodes)
    fg_nodes.add_to(m)
    
    inject_compact_map_styles(m)
    add_chromatic_scale_legend(
        m,
        colormap_name=colormap_name,
        title=f"Centralidad: {val_label}",
        vmin_label=f"{vmin:.2f} (Baja)",
        vmax_label=f"{vmax:.2f} (Alta)"
    )
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def create_folium_simplicial_map(
    sc: SimplicialComplex,
    centralities: SimplicialCentralities,
    hodge: HodgeOperators,
    basemap_name: str = "Lienzo en Blanco",
    colormap_name: str = "Plasma",
    metric_type: str = "degree",  # "degree", "betweenness", "eigenvector"
    show_k0: bool = True,
    show_k1: bool = True,
    show_k2: bool = True
) -> folium.Map:
    """
    Construye el mapa multicapa interactivo de la Pestaña 2:
    - Soporta capas base: Lienzo Blanco, Esri Claro, CartoDB Positron, CartoDB Dark Matter.
    - Soporta capas independientes activables/desactivables para k=0 (nodos), k=1 (aristas) y k=2 (triángulos).
    - Cada dimensión se colorea y parametriza según la métrica simplicial seleccionada.
    """
    if len(sc.s0_nodes) == 0:
        return folium.Map(location=[38.346254, -0.490899], zoom_start=12)
        
    center_lat = float(np.mean(sc.coords_geographic[:, 0]))
    center_lon = float(np.mean(sc.coords_geographic[:, 1]))
    
    # 1. Configurar mapa base principal y capas seleccionables
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles=None,
        prefer_canvas=True
    )
    
    # Inyección de CSS para fondo blanco puro cuando no hay teselas o lienzo blanco
    m.get_root().header.add_child(
        folium.Element("<style>.leaflet-container { background-color: #ffffff !important; background: #ffffff !important; }</style>")
    )
    
    # Capas base disponibles como opciones de radio en el LayerControl del mapa
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="🗺️ Base: Esri Claro",
        overlay=False,
        control=True,
        show=("Esri Claro" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_2r6b_1_91a5b5d056fb6597047e0812",
        attr='&copy; <a href="https://carto.com/">CARTO</a>',
        name="🗺️ Base: CartoDB Positron",
        overlay=False,
        control=True,
        show=("CartoDB Positron" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_2r6b_1_91a5b5d056fb6597047e0812",
        attr='&copy; <a href="https://carto.com/">CARTO</a>',
        name="🗺️ Base: CartoDB Dark Matter",
        overlay=False,
        control=True,
        show=("Dark" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="OpenStreetMap",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        name="🗺️ Base: OpenStreetMap",
        overlay=False,
        control=True,
        show=("OpenStreetMap" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri Satellite",
        name="🗺️ Base: Esri Satélite",
        overlay=False,
        control=True,
        show=("Satélite" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=",
        attr="Lienzo Blanco Neutro",
        name="⬜ Base: Lienzo en Blanco",
        overlay=False,
        control=True,
        show=("Lienzo" in basemap_name)
    ).add_to(m)

    # 2. CAPA 2-SÍMPLICES (k=2: Triángulos / Caras)
    df_s2 = centralities.df_centrality_s2
    fg_k2 = folium.FeatureGroup(name="🔺 Capa Centralidad k=2 (Triángulos / Caras)", show=show_k2, overlay=True)
    
    if len(sc.s2_triangles) > 0 and len(df_s2) > 0:
        if metric_type == "degree":
            s2_vals = df_s2['degree_adjacent'].values
            s2_label = "Grado Adyacente k=2"
        elif metric_type == "betweenness":
            s2_vals = df_s2['betweenness_score'].values
            s2_label = "Betweenness Simplicial k=2"
        else:
            s2_vals = df_s2['eigenvector_score'].values
            s2_label = "Eigenvector Simplicial k=2"
            
        s2_colors, _ = _get_colormap_rgba(colormap_name, s2_vals)
        
        for idx_t, (u, v, w) in enumerate(sc.s2_triangles):
            lat_u, lon_u = sc.coords_geographic[u]
            lat_v, lon_v = sc.coords_geographic[v]
            lat_w, lon_w = sc.coords_geographic[w]
            color_t = s2_colors[idx_t] if idx_t < len(s2_colors) else "#94a3b8"
            score_t = float(s2_vals[idx_t]) if idx_t < len(s2_vals) else 0.0
            r_row = df_s2.iloc[idx_t] if idx_t < len(df_s2) else None
            area_m2 = float(r_row['area_m2']) if r_row is not None and 'area_m2' in r_row else 0.0
            circ_m = float(r_row['circumradius']) if r_row is not None and 'circumradius' in r_row else 0.0
            
            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; font-size: 13px; width: 230px; line-height: 1.4;">
                <b style="color: #4338ca; font-size: 14px;">🔺 Triángulo (2-Símplex) #{idx_t}</b><hr style="margin: 4px 0; border: 0.5px solid #cbd5e1;"/>
                <b>Vértices OSM:</b> [{sc.idx_to_node[u]}, {sc.idx_to_node[v]}, {sc.idx_to_node[w]}]<br/>
                <b style="color: #6366f1;">{s2_label}:</b> {score_t:.5f}<br/>
                <b>Área 2D:</b> {area_m2:.1f} m²<br/>
                <b>Circunradio (R):</b> {circ_m:.1f} m
            </div>
            """
            folium.Polygon(
                locations=[[lat_u, lon_u], [lat_v, lon_v], [lat_w, lon_w]],
                color=color_t,
                weight=1.5,
                fill=True,
                fill_color=color_t,
                fill_opacity=0.55,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"2-Símplex #{idx_t} | {s2_label}: {score_t:.4f}"
            ).add_to(fg_k2)
    fg_k2.add_to(m)
    
    # 3. CAPA 1-SÍMPLICES (k=1: Aristas / Calles)
    df_s1 = centralities.df_centrality_s1
    fg_k1 = folium.FeatureGroup(name="➖ Capa Centralidad k=1 (Aristas / Calles)", show=show_k1, overlay=True)
    
    if len(sc.s1_edges) > 0 and len(df_s1) > 0:
        if metric_type == "degree":
            s1_vals = df_s1['degree_upper'].values
            s1_label = "Upper Degree (k=2 Co-caras)"
        elif metric_type == "betweenness":
            s1_vals = df_s1['betweenness_score'].values
            s1_label = "Betweenness de Aristas (L1)"
        else:
            s1_vals = df_s1['eigenvector_total'].values
            s1_label = "Eigenvector Total (L1)"
            
        s1_colors, _ = _get_colormap_rgba(colormap_name, s1_vals)
        
        for idx_e, (u, v) in enumerate(sc.s1_edges):
            lat_u, lon_u = sc.coords_geographic[u]
            lat_v, lon_v = sc.coords_geographic[v]
            color_e = s1_colors[idx_e] if idx_e < len(s1_colors) else "#475569"
            score_e = float(s1_vals[idx_e]) if idx_e < len(s1_vals) else 0.0
            r_e = df_s1.iloc[idx_e] if idx_e < len(df_s1) else None
            length_m = float(r_e['length_m']) if r_e is not None and 'length_m' in r_e else 0.0
            upper_deg = int(r_e['degree_upper']) if r_e is not None and 'degree_upper' in r_e else 0
            
            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; font-size: 13px; width: 220px; line-height: 1.4;">
                <b style="color: #0f172a; font-size: 14px;">➖ Arista (1-Símplex) #{idx_e}</b><hr style="margin: 4px 0; border: 0.5px solid #cbd5e1;"/>
                <b>Vértices OSM:</b> [{sc.idx_to_node[u]}, {sc.idx_to_node[v]}]<br/>
                <b style="color: #6366f1;">{s1_label}:</b> {score_e:.5f}<br/>
                <b>Longitud:</b> {length_m:.1f} m<br/>
                <b>Grado Superior (Caras):</b> {upper_deg}
            </div>
            """
            folium.PolyLine(
                locations=[[lat_u, lon_u], [lat_v, lon_v]],
                color=color_e,
                weight=3.5,
                opacity=0.85,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"1-Símplex #{idx_e} | {s1_label}: {score_e:.4f}"
            ).add_to(fg_k1)
    fg_k1.add_to(m)
    
    # 4. CAPA 0-SÍMPLICES (k=0: Nodos / Intersecciones)
    df_s0 = centralities.df_centrality_s0
    fg_k0 = folium.FeatureGroup(name="🔘 Capa Centralidad k=0 (Nodos / Intersecciones)", show=show_k0, overlay=True)
    
    if len(sc.s0_nodes) > 0 and len(df_s0) > 0:
        if metric_type == "degree":
            s0_vals = df_s0['degree_k1_edges'].values
            s0_label = "Simplicial Degree (k=1 Calles)"
        elif metric_type == "betweenness":
            s0_vals = df_s0['betweenness_score'].values
            s0_label = "Betweenness Nodos"
        else:
            s0_vals = df_s0['eigenvector_node'].values
            s0_label = "Eigenvector Nodos"
            
        s0_colors, _ = _get_colormap_rgba(colormap_name, s0_vals)
        
        for i, r in df_s0.iterrows():
            lat, lon = float(r['lat']), float(r['lon'])
            color_n = s0_colors[int(r['simplex_id'])]
            osmid = r['osmid']
            score_n = float(s0_vals[int(r['simplex_id'])])
            deg_k1 = int(r['degree_k1_edges'])
            deg_k2 = int(r['degree_k2_triangles'])
            
            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; font-size: 13px; width: 220px; line-height: 1.4;">
                <b style="color: #0f172a; font-size: 14px;">🔘 Nodo (0-Símplex) #{int(r['simplex_id'])}</b><hr style="margin: 4px 0; border: 0.5px solid #cbd5e1;"/>
                <b>OSM ID:</b> {osmid}<br/>
                <b>Coordenadas:</b> ({lat:.5f}, {lon:.5f})<br/>
                <b>Grado Calles (k=1):</b> {deg_k1}<br/>
                <b>Grado Triángulos (k=2):</b> {deg_k2}<br/>
                <b style="color: #6366f1;">{s0_label}:</b> {score_n:.5f}
            </div>
            """
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color="#0f172a",
                fill=True,
                fill_color=color_n,
                fill_opacity=0.95,
                weight=1.5,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"0-Símplex #{int(r['simplex_id'])} | {s0_label}: {score_n:.4f}"
            ).add_to(fg_k0)
    fg_k0.add_to(m)

    # Control de Capas interactivo y leyenda cromática
    inject_compact_map_styles(m)
    m_label = "Grado" if metric_type == "degree" else ("Betweenness" if metric_type == "betweenness" else "Eigenvector")
    add_chromatic_scale_legend(
        m,
        colormap_name=colormap_name,
        title=f"Escala: {m_label}",
        vmin_label="0.0 (Baja)",
        vmax_label="1.0 (Alta)"
    )
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def create_white_canvas_simplicial_figure(
    sc: SimplicialComplex,
    centralities: SimplicialCentralities,
    hodge: HodgeOperators,
    metric_type: str = "degree",  # "degree", "betweenness", "eigenvector"
    dimension_k: int = 1,         # 0 (nodos), 1 (aristas), 2 (triángulos)
    colormap_name: str = "Plasma"
) -> go.Figure:
    """
    Construye la visualización matemática interactiva sobre LIENZO EN BLANCO puro
    utilizando Plotly, con colores mapeados a la centralidad simplicial elegida.
    """
    fig = go.Figure()
    
    # Obtener valores y nombres según la métrica seleccionada
    if dimension_k == 0:
        # Nodos
        df = centralities.df_centrality_s0
        if metric_type == "degree":
            vals = df['degree_k1_edges'].values
            title_metric = "Simplicial Degree k=1 (Nodos)"
        elif metric_type == "betweenness":
            vals = df['betweenness_score'].values
            title_metric = "Betweenness Centrality (Nodos)"
        else:
            vals = df['eigenvector_node'].values
            title_metric = "Eigenvector Centrality (Nodos)"
    elif dimension_k == 1:
        # Aristas
        df = centralities.df_centrality_s1
        if metric_type == "degree":
            vals = df['degree_upper'].values
            title_metric = "Upper Degree k=2 (Aristas)"
        elif metric_type == "betweenness":
            vals = df['betweenness_score'].values
            title_metric = "Simplicial Edge Betweenness"
        else:
            vals = df['eigenvector_total'].values
            title_metric = "Simplicial Eigenvector (Total Hodge)"
    else:
        # Triángulos (k=2)
        df = centralities.df_centrality_s2
        if len(df) > 0:
            if metric_type == "degree":
                vals = df['degree_adjacent'].values
                title_metric = "Triangle Adjacency Degree"
            elif metric_type == "betweenness":
                vals = df['betweenness_score'].values
                title_metric = "Simplicial Triangle Betweenness"
            else:
                vals = df['eigenvector_score'].values
                title_metric = "Triangle Eigenvector Centrality"
        else:
            vals = np.array([])
            title_metric = "No 2-simplices"

    # Paleta de color normalizada
    colors_hex, (vmin, vmax) = _get_colormap_rgba(colormap_name, vals)
    
    # 1. RENDERIZAR 2-SÍMPLICES (Triángulos)
    if len(sc.s2_triangles) > 0:
        for idx_t, (u, v, w) in enumerate(sc.s2_triangles):
            p_u, p_v, p_w = sc.coords_projected[u], sc.coords_projected[v], sc.coords_projected[w]
            
            if dimension_k == 2 and len(colors_hex) > idx_t:
                fill_color = colors_hex[idx_t]
                m_score = vals[idx_t] if idx_t < len(vals) else 0.0
                hover_txt = (
                    f"<b>Triángulo (2-símplex) #{idx_t}</b><br>"
                    f"Vértices: [{sc.idx_to_node[u]}, {sc.idx_to_node[v]}, {sc.idx_to_node[w]}]<br>"
                    f"{title_metric}: {m_score:.5f}<br>"
                    f"Área: {_triangle_area_2d_approx(p_u, p_v, p_w):.1f} m²"
                )
            else:
                fill_color = "rgba(148, 163, 184, 0.25)"
                hover_txt = f"Triángulo #{idx_t} | [{sc.idx_to_node[u]}, {sc.idx_to_node[v]}, {sc.idx_to_node[w]}]"
                
            fig.add_trace(go.Scatter(
                x=[p_u[0], p_v[0], p_w[0], p_u[0]],
                y=[p_u[1], p_v[1], p_w[1], p_u[1]],
                fill="toself",
                fillcolor=fill_color,
                line=dict(color="rgba(100, 116, 139, 0.3)", width=0.5),
                mode="lines",
                hoverinfo="text",
                text=hover_txt,
                showlegend=False
            ))
            
    # 2. RENDERIZAR 1-SÍMPLICES (Aristas)
    if len(sc.s1_edges) > 0:
        if dimension_k == 1:
            for idx_e, (u, v) in enumerate(sc.s1_edges):
                p_u, p_v = sc.coords_projected[u], sc.coords_projected[v]
                edge_col = colors_hex[idx_e] if idx_e < len(colors_hex) else "#64748b"
                m_score = vals[idx_e] if idx_e < len(vals) else 0.0
                length_m = np.linalg.norm(p_u - p_v)
                
                hover_txt = (
                    f"<b>Arista (1-símplex) #{idx_e}</b><br>"
                    f"Vértices: [{sc.idx_to_node[u]}, {sc.idx_to_node[v]}]<br>"
                    f"{title_metric}: {m_score:.5f}<br>"
                    f"Longitud: {length_m:.1f} m"
                )
                
                fig.add_trace(go.Scatter(
                    x=[p_u[0], p_v[0], None],
                    y=[p_u[1], p_v[1], None],
                    mode="lines",
                    line=dict(color=edge_col, width=2.5),
                    hoverinfo="text",
                    text=hover_txt,
                    showlegend=False
                ))
        else:
            edge_x, edge_y = [], []
            for u, v in sc.s1_edges:
                p_u, p_v = sc.coords_projected[u], sc.coords_projected[v]
                edge_x.extend([p_u[0], p_v[0], None])
                edge_y.extend([p_u[1], p_v[1], None])
                
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="#cbd5e1", width=1.0),
                hoverinfo="none",
                showlegend=False
            ))
            
    # 3. RENDERIZAR 0-SÍMPLICES (Nodos)
    if len(sc.s0_nodes) > 0:
        node_x = sc.coords_projected[:, 0]
        node_y = sc.coords_projected[:, 1]
        
        if dimension_k == 0:
            node_colors = colors_hex
            node_sizes = 5 + 7 * (vals / (vmax if vmax > 0 else 1.0))
            hover_nodes = [
                f"<b>Nodo (0-símplex) #{i}</b><br>"
                f"OSM ID: {sc.idx_to_node[i]}<br>"
                f"{title_metric}: {vals[i]:.5f}<br>"
                f"Grado k=1: {int(centralities.df_centrality_s0['degree_k1_edges'].iloc[i])}"
                for i in range(len(sc.s0_nodes))
            ]
        else:
            node_colors = "#334155"
            node_sizes = 3.5
            hover_nodes = [
                f"Nodo #{i} | OSM ID: {sc.idx_to_node[i]}"
                for i in range(len(sc.s0_nodes))
            ]
            
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=0.5, color="#0f172a")
            ),
            hoverinfo="text",
            text=hover_nodes,
            showlegend=False
        ))

    # Trazado invisible para la Barra de Color (Colorbar)
    cscale_map = {
        "plasma": "Plasma",
        "viridis": "Viridis",
        "inferno": "Inferno",
        "magma": "Magma",
        "turbo": "Turbo",
        "cividis": "Cividis"
    }
    plotly_cscale = cscale_map.get(colormap_name.lower(), "Plasma")
    
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale=plotly_cscale,
            cmin=vmin,
            cmax=vmax,
            color=[vmin, vmax],
            colorbar=dict(
                title=dict(text=title_metric, font=dict(size=12, family="Inter, sans-serif")),
                tickfont=dict(size=10, family="Inter, sans-serif"),
                thickness=14,
                len=0.75,
                outlinewidth=0
            ),
            showscale=True
        ),
        hoverinfo="none",
        showlegend=False
    ))

    # Layout de Lienzo en Blanco puro (Altura 1000)
    fig.update_layout(
        height=1000,
        autosize=True,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            showticklabels=False,
            title="Coordenada X (metros proyectados UTM)",
            title_font=dict(size=11, color="#64748b")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            showticklabels=False,
            title="Coordenada Y (metros proyectados UTM)",
            title_font=dict(size=11, color="#64748b"),
            scaleanchor="x",
            scaleratio=1
        ),
        dragmode="pan"
    )
    
    return fig


def _triangle_area_2d_approx(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    return 0.5 * abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))


def create_filtration_histogram_figure(
    values: np.ndarray,
    metric_name: str,
    threshold: float,
    dimension_label: str = "k=0 (Nodos)",
    colormap_name: str = "Plasma"
) -> go.Figure:
    """
    Genera un histograma interactivo en Plotly mostrando la distribución de frecuencias
    de la centralidad seleccionada y la línea de corte del umbral de filtración.
    """
    val_clean = np.nan_to_num(values, nan=0.0)
    if len(val_clean) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No hay datos disponibles para esta dimensión", showarrow=False)
        fig.update_layout(height=320, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
        return fig
        
    vmin, vmax = float(np.min(val_clean)), float(np.max(val_clean))
    count_pass = int(np.sum(val_clean >= threshold))
    pct_pass = (count_pass / len(val_clean)) * 100.0 if len(val_clean) > 0 else 0.0
    
    # Colores para el histograma
    cscale_map = {
        "plasma": "Plasma",
        "viridis": "Viridis",
        "inferno": "Inferno",
        "magma": "Magma",
        "turbo": "Turbo",
        "cividis": "Cividis"
    }
    plotly_cscale = cscale_map.get(colormap_name.lower(), "Plasma")

    fig = go.Figure()
    
    # Histograma con nbins
    fig.add_trace(go.Histogram(
        x=val_clean,
        nbinsx=30,
        marker=dict(
            color=val_clean,
            colorscale=plotly_cscale,
            line=dict(color="#1e293b", width=0.5)
        ),
        name="Distribución de Frecuencia",
        hovertemplate="Rango de Centralidad: %{x}<br>Frecuencia: %{y} elementos<extra></extra>"
    ))
    
    # Línea vertical del umbral de filtración
    fig.add_vline(
        x=threshold,
        line_width=2.5,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Umbral: {threshold:.4f} ({count_pass} elem. | {pct_pass:.1f}%)",
        annotation_position="top right",
        annotation_font=dict(color="#ef4444", size=11, family="Inter, sans-serif")
    )
    
    fig.update_layout(
        title=dict(
            text=f"Distribución de {metric_name} [{dimension_label}]",
            font=dict(size=14, family="Inter, sans-serif", color="#0f172a")
        ),
        height=320,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(
            title=dict(text=f"Valor de Centralidad ({metric_name})", font=dict(size=11, color="#64748b")),
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Cantidad de Símplices", font=dict(size=11, color="#64748b")),
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False
        ),
        showlegend=False
    )
    
    return fig


def create_folium_filtered_simplicial_map(
    sc: SimplicialComplex,
    centralities: SimplicialCentralities,
    hodge: HodgeOperators,
    metric_type: str = "degree",
    threshold_k0: float = 0.0,
    threshold_k1: float = 0.0,
    threshold_k2: float = 0.0,
    highlight_mode: str = "Ocultar descartados",  # "Ocultar descartados" o "Atenuar descartados"
    show_k0: bool = True,
    show_k1: bool = True,
    show_k2: bool = True,
    basemap_name: str = "Lienzo en Blanco",
    colormap_name: str = "Plasma",
    highlighted_simplex: Optional[Dict[str, Any]] = None
) -> folium.Map:
    """
    Construye el mapa interactivo para la Pestaña 3 (Filtración Topológica):
    - Permite filtrar independientemente por valor de centralidad en k=0, k=1 y k=2.
    - Soporta modo 'Ocultar descartados' o 'Atenuar descartados' en gris.
    - Soporta fondos: Lienzo en Blanco, Esri Claro, CartoDB Positron, etc.
    - Resalta prominentemente en pantalla el nodo, arista o triángulo seleccionado del Top.
    """
    if len(sc.s0_nodes) == 0:
        return folium.Map(location=[38.346254, -0.490899], zoom_start=12)
        
    center_lat = float(np.mean(sc.coords_geographic[:, 0]))
    center_lon = float(np.mean(sc.coords_geographic[:, 1]))
    zoom_start = 15
    
    # Si hay un elemento seleccionado para resaltar, centramos y enfocamos la cámara
    if highlighted_simplex is not None:
        h_dim = highlighted_simplex.get("dim", None)
        h_id = highlighted_simplex.get("id", None)
        if h_dim == 0 and h_id is not None and 0 <= h_id < len(sc.s0_nodes):
            center_lat, center_lon = float(sc.coords_geographic[h_id, 0]), float(sc.coords_geographic[h_id, 1])
            zoom_start = 17
        elif h_dim == 1 and h_id is not None and 0 <= h_id < len(sc.s1_edges):
            u, v = sc.s1_edges[h_id]
            lat1, lon1 = sc.coords_geographic[u]
            lat2, lon2 = sc.coords_geographic[v]
            center_lat, center_lon = float((lat1 + lat2) / 2.0), float((lon1 + lon2) / 2.0)
            zoom_start = 17
        elif h_dim == 2 and h_id is not None and 0 <= h_id < len(sc.s2_triangles):
            u, v, w = sc.s2_triangles[h_id]
            lat_u, lon_u = sc.coords_geographic[u]
            lat_v, lon_v = sc.coords_geographic[v]
            lat_w, lon_w = sc.coords_geographic[w]
            center_lat, center_lon = float((lat_u + lat_v + lat_w) / 3.0), float((lon_u + lon_v + lon_w) / 3.0)
            zoom_start = 17
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=None,
        prefer_canvas=True
    )
    
    # Inyección de CSS para fondo blanco puro cuando no hay teselas o lienzo blanco
    m.get_root().header.add_child(
        folium.Element("<style>.leaflet-container { background-color: #ffffff !important; background: #ffffff !important; }</style>")
    )
    
    # Capas base disponibles
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="🗺️ Base: Esri Claro",
        overlay=False,
        control=True,
        show=("Esri Claro" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_2r6b_1_91a5b5d056fb6597047e0812",
        attr='&copy; <a href="https://carto.com/">CARTO</a>',
        name="🗺️ Base: CartoDB Positron",
        overlay=False,
        control=True,
        show=("CartoDB Positron" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_2r6b_1_91a5b5d056fb6597047e0812",
        attr='&copy; <a href="https://carto.com/">CARTO</a>',
        name="🗺️ Base: CartoDB Dark Matter",
        overlay=False,
        control=True,
        show=("Dark" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="OpenStreetMap",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        name="🗺️ Base: OpenStreetMap",
        overlay=False,
        control=True,
        show=("OpenStreetMap" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=",
        attr="Lienzo Blanco Neutro",
        name="⬜ Base: Lienzo en Blanco",
        overlay=False,
        control=True,
        show=("Lienzo" in basemap_name)
    ).add_to(m)

    # 1. FILTRACIÓN DE 2-SÍMPLICES (k=2: Triángulos)
    df_s2 = centralities.df_centrality_s2
    fg_k2_filt = folium.FeatureGroup(name="🔺 2-Símplices Filtrados (k=2)", show=show_k2, overlay=True)
    
    if len(sc.s2_triangles) > 0 and len(df_s2) > 0:
        if metric_type == "degree":
            s2_vals = df_s2['degree_adjacent'].values
            s2_label = "Degree k=2"
        elif metric_type == "betweenness":
            s2_vals = df_s2['betweenness_score'].values
            s2_label = "Betweenness k=2"
        else:
            s2_vals = df_s2['eigenvector_score'].values
            s2_label = "Eigenvector k=2"
            
        s2_colors, _ = _get_colormap_rgba(colormap_name, s2_vals)
        
        for idx_t, (u, v, w) in enumerate(sc.s2_triangles):
            lat_u, lon_u = sc.coords_geographic[u]
            lat_v, lon_v = sc.coords_geographic[v]
            lat_w, lon_w = sc.coords_geographic[w]
            score_t = float(s2_vals[idx_t]) if idx_t < len(s2_vals) else 0.0
            r_row = df_s2.iloc[idx_t] if idx_t < len(df_s2) else None
            area_m2 = float(r_row['area_m2']) if r_row is not None and 'area_m2' in r_row else 0.0
            
            passed = (score_t >= threshold_k2)
            if not passed and highlight_mode == "Ocultar descartados":
                continue
                
            if passed:
                fill_col = s2_colors[idx_t] if idx_t < len(s2_colors) else "#6366f1"
                opacity = 0.65
                weight = 2.0
                status_txt = "✅ Supera Umbral"
            else:
                fill_col = "#cbd5e1"
                opacity = 0.15
                weight = 0.5
                status_txt = "❌ Excluido por Umbral"
                
            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; font-size: 13px; width: 230px; line-height: 1.4;">
                <b style="color: #4338ca; font-size: 14px;">🔺 Triángulo #{idx_t} ({status_txt})</b><hr style="margin: 4px 0; border: 0.5px solid #cbd5e1;"/>
                <b>Vértices:</b> [{sc.idx_to_node[u]}, {sc.idx_to_node[v]}, {sc.idx_to_node[w]}]<br/>
                <b style="color: #6366f1;">{s2_label}:</b> {score_t:.5f}<br/>
                <b>Área:</b> {area_m2:.1f} m²
            </div>
            """
            folium.Polygon(
                locations=[[lat_u, lon_u], [lat_v, lon_v], [lat_w, lon_w]],
                color=fill_col,
                weight=weight,
                fill=True,
                fill_color=fill_col,
                fill_opacity=opacity,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"2-Símplex #{idx_t} | {s2_label}: {score_t:.4f} ({status_txt})"
            ).add_to(fg_k2_filt)
    fg_k2_filt.add_to(m)

    # 2. FILTRACIÓN DE 1-SÍMPLICES (k=1: Aristas)
    df_s1 = centralities.df_centrality_s1
    fg_k1_filt = folium.FeatureGroup(name="➖ 1-Símplices Filtrados (k=1)", show=show_k1, overlay=True)
    
    if len(sc.s1_edges) > 0 and len(df_s1) > 0:
        if metric_type == "degree":
            s1_vals = df_s1['degree_upper'].values
            s1_label = "Upper Degree k=1"
        elif metric_type == "betweenness":
            s1_vals = df_s1['betweenness_score'].values
            s1_label = "Betweenness k=1"
        else:
            s1_vals = df_s1['eigenvector_total'].values
            s1_label = "Eigenvector k=1"
            
        s1_colors, _ = _get_colormap_rgba(colormap_name, s1_vals)
        
        for idx_e, (u, v) in enumerate(sc.s1_edges):
            lat_u, lon_u = sc.coords_geographic[u]
            lat_v, lon_v = sc.coords_geographic[v]
            score_e = float(s1_vals[idx_e]) if idx_e < len(s1_vals) else 0.0
            r_e = df_s1.iloc[idx_e] if idx_e < len(df_s1) else None
            length_m = float(r_e['length_m']) if r_e is not None and 'length_m' in r_e else 0.0
            
            passed = (score_e >= threshold_k1)
            if not passed and highlight_mode == "Ocultar descartados":
                continue
                
            if passed:
                line_col = s1_colors[idx_e] if idx_e < len(s1_colors) else "#0ea5e9"
                opacity = 0.9
                weight = 4.0
                status_txt = "✅ Supera Umbral"
            else:
                line_col = "#cbd5e1"
                opacity = 0.25
                weight = 1.0
                status_txt = "❌ Excluido por Umbral"
                
            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; font-size: 13px; width: 220px; line-height: 1.4;">
                <b style="color: #0f172a; font-size: 14px;">➖ Arista #{idx_e} ({status_txt})</b><hr style="margin: 4px 0; border: 0.5px solid #cbd5e1;"/>
                <b>Vértices:</b> [{sc.idx_to_node[u]}, {sc.idx_to_node[v]}]<br/>
                <b style="color: #6366f1;">{s1_label}:</b> {score_e:.5f}<br/>
                <b>Longitud:</b> {length_m:.1f} m
            </div>
            """
            folium.PolyLine(
                locations=[[lat_u, lon_u], [lat_v, lon_v]],
                color=line_col,
                weight=weight,
                opacity=opacity,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"1-Símplex #{idx_e} | {s1_label}: {score_e:.4f} ({status_txt})"
            ).add_to(fg_k1_filt)
    fg_k1_filt.add_to(m)

    # 3. FILTRACIÓN DE 0-SÍMPLICES (k=0: Nodos)
    df_s0 = centralities.df_centrality_s0
    fg_k0_filt = folium.FeatureGroup(name="🔘 0-Símplices Filtrados (k=0)", show=show_k0, overlay=True)
    
    if len(sc.s0_nodes) > 0 and len(df_s0) > 0:
        if metric_type == "degree":
            s0_vals = df_s0['degree_k1_edges'].values
            s0_label = "Degree k=0"
        elif metric_type == "betweenness":
            s0_vals = df_s0['betweenness_score'].values
            s0_label = "Betweenness k=0"
        else:
            s0_vals = df_s0['eigenvector_node'].values
            s0_label = "Eigenvector k=0"
            
        s0_colors, _ = _get_colormap_rgba(colormap_name, s0_vals)
        
        for i, r in df_s0.iterrows():
            lat, lon = float(r['lat']), float(r['lon'])
            osmid = r['osmid']
            score_n = float(s0_vals[int(r['simplex_id'])])
            
            passed = (score_n >= threshold_k0)
            if not passed and highlight_mode == "Ocultar descartados":
                continue
                
            if passed:
                node_col = s0_colors[int(r['simplex_id'])]
                radius = 6.0
                opacity = 1.0
                border_col = "#0f172a"
                status_txt = "✅ Supera Umbral"
            else:
                node_col = "#e2e8f0"
                radius = 3.0
                opacity = 0.3
                border_col = "#94a3b8"
                status_txt = "❌ Excluido por Umbral"
                
            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; font-size: 13px; width: 220px; line-height: 1.4;">
                <b style="color: #0f172a; font-size: 14px;">🔘 Nodo #{int(r['simplex_id'])} ({status_txt})</b><hr style="margin: 4px 0; border: 0.5px solid #cbd5e1;"/>
                <b>OSM ID:</b> {osmid}<br/>
                <b>Coordenadas:</b> ({lat:.5f}, {lon:.5f})<br/>
                <b style="color: #6366f1;">{s0_label}:</b> {score_n:.5f}
            </div>
            """
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=border_col,
                fill=True,
                fill_color=node_col,
                fill_opacity=opacity,
                weight=1.5,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"0-Símplex #{int(r['simplex_id'])} | {s0_label}: {score_n:.4f} ({status_txt})"
            ).add_to(fg_k0_filt)
    fg_k0_filt.add_to(m)

    # 4. CAPA ESPECIAL DE RESALTADO DE SÍMPLICE SELECCIONADO (CLICK EN TOP RANKING)
    if highlighted_simplex is not None:
        h_dim = highlighted_simplex.get("dim", None)
        h_id = highlighted_simplex.get("id", None)
        fg_highlight = folium.FeatureGroup(name="🌟 Símplice Resaltado (Top)", show=True, overlay=True)
        
        if h_dim == 0 and h_id is not None and 0 <= h_id < len(sc.s0_nodes):
            lat_n, lon_n = float(sc.coords_geographic[h_id, 0]), float(sc.coords_geographic[h_id, 1])
            osmid_n = sc.idx_to_node[h_id]
            # Halo exterior brillante
            folium.CircleMarker(
                location=[lat_n, lon_n],
                radius=18,
                color="#f43f5e",
                weight=3.5,
                fill=True,
                fill_color="#f43f5e",
                fill_opacity=0.35,
                tooltip=f"🌟 SELECCIONADO: Nodo #{h_id} (OSM {osmid_n})"
            ).add_to(fg_highlight)
            
            # Marcador central
            folium.CircleMarker(
                location=[lat_n, lon_n],
                radius=9,
                color="#ffffff",
                weight=2.5,
                fill=True,
                fill_color="#e11d48",
                fill_opacity=1.0,
                popup=folium.Popup(f"""
                <div style="font-family: 'Inter', sans-serif; font-size: 13px; width: 220px; line-height: 1.4;">
                    <b style="color: #e11d48; font-size: 14px;">🌟 Nodo Top Seleccionado #{h_id}</b><hr style="margin: 4px 0; border: 0.5px solid #cbd5e1;"/>
                    <b>OSM ID:</b> {osmid_n}<br/>
                    <b>Coordenadas:</b> ({lat_n:.5f}, {lon_n:.5f})
                </div>
                """, max_width=250),
                tooltip=f"🌟 Nodo Top #{h_id}"
            ).add_to(fg_highlight)
            
        elif h_dim == 1 and h_id is not None and 0 <= h_id < len(sc.s1_edges):
            u, v = sc.s1_edges[h_id]
            lat1, lon1 = float(sc.coords_geographic[u, 0]), float(sc.coords_geographic[u, 1])
            lat2, lon2 = float(sc.coords_geographic[v, 0]), float(sc.coords_geographic[v, 1])
            
            # Halo exterior grueso
            folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color="#f43f5e",
                weight=12,
                opacity=0.45
            ).add_to(fg_highlight)
            
            # Línea central destacada
            folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color="#e11d48",
                weight=6.0,
                opacity=1.0,
                popup=folium.Popup(f"""
                <div style="font-family: 'Inter', sans-serif; font-size: 13px; width: 220px; line-height: 1.4;">
                    <b style="color: #e11d48; font-size: 14px;">🌟 Arista Top Seleccionada #{h_id}</b><hr style="margin: 4px 0; border: 0.5px solid #cbd5e1;"/>
                    <b>Vértices OSM:</b> [{sc.idx_to_node[u]}, {sc.idx_to_node[v]}]
                </div>
                """, max_width=250),
                tooltip=f"🌟 Arista Top #{h_id}"
            ).add_to(fg_highlight)
            
            # Extremos
            for lu, lo in [[lat1, lon1], [lat2, lon2]]:
                folium.CircleMarker(
                    location=[lu, lo],
                    radius=5.5,
                    color="#ffffff",
                    fill=True,
                    fill_color="#e11d48",
                    fill_opacity=1.0,
                    weight=2.0
                ).add_to(fg_highlight)
                
        elif h_dim == 2 and h_id is not None and 0 <= h_id < len(sc.s2_triangles):
            u, v, w = sc.s2_triangles[h_id]
            lat_u, lon_u = float(sc.coords_geographic[u, 0]), float(sc.coords_geographic[u, 1])
            lat_v, lon_v = float(sc.coords_geographic[v, 0]), float(sc.coords_geographic[v, 1])
            lat_w, lon_w = float(sc.coords_geographic[w, 0]), float(sc.coords_geographic[w, 1])
            
            # Polígono destacado
            folium.Polygon(
                locations=[[lat_u, lon_u], [lat_v, lon_v], [lat_w, lon_w]],
                color="#e11d48",
                weight=4.5,
                fill=True,
                fill_color="#f43f5e",
                fill_opacity=0.75,
                popup=folium.Popup(f"""
                <div style="font-family: 'Inter', sans-serif; font-size: 13px; width: 230px; line-height: 1.4;">
                    <b style="color: #e11d48; font-size: 14px;">🌟 Triángulo Top Seleccionado #{h_id}</b><hr style="margin: 4px 0; border: 0.5px solid #cbd5e1;"/>
                    <b>Vértices OSM:</b> [{sc.idx_to_node[u]}, {sc.idx_to_node[v]}, {sc.idx_to_node[w]}]
                </div>
                """, max_width=260),
                tooltip=f"🌟 Triángulo Top #{h_id}"
            ).add_to(fg_highlight)
            
            # Vértices del triángulo
            for lu, lo in [[lat_u, lon_u], [lat_v, lon_v], [lat_w, lon_w]]:
                folium.CircleMarker(
                    location=[lu, lo],
                    radius=5.5,
                    color="#ffffff",
                    fill=True,
                    fill_color="#e11d48",
                    fill_opacity=1.0,
                    weight=2.0
                ).add_to(fg_highlight)
                
        fg_highlight.add_to(m)

    inject_compact_map_styles(m)
    m_label = "Grado" if metric_type == "degree" else ("Betweenness" if metric_type == "betweenness" else "Eigenvector")
    add_chromatic_scale_legend(
        m,
        colormap_name=colormap_name,
        title=f"Filtración: {m_label}",
        vmin_label="0.0 (Baja)",
        vmax_label="1.0 (Alta)"
    )
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def create_folium_top_simplices_map(
    sc: SimplicialComplex,
    centralities: SimplicialCentralities,
    hodge: HodgeOperators,
    metric_type: str = "degree",  # "degree", "betweenness", "eigenvector"
    show_k0: bool = True,
    show_k1: bool = True,
    show_k2: bool = True,
    top_k0: int = 10,
    top_k1: int = 10,
    top_k2: int = 10,
    basemap_name: str = "Esri Claro (Light Gray Canvas)",
    colormap_name: str = "Plasma"
) -> folium.Map:
    """
    Construye un mapa exclusivo, limpio e interactivo para visualizar ÚNICAMENTE
    los símplices (nodos, aristas y triángulos) de mayor centralidad (Top-K).
    Sin representar el resto de elementos no seleccionados de la red.
    """
    if len(sc.s0_nodes) == 0:
        return folium.Map(location=[38.346254, -0.490899], zoom_start=12)
        
    center_lat = float(np.mean(sc.coords_geographic[:, 0]))
    center_lon = float(np.mean(sc.coords_geographic[:, 1]))
    
    # 1. Configurar mapa base principal y capas base seleccionables
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles=None,
        prefer_canvas=True
    )
    
    m.get_root().header.add_child(
        folium.Element("<style>.leaflet-container { background-color: #ffffff !important; background: #ffffff !important; }</style>")
    )
    
    # Capas base disponibles
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="🗺️ Base: Esri Claro",
        overlay=False,
        control=True,
        show=("Esri Claro" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_2r6b_1_91a5b5d056fb6597047e0812",
        attr='&copy; <a href="https://carto.com/">CARTO</a>',
        name="🗺️ Base: CartoDB Positron",
        overlay=False,
        control=True,
        show=("CartoDB Positron" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_2r6b_1_91a5b5d056fb6597047e0812",
        attr='&copy; <a href="https://carto.com/">CARTO</a>',
        name="🗺️ Base: CartoDB Dark Matter",
        overlay=False,
        control=True,
        show=("Dark" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="OpenStreetMap",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        name="🗺️ Base: OpenStreetMap",
        overlay=False,
        control=True,
        show=("OpenStreetMap" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri Satellite",
        name="🗺️ Base: Esri Satélite",
        overlay=False,
        control=True,
        show=("Satélite" in basemap_name or "Imagery" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri Topo",
        name="🗺️ Base: Esri Topo",
        overlay=False,
        control=True,
        show=("Topo" in basemap_name)
    ).add_to(m)
    
    folium.TileLayer(
        tiles="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=",
        attr="Lienzo Blanco Neutro",
        name="⬜ Base: Lienzo en Blanco",
        overlay=False,
        control=True,
        show=("Lienzo" in basemap_name)
    ).add_to(m)

    # Identificar métricas
    if metric_type == "degree":
        col_s0, col_s1, col_s2 = 'degree_k1_edges', 'degree_upper', 'degree_adjacent'
        m_name = "Grado Simplicial"
    elif metric_type == "betweenness":
        col_s0, col_s1, col_s2 = 'betweenness_score', 'betweenness_score', 'betweenness_score'
        m_name = "Betweenness Simplicial"
    else:
        col_s0, col_s1, col_s2 = 'eigenvector_node', 'eigenvector_total', 'eigenvector_score'
        m_name = "Eigenvector Simplicial"

    # 2. CAPA TOP TRIÁNGULOS (k=2)
    df_s2 = centralities.df_centrality_s2
    if show_k2 and top_k2 > 0 and len(sc.s2_triangles) > 0 and len(df_s2) > 0:
        actual_k2 = min(top_k2, len(df_s2))
        top_tri = df_s2.sort_values(by=col_s2, ascending=False).head(actual_k2).copy()
        top_tri['rank'] = range(1, len(top_tri) + 1)
        tri_scores = top_tri[col_s2].values
        tri_colors, _ = _get_colormap_rgba(colormap_name, tri_scores)
        
        fg_k2 = folium.FeatureGroup(name=f"🔺 Top {actual_k2} Triángulos (k=2)", show=True, overlay=True)
        for idx_row, (_, row) in enumerate(top_tri.iterrows()):
            s_id = int(row['simplex_id'])
            u, v, w = sc.s2_triangles[s_id]
            lat_u, lon_u = sc.coords_geographic[u]
            lat_v, lon_v = sc.coords_geographic[v]
            lat_w, lon_w = sc.coords_geographic[w]
            color_t = tri_colors[idx_row]
            score_t = float(row[col_s2])
            rank_t = int(row['rank'])
            area_m2 = float(row.get('area_m2', 0.0))
            circ_m = float(row.get('circumradius', 0.0))
            
            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; font-size: 13px; width: 240px; line-height: 1.4;">
                <b style="color: #4338ca; font-size: 14px;">🔺 Triángulo Top #{rank_t} (k=2)</b>
                <hr style="margin: 4px 0; border: 0.5px solid #cbd5e1;"/>
                <b>ID Símplex:</b> #{s_id}<br/>
                <b>Vértices OSM:</b> [{sc.idx_to_node[u]}, {sc.idx_to_node[v]}, {sc.idx_to_node[w]}]<br/>
                <b style="color: #6366f1;">{m_name}:</b> {score_t:.5f}<br/>
                <b>Área:</b> {area_m2:.1f} m² | <b>Circunradio:</b> {circ_m:.1f} m
            </div>
            """
            folium.Polygon(
                locations=[[lat_u, lon_u], [lat_v, lon_v], [lat_w, lon_w]],
                color=color_t,
                weight=3.0,
                fill=True,
                fill_color=color_t,
                fill_opacity=0.65,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"🔺 Triángulo Top #{rank_t} | {m_name}: {score_t:.4f}"
            ).add_to(fg_k2)
        fg_k2.add_to(m)

    # 3. CAPA TOP CALLES / ARISTAS (k=1)
    df_s1 = centralities.df_centrality_s1
    if show_k1 and top_k1 > 0 and len(sc.s1_edges) > 0 and len(df_s1) > 0:
        actual_k1 = min(top_k1, len(df_s1))
        top_edges = df_s1.sort_values(by=col_s1, ascending=False).head(actual_k1).copy()
        top_edges['rank'] = range(1, len(top_edges) + 1)
        edge_scores = top_edges[col_s1].values
        edge_colors, _ = _get_colormap_rgba(colormap_name, edge_scores)
        
        fg_k1 = folium.FeatureGroup(name=f"🛣️ Top {actual_k1} Calles (k=1)", show=True, overlay=True)
        for idx_row, (_, row) in enumerate(top_edges.iterrows()):
            s_id = int(row['simplex_id'])
            u, v = sc.s1_edges[s_id]
            lat1, lon1 = sc.coords_geographic[u]
            lat2, lon2 = sc.coords_geographic[v]
            color_e = edge_colors[idx_row]
            score_e = float(row[col_s1])
            rank_e = int(row['rank'])
            length_m = float(row.get('length_m', 0.0))
            hwy = str(row.get('highway', 'road'))
            
            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; font-size: 13px; width: 240px; line-height: 1.4;">
                <b style="color: #059669; font-size: 14px;">🛣️ Calle Top #{rank_e} (k=1)</b>
                <hr style="margin: 4px 0; border: 0.5px solid #cbd5e1;"/>
                <b>ID Símplex:</b> #{s_id}<br/>
                <b>Vértices OSM:</b> [{sc.idx_to_node[u]}, {sc.idx_to_node[v]}]<br/>
                <b style="color: #059669;">{m_name}:</b> {score_e:.5f}<br/>
                <b>Longitud:</b> {length_m:.1f} m | <b>Tipo:</b> {hwy}
            </div>
            """
            folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color=color_e,
                weight=6.0,
                opacity=0.95,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"🛣️ Calle Top #{rank_e} | {m_name}: {score_e:.4f}"
            ).add_to(fg_k1)
        fg_k1.add_to(m)

    # 4. CAPA TOP NODOS (k=0)
    df_s0 = centralities.df_centrality_s0
    if show_k0 and top_k0 > 0 and len(sc.s0_nodes) > 0 and len(df_s0) > 0:
        actual_k0 = min(top_k0, len(df_s0))
        top_nodes = df_s0.sort_values(by=col_s0, ascending=False).head(actual_k0).copy()
        top_nodes['rank'] = range(1, len(top_nodes) + 1)
        node_scores = top_nodes[col_s0].values
        node_colors, _ = _get_colormap_rgba(colormap_name, node_scores)
        
        fg_k0 = folium.FeatureGroup(name=f"⭐ Top {actual_k0} Nodos (k=0)", show=True, overlay=True)
        for idx_row, (_, row) in enumerate(top_nodes.iterrows()):
            s_id = int(row['simplex_id'])
            lat, lon = float(row['lat']), float(row['lon'])
            osmid = row['osmid']
            color_n = node_colors[idx_row]
            score_n = float(row[col_s0])
            rank_n = int(row['rank'])
            deg_k1 = int(row.get('degree_k1_edges', 0))
            deg_k2 = int(row.get('degree_k2_triangles', 0))
            
            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; font-size: 13px; width: 230px; line-height: 1.4;">
                <b style="color: #e11d48; font-size: 14px;">⭐ Nodo Top #{rank_n} (k=0)</b>
                <hr style="margin: 4px 0; border: 0.5px solid #cbd5e1;"/>
                <b>OSM ID:</b> #{osmid} (Símplex #{s_id})<br/>
                <b>Coordenadas:</b> ({lat:.5f}, {lon:.5f})<br/>
                <b style="color: #e11d48;">{m_name}:</b> {score_n:.5f}<br/>
                <b>Grado k=1 (Calles):</b> {deg_k1}<br/>
                <b>Grado k=2 (Triángulos):</b> {deg_k2}
            </div>
            """
            folium.CircleMarker(
                location=[lat, lon],
                radius=8.5,
                color="#ffffff",
                fill=True,
                fill_color=color_n,
                fill_opacity=1.0,
                weight=2.5,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"⭐ Nodo Top #{rank_n} | {m_name}: {score_n:.4f}"
            ).add_to(fg_k0)
        fg_k0.add_to(m)

    inject_compact_map_styles(m)
    add_chromatic_scale_legend(
        m,
        colormap_name=colormap_name,
        title=f"Top Símplices ({m_name})",
        vmin_label="Menor Top",
        vmax_label="Mayor Top"
    )
    folium.LayerControl(collapsed=False).add_to(m)
    return m

"""
Plataforma de Análisis de Datos Topológicos (TDA) y Centralidad Simplicial en Redes Urbanas.
Edición Especial: Universidad del Pacífico (UP) — Casos de Estudio en Lima Metropolitana.
Desarrollada para docencia e investigación sin necesidad de instalación local.
"""
import os
import json
import streamlit as st
import numpy as np
import pandas as pd
from streamlit_folium import st_folium

from src.network_ops import (
    fetch_urban_network_by_coords,
    fetch_urban_network_by_place,
    process_and_project_graph
)
from src.simplicial import (
    SimplicialComplex,
    build_alpha_complex_from_graph,
    build_flag_complex_from_graph
)
from src.hodge_centrality import (
    HodgeOperators,
    SimplicialCentralities,
    compute_hodge_operators,
    compute_simplicial_centralities
)
from src.visualization import (
    create_folium_geographic_map,
    create_folium_simplicial_map,
    create_folium_filtered_simplicial_map,
    create_filtration_histogram_figure,
    BASEMAP_TILES
)
from src.export_utils import (
    build_hodge_matrices_zip,
    get_degree_centrality_export_csv,
    get_betweenness_centrality_export_csv,
    get_eigenvector_centrality_export_csv
)
from src.controller import (
    get_graph_from_coords,
    get_graph_from_place,
    load_cached_lima_district,
    run_pipeline
)

# Configuración de página Streamlit
st.set_page_config(
    page_title="TDA Urban Centrality | Universidad del Pacífico",
    page_icon="🇵🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .app-header {
        background: linear-gradient(135deg, #0b192c 0%, #1e3e62 50%, #000000 100%);
        color: white;
        padding: 22px 28px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    .app-title {
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .app-subtitle {
        font-size: 13.5px;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .metric-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 16px;
    }
    .metric-badge {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 14px;
        min-width: 140px;
        flex: 1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .metric-badge-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-badge-value {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
    }
    .simplex-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        border-radius: 8px;
        padding: 14px;
        margin-top: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .status-badge-ok {
        background-color: #dcfce7;
        color: #166534;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 9999px;
    }
    .leaflet-container, .stFolium iframe {
        background-color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado
st.markdown("""
<div class="app-header">
    <div class="app-title">
        <span>🇵🇪</span> TDA Urban Centrality & Hodge Laplacian — Universidad del Pacífico
    </div>
    <div class="app-subtitle">
        Plataforma Web de Topología Simplicial y Análisis de Redes Urbanas en Lima Metropolitana | Curso TDA
    </div>
</div>
""", unsafe_allow_html=True)

# Cargar catálogo de distritos de Lima
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "distritos_lima", "manifest.json")
lima_manifest = {}
if os.path.exists(MANIFEST_PATH):
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            lima_manifest = json.load(f)
    except Exception:
        lima_manifest = {}

# Inicialización de estado de sesión
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None
if "selected_simplex_info" not in st.session_state:
    st.session_state.selected_simplex_info = None
if "t3_highlighted_simplex" not in st.session_state:
    st.session_state.t3_highlighted_simplex = None

# =============================================================================
# PANEL LATERAL (SIDEBAR)
# =============================================================================
with st.sidebar:
    st.markdown("### 🎓 Casos de Estudio — Lima (UP)")
    
    query_mode = st.radio(
        "Modalidad de Trabajo:",
        [
            "🇵🇪 Distritos de Lima Pre-cargados (Carga Instantánea)",
            "📍 Coordenadas + Radio (En vivo)",
            "🏙️ Topónimo Libre (En vivo)"
        ],
        index=0
    )
    
    district_selected_key = None
    if query_mode.startswith("🇵🇪"):
        dist_options = list(lima_manifest.keys())
        format_dict = {
            k: f"{lima_manifest[k].get('group_suggested', 'Grupo')}: {lima_manifest[k].get('name', k)}"
            for k in dist_options
        }
        
        district_selected_key = st.selectbox(
            "Seleccionar Distrito / Caso Asignado:",
            dist_options,
            format_func=lambda k: format_dict.get(k, k),
            index=0
        )
        
        d_info = lima_manifest[district_selected_key]
        st.info(f"**Asignación:** {d_info.get('group_suggested', '')}\n\n"
                f"**Zona:** {d_info.get('description', '')}\n\n"
                f"📊 *Red precargada:* {d_info.get('node_count')} nodos, {d_info.get('edge_count')} aristas.")
        lat_input, lon_input, radius_input = d_info["lat"], d_info["lon"], d_info["radius"]
        place_input = ""
    elif query_mode.startswith("📍"):
        c1, c2 = st.columns(2)
        with c1:
            lat_input = st.number_input("Latitud", value=-12.0833, format="%.6f")
        with c2:
            lon_input = st.number_input("Longitud", value=-77.0494, format="%.6f")
        radius_input = st.slider("Radio (m):", 200, 2500, 750, 50, help="Para clase, radios de 500-1000m aseguran respuesta en segundos.")
        place_input = ""
    else:
        place_input = st.text_input("Lugar OSM:", value="Miraflores, Lima, Peru")
        lat_input, lon_input, radius_input = 0.0, 0.0, 750

    network_type_input = st.selectbox("Tipo de Red:", ["walk", "drive", "bike", "all"], index=0)

    st.markdown("---")
    st.markdown("### 🔺 Complejo Simplicial")
    complex_type_input = st.radio(
        "Modelo de Complejo:",
        ["Alpha Complex", "Flag Complex (Cliques)"],
        index=0,
        help="Alpha Complex: Recomendado para redes viales urbanas (parametrizado por radio métrico)."
    )
    
    alpha_param_input = st.slider(
        "Parámetro Alpha α (metros):",
        10.0, 300.0, 65.0, 5.0,
        help="Radio circunscrito máximo para incluir 2-símplices (triángulos/manzanas cerradas)."
    ) if complex_type_input == "Alpha Complex" else 0.0

    st.markdown("---")
    st.markdown("### 🎨 Visualización")
    colormap_input = st.selectbox("Paleta de Color:", ["Plasma", "Viridis", "Inferno", "Magma", "Turbo"], index=0)
    basemap_input = st.selectbox("Mapa Base:", list(BASEMAP_TILES.keys()), index=0)

    btn_process = st.button("🚀 Procesar Red Simplicial", type="primary", use_container_width=True)

# =============================================================================
# EJECUCIÓN DEL PIPELINE
# =============================================================================
if btn_process or st.session_state.pipeline_results is None:
    try:
        if district_selected_key is not None:
            with st.spinner("⚡ Cargando distrito de Lima desde el repositorio..."):
                G_simple, df_nodes, df_edges, meta = load_cached_lima_district(district_selected_key)
        else:
            with st.spinner("🌐 Descargando red vial desde OpenStreetMap..."):
                if query_mode.startswith("📍"):
                    G_simple, df_nodes, df_edges = get_graph_from_coords(
                        lat_input, lon_input, radius_input, network_type_input, node_merge_tolerance=10.0
                    )
                else:
                    G_simple, df_nodes, df_edges = get_graph_from_place(
                        place_input, network_type_input, node_merge_tolerance=10.0
                    )

        with st.spinner("📐 Calculando Complejo Simplicial, Operadores Frontera y Laplaciano de Hodge..."):
            c_type = "alpha" if complex_type_input == "Alpha Complex" else "flag"
            sc, hodge, centralities = run_pipeline(
                G_simple,
                complex_type=c_type,
                alpha_meters=alpha_param_input
            )
            st.session_state.pipeline_results = {
                "G_simple": G_simple,
                "sc": sc,
                "hodge": hodge,
                "centralities": centralities,
                "colormap": colormap_input,
                "basemap": basemap_input
            }
            st.toast("✅ Análisis topológico completado con éxito", icon="🎉")
    except Exception as ex:
        st.error(f"❌ Error al procesar la red: {ex}")
        st.stop()

# =============================================================================
# VISUALIZACIÓN Y PESTAÑAS DE TRABAJO
# =============================================================================
if st.session_state.pipeline_results is not None:
    res = st.session_state.pipeline_results
    sc: SimplicialComplex = res["sc"]
    hodge: HodgeOperators = res["hodge"]
    centralities: SimplicialCentralities = res["centralities"]

    n0, n1, n2 = len(sc.s0_nodes), len(sc.s1_edges), len(sc.s2_triangles)
    chain_ok = (hodge.chain_property_norm < 1e-9)
    chain_badge = '<span class="status-badge-ok">✓ ||B₁B₂|| = 0</span>' if chain_ok else f'<span style="color:red;">Norma: {hodge.chain_property_norm:.2e}</span>'

    st.markdown(f"""
    <div class="metric-badge-container">
        <div class="metric-badge">
            <div class="metric-badge-label">Nodos (S₀)</div>
            <div class="metric-badge-value">{n0:,}</div>
        </div>
        <div class="metric-badge">
            <div class="metric-badge-label">Calles / Aristas (S₁)</div>
            <div class="metric-badge-value">{n1:,}</div>
        </div>
        <div class="metric-badge">
            <div class="metric-badge-label">Manzanas / Caras (S₂)</div>
            <div class="metric-badge-value">{n2:,}</div>
        </div>
        <div class="metric-badge">
            <div class="metric-badge-label">Números de Betti</div>
            <div class="metric-badge-value">β₀={hodge.betti_0} | β₁={hodge.betti_1}</div>
        </div>
        <div class="metric-badge">
            <div class="metric-badge-label">Propiedad Homológica</div>
            <div class="metric-badge-value" style="font-size:14px; margin-top:4px;">{chain_badge}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Pestaña 1: Exploración Geográfica",
        "📐 Pestaña 2: Centralidades Simpliciales",
        "🔬 Pestaña 3: Filtración y Rankings",
        "📘 Pestaña 4: Guía de Interpretación (UP)"
    ])

    # PESTAÑA 1: EXPLORACIÓN GEOGRÁFICA
    with tab1:
        c_t1_l, c_t1_r = st.columns([1, 2.8], gap="medium")
        with c_t1_l:
            st.markdown("#### 🔍 Inspección de Nodos Urbanos")
            geo_metric = st.selectbox(
                "Colorear nodos por:",
                ["Grado de Conectividad", "Intermediación (Betweenness)", "Autovector"],
                index=0
            )
            m_key = "degree" if "Grado" in geo_metric else ("betweenness" if "Intermediación" in geo_metric else "eigenvector")
            
            sel_node = st.selectbox("Seleccionar Nodo (ID):", range(n0), index=0, format_func=lambda i: f"Nodo #{i} (OSM {sc.idx_to_node[i]})")
            if sel_node is not None and sel_node < len(centralities.df_centrality_s0):
                r = centralities.df_centrality_s0.iloc[sel_node]
                st.markdown(f"""
                <div class="simplex-card">
                    <h4 style="margin:0; color:#0f172a;">📍 Nodo OSM #{r['osmid']}</h4>
                    <p style="margin:4px 0;"><b>Coordenadas:</b> ({r['lat']:.5f}, {r['lon']:.5f})</p>
                    <p style="margin:4px 0;"><b>Grado Calles (k=1):</b> {int(r['degree_k1_edges'])}</p>
                    <p style="margin:4px 0;"><b>Grado Triángulos (k=2):</b> {int(r['degree_k2_triangles'])}</p>
                    <p style="margin:4px 0;"><b>Betweenness:</b> {r['betweenness_score']:.6f}</p>
                    <p style="margin:0;"><b>Eigenvector:</b> {r['eigenvector_simplicial_proj']:.6f}</p>
                </div>
                """, unsafe_allow_html=True)
        with c_t1_r:
            st.markdown("#### 🗺️ Visor Georreferenciado")
            f_map = create_folium_geographic_map(sc, centralities, basemap_name=basemap_input, colormap_name=colormap_input, metric_name=m_key)
            st_folium(f_map, use_container_width=True, height=750, returned_objects=["last_object_clicked"])

    # PESTAÑA 2: CENTRALIDADES SIMPLICIALES
    with tab2:
        c_t2_l, c_t2_r = st.columns([1, 2.8], gap="medium")
        with c_t2_l:
            st.markdown("#### ⚙️ Control Multicapa")
            c_choice = st.radio(
                "Métrica Simplicial:",
                ["Simplicial Degree Centrality", "Simplicial Betweenness Centrality", "Simplicial Eigenvector Centrality"],
                index=0
            )
            c_code = "degree" if "Degree" in c_choice else ("betweenness" if "Betweenness" in c_choice else "eigenvector")
            
            basemap_t2 = st.radio(
                "Fondo de Mapa:",
                ["Esri Claro (Light Gray Canvas)", "Lienzo en Blanco (Sin teselas)", "CartoDB Positron (Fondo claro)", "CartoDB Dark Matter (Fondo oscuro)"],
                index=0
            )
            st.markdown("---")
            st.markdown("#### 🧩 Capas Activas:")
            show_k0 = st.checkbox("🔘 Nodos (k=0)", value=True)
            show_k1 = st.checkbox("➖ Calles / Aristas (k=1)", value=True)
            show_k2 = st.checkbox("🔺 Manzanas / Caras (k=2)", value=True)

            st.markdown("---")
            st.markdown("#### 📥 Exportar Resultados:")
            z_bytes = build_hodge_matrices_zip(hodge)
            st.download_button("📦 Matrices Hodge (ZIP)", data=z_bytes, file_name="matrices_hodge.zip", mime="application/zip", use_container_width=True)
            csv_deg = get_degree_centrality_export_csv(sc, centralities)
            st.download_button("📄 Centralidad de Grado (CSV)", data=csv_deg, file_name="degree_centrality.csv", mime="text/csv", use_container_width=True)
        with c_t2_r:
            st.markdown("#### 📐 Visor de Centralidad Simplicial de Orden Superior")
            sim_map = create_folium_simplicial_map(
                sc=sc, centralities=centralities, hodge=hodge,
                basemap_name=basemap_t2, colormap_name=colormap_input,
                metric_type=c_code, show_k0=show_k0, show_k1=show_k1, show_k2=show_k2
            )
            st_folium(sim_map, use_container_width=True, height=750, returned_objects=["last_object_clicked"])

    # PESTAÑA 3: FILTRACIÓN Y RANKINGS
    with tab3:
        c_t3_l, c_t3_r = st.columns([1, 2.8], gap="medium")
        with c_t3_l:
            st.markdown("#### 🔬 Control de Filtración")
            filt_m = st.radio("Métrica de Filtrado:", ["Simplicial Degree Centrality", "Simplicial Betweenness Centrality", "Simplicial Eigenvector Centrality"], index=0, key="t3_m")
            f_code = "degree" if "Degree" in filt_m else ("betweenness" if "Betweenness" in filt_m else "eigenvector")

            df_s0, df_s1, df_s2 = centralities.df_centrality_s0, centralities.df_centrality_s1, centralities.df_centrality_s2
            if f_code == "degree":
                v0, v1, v2 = df_s0['degree_k1_edges'].values, df_s1['degree_upper'].values, (df_s2['degree_adjacent'].values if len(df_s2)>0 else np.array([]))
                col0, col1, col2 = 'degree_k1_edges', 'degree_upper', 'degree_adjacent'
            elif f_code == "betweenness":
                v0, v1, v2 = df_s0['betweenness_score'].values, df_s1['betweenness_score'].values, (df_s2['betweenness_score'].values if len(df_s2)>0 else np.array([]))
                col0, col1, col2 = 'betweenness_score', 'betweenness_score', 'betweenness_score'
            else:
                v0, v1, v2 = df_s0['eigenvector_node'].values, df_s1['eigenvector_total'].values, (df_s2['eigenvector_score'].values if len(df_s2)>0 else np.array([]))
                col0, col1, col2 = 'eigenvector_node', 'eigenvector_total', 'eigenvector_score'

            dim_hist = st.selectbox("Histograma para:", ["k = 0 (Nodos)", "k = 1 (Aristas)", "k = 2 (Triángulos)"], index=0)
            hist_v = v0 if "k = 0" in dim_hist else (v1 if "k = 1" in dim_hist else v2)
            d_tag = "k=0 (Nodos)" if "k = 0" in dim_hist else ("k=1 (Aristas)" if "k = 1" in dim_hist else "k=2 (Triángulos)")

            max0 = float(np.max(v0)) if len(v0)>0 else 1.0
            min0 = float(np.min(v0)) if len(v0)>0 else 0.0
            th0 = st.slider("Umbral Nodos (k=0):", min0, max0, min0, (max0-min0)/100.0 if max0>min0 else 0.01)

            max1 = float(np.max(v1)) if len(v1)>0 else 1.0
            min1 = float(np.min(v1)) if len(v1)>0 else 0.0
            th1 = st.slider("Umbral Aristas (k=1):", min1, max1, min1, (max1-min1)/100.0 if max1>min1 else 0.01)

            th2 = 0.0
            if len(v2) > 0:
                max2 = float(np.max(v2))
                min2 = float(np.min(v2))
                th2 = st.slider("Umbral Triángulos (k=2):", min2, max2, min2, (max2-min2)/100.0 if max2>min2 else 0.01)

            cur_th = th0 if "k = 0" in dim_hist else (th1 if "k = 1" in dim_hist else th2)
            f_hist = create_filtration_histogram_figure(values=hist_v, metric_name=filt_m, threshold=cur_th, dimension_label=d_tag, colormap_name=colormap_input)
            st.plotly_chart(f_hist, use_container_width=True)

            hl_mode = st.radio("Elementos Descartados:", ["Ocultar descartados", "Atenuar descartados"], index=0)
            bm_t3 = st.selectbox("Fondo:", ["Esri Claro (Light Gray Canvas)", "Lienzo en Blanco (Sin teselas)", "CartoDB Positron (Fondo claro)"], index=0)
            t3_k0 = st.checkbox("Ver Nodos", value=True)
            t3_k1 = st.checkbox("Ver Aristas", value=True)
            t3_k2 = st.checkbox("Ver Triángulos", value=True)
        with c_t3_r:
            st.markdown("#### 🔬 Visor del Subcomplejo Filtrado")
            p0 = int(np.sum(v0 >= th0))
            p1 = int(np.sum(v1 >= th1))
            p2 = int(np.sum(v2 >= th2)) if len(v2)>0 else 0
            st.markdown(f"**Retención:** Nodos supervivientes: **{p0}/{len(v0)}** | Calles supervivientes: **{p1}/{len(v1)}** | Triángulos supervivientes: **{p2}/{len(v2)}**")
            
            f_filt_map = create_folium_filtered_simplicial_map(
                sc=sc, centralities=centralities, hodge=hodge, metric_type=f_code,
                threshold_k0=th0, threshold_k1=th1, threshold_k2=th2,
                highlight_mode=hl_mode, show_k0=t3_k0, show_k1=t3_k1, show_k2=t3_k2,
                basemap_name=bm_t3, colormap_name=colormap_input,
                highlighted_simplex=st.session_state.t3_highlighted_simplex
            )
            st_folium(f_filt_map, use_container_width=True, height=750, returned_objects=["last_object_clicked"])

            st.markdown("#### 🏆 Ranking de Elementos Clave")
            rt0, rt1, rt2 = st.tabs(["Top Calles Más Centrales (k=1)", "Top Nodos Más Centrales (k=0)", "Top Triángulos (k=2)"])
            with rt0:
                top_edges = df_s1.sort_values(by=col1, ascending=False).head(20).copy()
                top_edges['ranking'] = range(1, len(top_edges) + 1)
                cols_edges = list(dict.fromkeys(['ranking', 'simplex_id', 'vertices', col1, 'length_m', 'degree_upper', 'betweenness_score', 'eigenvector_total']))
                cols_edges = [c for c in cols_edges if c in top_edges.columns]
                st.dataframe(top_edges[cols_edges], use_container_width=True)
                csv_e = top_edges[cols_edges].to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar Top Calles (CSV)", data=csv_e, file_name=f"top_calles_{f_code}.csv", mime="text/csv", key="dl_top_calles_t3")
            with rt1:
                top_nodes = df_s0.sort_values(by=col0, ascending=False).head(20).copy()
                top_nodes['ranking'] = range(1, len(top_nodes) + 1)
                cols_nodes = list(dict.fromkeys(['ranking', 'simplex_id', 'osmid', col0, 'degree_k1_edges', 'degree_k2_triangles', 'betweenness_score', 'eigenvector_simplicial_proj']))
                cols_nodes = [c for c in cols_nodes if c in top_nodes.columns]
                st.dataframe(top_nodes[cols_nodes], use_container_width=True)
                csv_n = top_nodes[cols_nodes].to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar Top Nodos (CSV)", data=csv_n, file_name=f"top_nodos_{f_code}.csv", mime="text/csv", key="dl_top_nodos_t3")
            with rt2:
                if len(df_s2) > 0:
                    top_tri = df_s2.sort_values(by=col2, ascending=False).head(20).copy()
                    top_tri['ranking'] = range(1, len(top_tri) + 1)
                    cols_tri = list(dict.fromkeys(['ranking', 'simplex_id', 'vertices', col2, 'area_m2', 'circumradius', 'degree_adjacent', 'betweenness_score', 'eigenvector_score']))
                    cols_tri = [c for c in cols_tri if c in top_tri.columns]
                    st.dataframe(top_tri[cols_tri], use_container_width=True)
                    csv_t = top_tri[cols_tri].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Descargar Top Triángulos (CSV)", data=csv_t, file_name=f"top_triangulos_{f_code}.csv", mime="text/csv", key="dl_top_tri_t3")
                else:
                    st.info("No se generaron 2-símplices para la configuración actual.")

    # PESTAÑA 4: GUÍA PEDAGÓGICA
    with tab4:
        st.markdown("### 📘 Guía de Interpretación de Datos Topológicos Urbanos")
        st.markdown("""
        Esta plataforma complementa la teoría de grafos tradicional al estudiar las **interacciones de orden superior** en la red urbana de tu distrito asignado:
        
        #### 1. Números de Betti (Topología Algebraica)
        - **$\\beta_0$ (Componentes Conexas):** Representa el número de islas o sub-redes aisladas. Si $\\beta_0 = 1$, la red está completamente conectada. Si $\\beta_0 > 1$, existen zonas sin conexión vial directa con el resto.
        - **$\\beta_1$ (1-Ciclos / Agujeros no Triangulados):** Representa el número de vacíos urbanos, grandes parques o manzanas impenetrables que obligan al flujo vehicular y peatonal a circunnavegar en bucle.
        
        #### 2. Operador Laplaciano de Hodge ($L_1 = L_1^{\\text{down}} + L_1^{\\text{up}}$)
        - **Laplaciano Inferior ($L_1^{\\text{down}} = B_1^T B_1$):** Captura las interacciones inducidas a través de intersecciones compartidas (flujos de gradiente).
        - **Laplaciano Superior ($L_1^{\\text{up}} = B_2 B_2^T$):** Modela la circulación inducida por manzanas cerradas adyacentes (rotacional local).
        - **Propiedad de Cadena ($\\|B_1 B_2\\| = 0$):** Demuestra rigurosamente que el borde de una cara bidimensional no tiene bordes adicionales.
        
        #### 3. Preguntas Clave para el Informe de tu Grupo
        1. *¿Cuáles son las calles con mayor Betweenness Simplicial vs. Betweenness Clásico?*
        2. *¿Cómo influye la morfología del distrito (ej. cuadrícula renacentista en Lima Cercado vs. topografía costera en Barranco o parques lineales en San Borja) en el valor de $\\beta_1$?*
        3. *Al aumentar el umbral de filtración, ¿qué subcomplejo arterial sobrevive como espina dorsal del distrito?*
        """)

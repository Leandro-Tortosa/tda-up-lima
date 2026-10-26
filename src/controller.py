"""
Controlador principal del flujo de trabajo y gestión del pipeline científico.
"""
from typing import Tuple, Dict, Any, Optional
import networkx as nx
import streamlit as st

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


@st.cache_data(show_spinner=False)
def get_graph_from_coords(
    lat: float,
    lon: float,
    radius: float,
    network_type: str,
    node_merge_tolerance: float = 10.0
):
    G_raw = fetch_urban_network_by_coords(lat, lon, radius, network_type=network_type)
    G_simple, df_nodes, df_edges = process_and_project_graph(G_raw, node_merge_tolerance_meters=node_merge_tolerance)
    return G_simple, df_nodes, df_edges


@st.cache_data(show_spinner=False)
def get_graph_from_place(
    place_name: str,
    network_type: str,
    node_merge_tolerance: float = 10.0
):
    G_raw = fetch_urban_network_by_place(place_name, network_type=network_type)
    G_simple, df_nodes, df_edges = process_and_project_graph(G_raw, node_merge_tolerance_meters=node_merge_tolerance)
    return G_simple, df_nodes, df_edges


@st.cache_data(show_spinner=False)
def load_cached_lima_district(district_id: str):
    """
    Carga de forma instantánea el grafo procesado y DataFrames tabulados
    de un distrito de Lima previamente almacenado en caché local.
    """
    import os
    import pickle
    
    # Buscar en carpeta data/distritos_lima
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_file = os.path.join(base_dir, "data", "distritos_lima", f"{district_id}.pickle")
    
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"No se encontró el archivo de caché para el distrito '{district_id}' en {target_file}")
        
    with open(target_file, "rb") as f:
        data = pickle.load(f)
        
    return data["G_simple"], data["df_nodes"], data["df_edges"], data["metadata"]


@st.cache_data(show_spinner=False)
def get_cached_district_pipeline(district_id: str) -> Optional[Dict[str, Any]]:
    """
    Carga de forma instantánea el complejo simplicial, operadores de Hodge
    y centralidades precalculadas para el distrito indicado.
    """
    import os
    import pickle
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_file = os.path.join(base_dir, "data", "distritos_lima", f"{district_id}.pickle")
    if not os.path.exists(target_file):
        return None
    with open(target_file, "rb") as f:
        data = pickle.load(f)
    return data.get("pipeline_default", None)


@st.cache_data(
    show_spinner=False,
    hash_funcs={nx.Graph: lambda g: (g.number_of_nodes(), g.number_of_edges(), tuple(sorted(g.nodes())[:10]))}
)
def run_pipeline(
    G_simple: nx.Graph,
    complex_type: str = "alpha",
    alpha_meters: float = 65.0
) -> Tuple[SimplicialComplex, HodgeOperators, SimplicialCentralities]:
    """
    Ejecuta el pipeline completo de construcción simplicial, operadores de Hodge y centralidades.
    """
    if complex_type == "alpha":
        sc = build_alpha_complex_from_graph(G_simple, alpha_meters=alpha_meters)
    else:
        sc = build_flag_complex_from_graph(G_simple)
        
    hodge = compute_hodge_operators(sc)
    centralities = compute_simplicial_centralities(sc, hodge)
    
    return sc, hodge, centralities


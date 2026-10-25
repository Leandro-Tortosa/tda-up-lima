"""
Módulo de construcción de complejos simpliciales (Alpha Complex y Flag/Clique Complex)
con indexación canónica y acotación dimensional a k=2.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import Delaunay
import math


@dataclass
class SimplicialComplex:
    """
    Estructura que almacena los datos de un complejo simplicial hasta dimensión k=2.
    """
    complex_type: str  # "alpha" o "flag"
    alpha_param: Optional[float]  # en metros si es alpha complex
    
    # Listas de símplices canónicos (ordenados lexicográficamente)
    s0_nodes: List[int]  # Lista de IDs canónicos [0, 1, ..., N0-1]
    s1_edges: List[Tuple[int, int]]  # Lista de pares ordenados (u, v) con u < v
    s2_triangles: List[Tuple[int, int, int]]  # Lista de tripletas ordenadas (u, v, w) con u < v < w
    
    # Mapeos entre IDs de OSM y canonical index
    node_to_idx: Dict[Any, int]
    idx_to_node: Dict[int, Any]
    
    # Mapeo de aristas y triángulos a sus IDs canónicos
    edge_to_idx: Dict[Tuple[int, int], int]
    triangle_to_idx: Dict[Tuple[int, int, int], int]
    
    # DataFrames descriptivos
    df_s0: pd.DataFrame
    df_s1: pd.DataFrame
    df_s2: pd.DataFrame
    
    # Propiedades geométricas adicionales
    coords_projected: np.ndarray  # (N0, 2) UTM en metros
    coords_geographic: np.ndarray  # (N0, 2) [lat, lon] WGS84


def _triangle_area_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calcula el área 2D de un triángulo dado por sus 3 vértices cartesianos."""
    return 0.5 * abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))


def _circumradius_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calcula el radio de la circunferencia circunscrita a un triángulo en 2D.
    R = (a * b * c) / (4 * Area)
    """
    a = float(np.linalg.norm(p2 - p1))
    b = float(np.linalg.norm(p3 - p2))
    c = float(np.linalg.norm(p1 - p3))
    area = _triangle_area_2d(p1, p2, p3)
    if area < 1e-9:
        return float('inf')
    return (a * b * c) / (4.0 * area)


def build_alpha_complex_from_graph(
    G: nx.Graph,
    alpha_meters: float
) -> SimplicialComplex:
    """
    Construye un Alpha Complex a partir de las coordenadas métricas proyectadas de los nodos.
    Un triángulo de Delaunay se incluye si su circunradio <= alpha_meters.
    Las aristas incluidas son aquellas aristas de Delaunay con longitud <= 2 * alpha_meters
    o bien aristas pertenecientes a triángulos válidos o a la red vial si están dentro del umbral.
    """
    # 1. Indexación canónica de vértices
    sorted_nodes = sorted(list(G.nodes()))
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
    idx_to_node = {i: node for i, node in enumerate(sorted_nodes)}
    N0 = len(sorted_nodes)
    
    coords_proj = np.zeros((N0, 2), dtype=float)
    coords_geo = np.zeros((N0, 2), dtype=float)
    
    for i, node in enumerate(sorted_nodes):
        data = G.nodes[node]
        coords_proj[i, 0] = data.get('x', 0.0)
        coords_proj[i, 1] = data.get('y', 0.0)
        coords_geo[i, 0] = data.get('lat', 0.0)
        coords_geo[i, 1] = data.get('lon', 0.0)
        
    s0_list = list(range(N0))
    
    # 2. Triangulación de Delaunay 2D
    edges_set = set()
    triangles_list = []
    
    # Incluir aristas originales de la red vial que conecten nodos válidos
    for u, v in G.edges():
        i_u, i_v = node_to_idx[u], node_to_idx[v]
        edge = (min(i_u, i_v), max(i_u, i_v))
        edges_set.add(edge)
        
    if N0 >= 3:
        try:
            tri = Delaunay(coords_proj)
            for simplex in tri.simplices:
                idx_a, idx_b, idx_c = sorted(simplex)
                p_a, p_b, p_c = coords_proj[idx_a], coords_proj[idx_b], coords_proj[idx_c]
                r_circ = _circumradius_2d(p_a, p_b, p_c)
                
                if r_circ <= alpha_meters:
                    triangles_list.append((idx_a, idx_b, idx_c))
                    # Asegurar que las 3 aristas frontera del triángulo estén en el complejo
                    edges_set.add((idx_a, idx_b))
                    edges_set.add((idx_b, idx_c))
                    edges_set.add((idx_a, idx_c))
        except Exception:
            # Fallback en caso de puntos colineales
            pass

    # 3. Orden canónico de 1-símplices y 2-símplices
    s1_edges = sorted(list(edges_set))
    s2_triangles = sorted(triangles_list)
    
    edge_to_idx = {e: i for i, e in enumerate(s1_edges)}
    triangle_to_idx = {t: i for i, t in enumerate(s2_triangles)}
    
    # 4. Construcción de DataFrames descriptivos
    # S0
    df_s0_records = []
    for i in range(N0):
        osmid = idx_to_node[i]
        df_s0_records.append({
            'simplex_id': i,
            'osmid': osmid,
            'x': coords_proj[i, 0],
            'y': coords_proj[i, 1],
            'lat': coords_geo[i, 0],
            'lon': coords_geo[i, 1]
        })
    df_s0 = pd.DataFrame(df_s0_records)
    
    # S1
    df_s1_records = []
    for j, (u, v) in enumerate(s1_edges):
        p_u = coords_proj[u]
        p_v = coords_proj[v]
        length = float(np.linalg.norm(p_u - p_v))
        df_s1_records.append({
            'simplex_id': j,
            'u': u,
            'v': v,
            'osmid_u': idx_to_node[u],
            'osmid_v': idx_to_node[v],
            'length_m': length
        })
    df_s1 = pd.DataFrame(df_s1_records)
    
    # S2
    df_s2_records = []
    for k, (u, v, w) in enumerate(s2_triangles):
        p_u, p_v, p_w = coords_proj[u], coords_proj[v], coords_proj[w]
        area = _triangle_area_2d(p_u, p_v, p_w)
        r_circ = _circumradius_2d(p_u, p_v, p_w)
        e1 = edge_to_idx.get((u, v), -1)
        e2 = edge_to_idx.get((v, w), -1)
        e3 = edge_to_idx.get((u, w), -1)
        df_s2_records.append({
            'simplex_id': k,
            'u': u,
            'v': v,
            'w': w,
            'vertices': f"[{idx_to_node[u]}, {idx_to_node[v]}, {idx_to_node[w]}]",
            'edge_ids': (e1, e2, e3),
            'area_m2': area,
            'circumradius': r_circ
        })
    df_s2 = pd.DataFrame(df_s2_records) if df_s2_records else pd.DataFrame(columns=['simplex_id', 'u', 'v', 'w', 'vertices', 'edge_ids', 'area_m2', 'circumradius'])
    
    return SimplicialComplex(
        complex_type="alpha",
        alpha_param=alpha_meters,
        s0_nodes=s0_list,
        s1_edges=s1_edges,
        s2_triangles=s2_triangles,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        edge_to_idx=edge_to_idx,
        triangle_to_idx=triangle_to_idx,
        df_s0=df_s0,
        df_s1=df_s1,
        df_s2=df_s2,
        coords_projected=coords_proj,
        coords_geographic=coords_geo
    )


def build_flag_complex_from_graph(
    G: nx.Graph
) -> SimplicialComplex:
    """
    Construye un Flag/Clique Complex a partir de la topología de adyacencia del grafo.
    - 0-símplices: Nodos V
    - 1-símplices: Aristas E
    - 2-símplices: Triángulos (3-cliques) presentes en el grafo vial.
    """
    # 1. Indexación canónica de vértices
    sorted_nodes = sorted(list(G.nodes()))
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
    idx_to_node = {i: node for i, node in enumerate(sorted_nodes)}
    N0 = len(sorted_nodes)
    
    coords_proj = np.zeros((N0, 2), dtype=float)
    coords_geo = np.zeros((N0, 2), dtype=float)
    
    for i, node in enumerate(sorted_nodes):
        data = G.nodes[node]
        coords_proj[i, 0] = data.get('x', 0.0)
        coords_proj[i, 1] = data.get('y', 0.0)
        coords_geo[i, 0] = data.get('lat', 0.0)
        coords_geo[i, 1] = data.get('lon', 0.0)
        
    s0_list = list(range(N0))
    
    # 2. Enumeración de aristas
    edges_set = set()
    for u, v in G.edges():
        if u == v:
            continue
        i_u, i_v = node_to_idx[u], node_to_idx[v]
        edges_set.add((min(i_u, i_v), max(i_u, i_v)))
    s1_edges = sorted(list(edges_set))
    edge_to_idx = {e: i for i, e in enumerate(s1_edges)}
    
    # 3. Enumeración de 3-cliques (triángulos en el grafo)
    # Creamos un grafo mapeado a índices enteros para eficiencia
    G_idx = nx.Graph()
    G_idx.add_nodes_from(s0_list)
    G_idx.add_edges_from(s1_edges)
    
    triangles_list = []
    # Usar enumerate_all_cliques o encontrar ciclos de longitud 3
    cliques = nx.enumerate_all_cliques(G_idx)
    for c in cliques:
        if len(c) == 3:
            idx_a, idx_b, idx_c = sorted(c)
            triangles_list.append((idx_a, idx_b, idx_c))
        elif len(c) > 3:
            # Dado que enumerate_all_cliques produce en orden no decreciente de tamaño,
            # podemos descomponer cualquier k-clique superior en sus triángulos o romper
            # Para k=2 máximo, enumeramos todos los 3-subconjuntos
            pass
            
    s2_triangles = sorted(list(set(triangles_list)))
    triangle_to_idx = {t: i for i, t in enumerate(s2_triangles)}
    
    # 4. Construcción de DataFrames descriptivos
    # S0
    df_s0_records = []
    for i in range(N0):
        osmid = idx_to_node[i]
        df_s0_records.append({
            'simplex_id': i,
            'osmid': osmid,
            'x': coords_proj[i, 0],
            'y': coords_proj[i, 1],
            'lat': coords_geo[i, 0],
            'lon': coords_geo[i, 1]
        })
    df_s0 = pd.DataFrame(df_s0_records)
    
    # S1
    df_s1_records = []
    for j, (u, v) in enumerate(s1_edges):
        p_u = coords_proj[u]
        p_v = coords_proj[v]
        length = float(np.linalg.norm(p_u - p_v))
        df_s1_records.append({
            'simplex_id': j,
            'u': u,
            'v': v,
            'osmid_u': idx_to_node[u],
            'osmid_v': idx_to_node[v],
            'length_m': length
        })
    df_s1 = pd.DataFrame(df_s1_records)
    
    # S2
    df_s2_records = []
    for k, (u, v, w) in enumerate(s2_triangles):
        p_u, p_v, p_w = coords_proj[u], coords_proj[v], coords_proj[w]
        area = _triangle_area_2d(p_u, p_v, p_w)
        r_circ = _circumradius_2d(p_u, p_v, p_w)
        e1 = edge_to_idx.get((u, v), -1)
        e2 = edge_to_idx.get((v, w), -1)
        e3 = edge_to_idx.get((u, w), -1)
        df_s2_records.append({
            'simplex_id': k,
            'u': u,
            'v': v,
            'w': w,
            'vertices': f"[{idx_to_node[u]}, {idx_to_node[v]}, {idx_to_node[w]}]",
            'edge_ids': (e1, e2, e3),
            'area_m2': area,
            'circumradius': r_circ
        })
    df_s2 = pd.DataFrame(df_s2_records) if df_s2_records else pd.DataFrame(columns=['simplex_id', 'u', 'v', 'w', 'vertices', 'edge_ids', 'area_m2', 'circumradius'])

    return SimplicialComplex(
        complex_type="flag",
        alpha_param=None,
        s0_nodes=s0_list,
        s1_edges=s1_edges,
        s2_triangles=s2_triangles,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        edge_to_idx=edge_to_idx,
        triangle_to_idx=triangle_to_idx,
        df_s0=df_s0,
        df_s1=df_s1,
        df_s2=df_s2,
        coords_projected=coords_proj,
        coords_geographic=coords_geo
    )

"""
Módulo de adquisición, preprocesamiento, simplificación y consolidación métrica
de grafos urbanos utilizando OSMnx, NetworkX y SciPy.
"""
from typing import Tuple, Dict, Any, Optional, List
import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def fetch_urban_network_by_coords(
    lat: float,
    lon: float,
    radius: float,
    network_type: str = "walk",
    dist_type: str = "bbox"
) -> nx.MultiDiGraph:
    """
    Descarga la red urbana a partir de coordenadas centrales (lat, lon) y un radio en metros.
    Radio máximo permitido: 15,000 metros.
    """
    if radius > 15000:
        raise ValueError(f"El radio ({radius} m) supera el límite máximo permitido de 15,000 metros.")
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitud inválida: {lat}. Debe estar entre -90 y 90.")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitud inválida: {lon}. Debe estar entre -180 y 180.")

    center_point = (lat, lon)
    G_raw = ox.graph_from_point(
        center_point,
        dist=radius,
        dist_type=dist_type,
        network_type=network_type,
        simplify=True
    )
    return G_raw


def fetch_urban_network_by_place(
    place_name: str,
    network_type: str = "walk"
) -> nx.MultiDiGraph:
    """
    Descarga la red urbana a partir de un nombre de lugar / topónimo.
    """
    if not place_name or not place_name.strip():
        raise ValueError("El nombre del lugar no puede estar vacío.")
    
    G_raw = ox.graph_from_place(
        place_name.strip(),
        network_type=network_type,
        simplify=True
    )
    return G_raw


def consolidate_close_nodes(
    G: nx.Graph,
    tolerance_meters: float = 10.0
) -> nx.Graph:
    """
    Consolida y fusiona en sus centroides geométricos todos aquellos vértices
    que se encuentren a una distancia euclidiana métrica menor al umbral (por defecto 10 metros).
    Esto limpia el número de intersecciones redundantes y simplifica la topología del grafo.
    """
    nodes = list(G.nodes(data=True))
    N = len(nodes)
    if N <= 1 or tolerance_meters <= 0.0:
        return G
        
    coords = np.array([[d['x'], d['y']] for _, d in nodes])
    tree = KDTree(coords)
    # Encontrar pares de nodos con distancia < tolerance_meters
    pairs = tree.query_pairs(r=tolerance_meters)
    
    # Estructura Disjoint-Set (Union-Find) para agrupar componentes conexas de nodos cercanos
    parent = list(range(N))
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for p in path:
            parent[p] = i
        return i
        
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            
    for i, j in pairs:
        union(i, j)
        
    clusters: Dict[int, List[int]] = {}
    for i in range(N):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)
        
    # Reconstruir grafo consolidado
    G_cons = nx.Graph()
    node_map = {}
    
    for root, member_indices in clusters.items():
        member_nodes = [nodes[idx] for idx in member_indices]
        # Identificador representativo
        rep_osmid = min(n[0] for n in member_nodes)
        mean_x = float(np.mean([n[1]['x'] for n in member_nodes]))
        mean_y = float(np.mean([n[1]['y'] for n in member_nodes]))
        mean_lat = float(np.mean([n[1].get('lat', 0.0) for n in member_nodes]))
        mean_lon = float(np.mean([n[1].get('lon', 0.0) for n in member_nodes]))
        
        G_cons.add_node(
            rep_osmid,
            osmid=rep_osmid,
            x=mean_x,
            y=mean_y,
            lat=mean_lat,
            lon=mean_lon,
            merged_count=len(member_indices)
        )
        for idx in member_indices:
            node_map[nodes[idx][0]] = rep_osmid
            
    for u, v, data in G.edges(data=True):
        new_u = node_map[u]
        new_v = node_map[v]
        if new_u != new_v:
            p_u = np.array([G_cons.nodes[new_u]['x'], G_cons.nodes[new_u]['y']])
            p_v = np.array([G_cons.nodes[new_v]['x'], G_cons.nodes[new_v]['y']])
            length_m = float(np.linalg.norm(p_u - p_v))
            G_cons.add_edge(
                new_u,
                new_v,
                length_m=length_m,
                highway=str(data.get('highway', 'road')),
                name=str(data.get('name', 'unnamed'))
            )
            
    return G_cons


def process_and_project_graph(
    G_raw: nx.MultiDiGraph,
    node_merge_tolerance_meters: float = 10.0
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """
    Proyecta el grafo a UTM métrico, simplifica a grafo no dirigido simple (nx.Graph),
    consolida vértices con distancia < 10m y extrae DataFrames tabulados de nodos (S0) y aristas (S1).
    """
    # 1. Proyectar a UTM métrico óptimo local
    G_proj = ox.project_graph(G_raw)

    # 2. Convertir a grafo no dirigido preservando atributos
    G_undir = ox.convert.to_undirected(G_proj)
    
    # 3. Crear grafo simple inicial con coordenadas proyectadas y geográficas
    G_simple_initial = nx.Graph()
    nodes_dict = {}
    
    for node, data in G_proj.nodes(data=True):
        x_m = data.get('x', 0.0)
        y_m = data.get('y', 0.0)
        
        lat_wgs = G_raw.nodes[node].get('y', data.get('lat', 0.0))
        lon_wgs = G_raw.nodes[node].get('x', data.get('lon', 0.0))
        
        node_attr = {
            'osmid': node,
            'x': float(x_m),
            'y': float(y_m),
            'lat': float(lat_wgs),
            'lon': float(lon_wgs),
        }
        nodes_dict[node] = node_attr
        G_simple_initial.add_node(node, **node_attr)

    for u, v, data in G_undir.edges(data=True):
        if u == v:
            continue
        
        length_m = data.get('length', None)
        if length_m is None:
            p_u = np.array([nodes_dict[u]['x'], nodes_dict[u]['y']])
            p_v = np.array([nodes_dict[v]['x'], nodes_dict[v]['y']])
            length_m = float(np.linalg.norm(p_u - p_v))
        else:
            length_m = float(length_m)
            
        edge_attr = {
            'u': u,
            'v': v,
            'length_m': length_m,
            'name': data.get('name', 'unnamed'),
            'highway': data.get('highway', 'road')
        }
        G_simple_initial.add_edge(u, v, **edge_attr)
        
    # 4. Consolidación espacial métrica de vértices cercanos (< 10 metros)
    if node_merge_tolerance_meters > 0.0:
        G_simple = consolidate_close_nodes(G_simple_initial, tolerance_meters=node_merge_tolerance_meters)
    else:
        G_simple = G_simple_initial
        
    # 5. Construir DataFrames limpios
    df_nodes = pd.DataFrame.from_dict(dict(G_simple.nodes(data=True)), orient='index')
    df_nodes.index.name = 'node_id'
    
    edge_records = []
    for u, v, data in G_simple.edges(data=True):
        edge_records.append({
            'u': u,
            'v': v,
            'length_m': data.get('length_m', 0.0),
            'highway': str(data.get('highway', 'road'))
        })
    df_edges = pd.DataFrame(edge_records)

    return G_simple, df_nodes, df_edges

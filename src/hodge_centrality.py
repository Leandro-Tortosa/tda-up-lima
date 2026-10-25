"""
Módulo de Álgebra Lineal Topológica, Operadores de Hodge y Centralidades Simpliciales.
Manejo de matrices dispersas (scipy.sparse) y algoritmos espectrales.
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import networkx as nx
from src.simplicial import SimplicialComplex


@dataclass
class HodgeOperators:
    """Almacena los operadores frontera y matrices del Laplaciano de Hodge."""
    I1: sp.csr_matrix  # (N0 x N1) Incidencia no orientada nodo-arista
    I2: sp.csr_matrix  # (N0 x N2) Incidencia no orientada nodo-triangulo
    B1: sp.csr_matrix  # (N0 x N1) Operador frontera orientada de aristas
    B2: sp.csr_matrix  # (N1 x N2) Operador frontera orientada de triángulos
    
    L0: sp.csr_matrix          # (N0 x N0) Laplaciano de nodos B1 * B1^T
    L1_down: sp.csr_matrix     # (N1 x N1) Laplaciano inferior B1^T * B1
    L1_up: sp.csr_matrix       # (N1 x N1) Laplaciano superior B2 * B2^T
    L1: sp.csr_matrix          # (N1 x N1) Laplaciano de Hodge completo L1_down + L1_up
    
    chain_property_norm: float  # ||B1 * B2||_inf (debe ser 0.0)
    betti_0: int                # Número de componentes conexas
    betti_1: int                # Número de 1-ciclos no triangulados (agujeros)
    
    A1_down: sp.csr_matrix      # Adyacencia simplicial inferior
    A1_up: sp.csr_matrix        # Adyacencia simplicial superior
    A1_total: sp.csr_matrix     # Adyacencia simplicial combinada


@dataclass
class SimplicialCentralities:
    """Almacena DataFrames con todas las métricas de centralidad calculadas para k=0, k=1 y k=2."""
    df_centrality_s0: pd.DataFrame  # Nodos
    df_centrality_s1: pd.DataFrame  # Aristas
    df_centrality_s2: pd.DataFrame  # Triángulos


def compute_hodge_operators(sc: SimplicialComplex) -> HodgeOperators:
    """
    Construye las matrices de incidencia I1, I2, operadores frontera B1, B2,
    y los Laplacianos de Hodge L0, L1_down, L1_up, L1.
    """
    N0 = len(sc.s0_nodes)
    N1 = len(sc.s1_edges)
    N2 = len(sc.s2_triangles)
    
    # --- 1. Matriz de Incidencia I1 y Operador Frontera B1 (N0 x N1) ---
    if N1 > 0:
        rows_i1, cols_i1, data_i1 = [], [], []
        rows_b1, cols_b1, data_b1 = [], [], []
        
        for j, (u, v) in enumerate(sc.s1_edges):
            # u < v por construcción canónica
            # I1 (no orientado)
            rows_i1.extend([u, v])
            cols_i1.extend([j, j])
            data_i1.extend([1.0, 1.0])
            
            # B1 (orientado: -1 en u, +1 en v)
            rows_b1.extend([u, v])
            cols_b1.extend([j, j])
            data_b1.extend([-1.0, 1.0])
            
        I1 = sp.csr_matrix((data_i1, (rows_i1, cols_i1)), shape=(N0, N1), dtype=np.float64)
        B1 = sp.csr_matrix((data_b1, (rows_b1, cols_b1)), shape=(N0, N1), dtype=np.float64)
    else:
        I1 = sp.csr_matrix((N0, 0), dtype=np.float64)
        B1 = sp.csr_matrix((N0, 0), dtype=np.float64)
        
    # --- 2. Matriz de Incidencia I2 y Operador Frontera B2 (N1 x N2) ---
    if N2 > 0:
        rows_i2, cols_i2, data_i2 = [], [], []
        rows_b2, cols_b2, data_b2 = [], [], []
        
        for k, (u, v, w) in enumerate(sc.s2_triangles):
            # u < v < w
            # Incidencia nodo-triángulo I2 (N0 x N2)
            rows_i2.extend([u, v, w])
            cols_i2.extend([k, k, k])
            data_i2.extend([1.0, 1.0, 1.0])
            
            # Frontera orientada B2 (N1 x N2):
            # \partial_2([u, v, w]) = + [v, w] - [u, w] + [u, v]
            e_bc = sc.edge_to_idx.get((v, w))
            e_ac = sc.edge_to_idx.get((u, w))
            e_ab = sc.edge_to_idx.get((u, v))
            
            if e_bc is not None:
                rows_b2.append(e_bc)
                cols_b2.append(k)
                data_b2.append(1.0)
            if e_ac is not None:
                rows_b2.append(e_ac)
                cols_b2.append(k)
                data_b2.append(-1.0)
            if e_ab is not None:
                rows_b2.append(e_ab)
                cols_b2.append(k)
                data_b2.append(1.0)
                
        I2 = sp.csr_matrix((data_i2, (rows_i2, cols_i2)), shape=(N0, N2), dtype=np.float64)
        B2 = sp.csr_matrix((data_b2, (rows_b2, cols_b2)), shape=(N1, N2), dtype=np.float64)
    else:
        I2 = sp.csr_matrix((N0, 0), dtype=np.float64)
        B2 = sp.csr_matrix((N1, 0), dtype=np.float64)
        
    # --- 3. Verificación de la Propiedad Homológica B1 * B2 = 0 ---
    if N1 > 0 and N2 > 0:
        B1_B2 = B1.dot(B2)
        if B1_B2.nnz > 0:
            chain_norm = float(np.abs(B1_B2.data).max())
        else:
            chain_norm = 0.0
    else:
        chain_norm = 0.0
        
    # --- 4. Laplacianos de Hodge ---
    # L0 = B1 * B1^T (N0 x N0)
    L0 = B1.dot(B1.T)
    
    # L1_down = B1^T * B1 (N1 x N1)
    if N1 > 0:
        L1_down = B1.T.dot(B1)
        if N2 > 0:
            L1_up = B2.dot(B2.T)
        else:
            L1_up = sp.csr_matrix((N1, N1), dtype=np.float64)
        L1 = L1_down + L1_up
    else:
        L1_down = sp.csr_matrix((0, 0), dtype=np.float64)
        L1_up = sp.csr_matrix((0, 0), dtype=np.float64)
        L1 = sp.csr_matrix((0, 0), dtype=np.float64)
        
    # --- 5. Matrices de Adyacencia Simplicial ---
    if N1 > 0:
        # A1_down = |L1_down| - diag(|L1_down|)
        abs_down = sp.csr_matrix(np.abs(L1_down.toarray()))
        A1_down = abs_down - sp.diags(abs_down.diagonal())
        A1_down.eliminate_zeros()
        
        # A1_up = |L1_up| - diag(|L1_up|)
        abs_up = sp.csr_matrix(np.abs(L1_up.toarray()))
        A1_up = abs_up - sp.diags(abs_up.diagonal())
        A1_up.eliminate_zeros()
        
        A1_total = A1_down + A1_up
    else:
        A1_down = sp.csr_matrix((0, 0), dtype=np.float64)
        A1_up = sp.csr_matrix((0, 0), dtype=np.float64)
        A1_total = sp.csr_matrix((0, 0), dtype=np.float64)
        
    # --- 6. Números de Betti (Topología Algebraica) ---
    # beta_0 = dim(ker(L0)) = N0 - rank(B1)
    # beta_1 = dim(ker(L1)) = dim(ker(B1)) - rank(B2) = (N1 - rank(B1)) - rank(B2)
    if N0 > 0:
        rank_B1 = int(np.linalg.matrix_rank(B1.toarray())) if N1 > 0 else 0
        rank_B2 = int(np.linalg.matrix_rank(B2.toarray())) if N2 > 0 else 0
        betti_0 = max(1, N0 - rank_B1)
        betti_1 = max(0, (N1 - rank_B1) - rank_B2)
    else:
        betti_0 = 0
        betti_1 = 0

    return HodgeOperators(
        I1=I1,
        I2=I2,
        B1=B1,
        B2=B2,
        L0=L0,
        L1_down=L1_down,
        L1_up=L1_up,
        L1=L1,
        chain_property_norm=chain_norm,
        betti_0=betti_0,
        betti_1=betti_1,
        A1_down=A1_down,
        A1_up=A1_up,
        A1_total=A1_total
    )


def _compute_leading_eigenvector(A: sp.csr_matrix) -> np.ndarray:
    """
    Calcula el autovector dominante no negativo normalizado para una matriz de adyacencia simplicial.
    Usa el método de las potencias o eigs de Arnoldi con fallback robusto.
    """
    n = A.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
        
    # Si la matriz es estrictamente nula
    if A.nnz == 0:
        return np.ones(n) / np.sqrt(n)
        
    # Método iterativo de las potencias con arranque uniforme
    v = np.ones(n, dtype=float) / np.sqrt(n)
    for _ in range(100):
        v_next = A.dot(v)
        norm = np.linalg.norm(v_next)
        if norm < 1e-12:
            break
        v_next = v_next / norm
        if np.linalg.norm(v_next - v) < 1e-7:
            v = v_next
            break
        v = v_next
        
    # Asegurar signo no negativo
    if np.sum(v) < 0:
        v = -v
    v = np.maximum(v, 0.0)
    
    # Normalizar al rango [0, 1] para visualización y análisis
    max_val = np.max(v)
    if max_val > 1e-12:
        v_norm = v / max_val
    else:
        v_norm = v
    return v_norm


def compute_simplicial_centralities(
    sc: SimplicialComplex,
    hodge: HodgeOperators
) -> SimplicialCentralities:
    """
    Calcula las tres familias de centralidades simpliciales:
    1. Simplicial Degree Centrality
    2. Simplicial Betweenness Centrality
    3. Simplicial Eigenvector Centrality
    para Nodos (k=0), Aristas (k=1) y Triángulos (k=2).
    """
    N0 = len(sc.s0_nodes)
    N1 = len(sc.s1_edges)
    N2 = len(sc.s2_triangles)
    
    # =========================================================================
    # 1. SIMPLICIAL DEGREE CENTRALITY
    # =========================================================================
    # 0-símplices (Nodos)
    if N1 > 0:
        deg_k1 = np.array(hodge.I1.sum(axis=1)).flatten()  # Grado de aristas
    else:
        deg_k1 = np.zeros(N0)
        
    if N2 > 0:
        deg_k2 = np.array(hodge.I2.sum(axis=1)).flatten()  # Grado de triángulos
    else:
        deg_k2 = np.zeros(N0)
        
    # Normalización combinatoria
    norm_deg_k1 = deg_k1 / max(1, (N0 - 1))
    norm_deg_k2 = deg_k2 / max(1, ((N0 - 1) * (N0 - 2) / 2.0)) if N0 > 2 else np.zeros(N0)
    
    # 1-símplices (Aristas)
    if N2 > 0 and N1 > 0:
        # Número de 2-símplices que comparten la arista (grado superior)
        # Suma de entradas absolutas a lo largo de las filas de B2
        edge_upper_deg = np.array(np.abs(hodge.B2).sum(axis=1)).flatten()
    else:
        edge_upper_deg = np.zeros(N1)
    norm_edge_deg = edge_upper_deg / max(1, (N0 - 2)) if N0 > 2 else np.zeros(N1)
    
    # 2-símplices (Triángulos)
    # Grado dual: número de triángulos que comparten al menos una arista
    if N2 > 0:
        A2_down = hodge.B2.T.dot(hodge.B2)
        tri_adj = sp.csr_matrix(np.abs(A2_down.toarray())) - sp.diags(A2_down.diagonal())
        tri_deg = np.array(tri_adj.sum(axis=1)).flatten()
        norm_tri_deg = tri_deg / 3.0  # Máximo 3 triángulos adyacentes por sus 3 caras
    else:
        tri_deg = np.zeros(0)
        norm_tri_deg = np.zeros(0)
        
    # =========================================================================
    # 2. SIMPLICIAL BETWEENNESS CENTRALITY
    # =========================================================================
    # Grafo base para betweenness de red
    G_base = nx.Graph()
    G_base.add_nodes_from(range(N0))
    G_base.add_edges_from(sc.s1_edges)
    
    # Betweenness de nodos y aristas clásico
    node_betw_dict = nx.betweenness_centrality(G_base, normalized=True)
    node_betw_raw = np.array([node_betw_dict.get(i, 0.0) for i in range(N0)])
    
    if N1 > 0:
        edge_betw_dict = nx.edge_betweenness_centrality(G_base, normalized=True)
        edge_betw_raw = np.zeros(N1)
        for j, (u, v) in enumerate(sc.s1_edges):
            edge_betw_raw[j] = edge_betw_dict.get((u, v), edge_betw_dict.get((v, u), 0.0))
    else:
        edge_betw_raw = np.zeros(0)
        
    # Betweenness para 2-símplices (dual graph de triángulos adyacentes por caras compartidas)
    if N2 > 0:
        G_tri_dual = nx.Graph()
        G_tri_dual.add_nodes_from(range(N2))
        for k1 in range(N2):
            for k2 in range(k1 + 1, N2):
                t1 = set(sc.s2_triangles[k1])
                t2 = set(sc.s2_triangles[k2])
                if len(t1.intersection(t2)) == 2:  # Comparten una arista
                    G_tri_dual.add_edge(k1, k2)
        tri_betw_dict = nx.betweenness_centrality(G_tri_dual, normalized=True)
        tri_betw_raw = np.array([tri_betw_dict.get(k, 0.0) for k in range(N2)])
    else:
        tri_betw_raw = np.zeros(0)
        
    # Agregación simplicial de betweenness a nodos
    if N2 > 0 and N0 > 0:
        node_tri_betw_agg = np.array(hodge.I2.dot(tri_betw_raw)).flatten()
        if np.max(node_tri_betw_agg) > 1e-12:
            node_tri_betw_agg = node_tri_betw_agg / np.max(node_tri_betw_agg)
    else:
        node_tri_betw_agg = np.zeros(N0)
        
    # =========================================================================
    # 3. SIMPLICIAL EIGENVECTOR CENTRALITY
    # =========================================================================
    # 1-símplices: autovectores de A1_up, A1_down y A1_total
    eig_s1_down = _compute_leading_eigenvector(hodge.A1_down) if N1 > 0 else np.zeros(0)
    eig_s1_up = _compute_leading_eigenvector(hodge.A1_up) if N1 > 0 else np.zeros(0)
    eig_s1_total = _compute_leading_eigenvector(hodge.A1_total) if N1 > 0 else np.zeros(0)
    
    # 2-símplices: autovector de adyacencia de triángulos
    if N2 > 0:
        A2_down = hodge.B2.T.dot(hodge.B2)
        A2_adj = sp.csr_matrix(np.abs(A2_down.toarray())) - sp.diags(A2_down.diagonal())
        A2_adj.eliminate_zeros()
        eig_s2 = _compute_leading_eigenvector(A2_adj)
    else:
        eig_s2 = np.zeros(0)
        
    # 0-símplices: proyección ponderada desde aristas (A1_total) y Laplaciano L0
    if N0 > 0 and N1 > 0:
        # Autovector de adyacencia de nodos estándar
        A0 = sp.csr_matrix(np.abs(hodge.L0.toarray())) - sp.diags(hodge.L0.diagonal())
        eig_s0_base = _compute_leading_eigenvector(A0)
        # Proyección simplicial agregada de aristas
        eig_s0_from_s1 = np.array(hodge.I1.dot(eig_s1_total)).flatten()
        if np.max(eig_s0_from_s1) > 1e-12:
            eig_s0_from_s1 = eig_s0_from_s1 / np.max(eig_s0_from_s1)
    else:
        eig_s0_base = np.zeros(N0)
        eig_s0_from_s1 = np.zeros(N0)
        
    # =========================================================================
    # CONSTRUCCIÓN DE DATAFRAMES RESULTANTES
    # =========================================================================
    # S0 (Nodos)
    df_s0 = sc.df_s0.copy()
    df_s0['dimension_k'] = 0
    df_s0['degree_k1_edges'] = deg_k1
    df_s0['degree_k2_triangles'] = deg_k2
    df_s0['degree_normalized'] = norm_deg_k1
    df_s0['degree_triangles_normalized'] = norm_deg_k2
    df_s0['betweenness_score'] = node_betw_raw
    df_s0['betweenness_triangles_agg'] = node_tri_betw_agg
    df_s0['eigenvector_node'] = eig_s0_base
    df_s0['eigenvector_simplicial_proj'] = eig_s0_from_s1
    
    # S1 (Aristas)
    df_s1 = sc.df_s1.copy()
    df_s1['dimension_k'] = 1
    df_s1['vertices'] = [f"[{sc.idx_to_node[u]}, {sc.idx_to_node[v]}]" for u, v in sc.s1_edges]
    df_s1['degree_upper'] = edge_upper_deg
    df_s1['degree_normalized'] = norm_edge_deg
    df_s1['betweenness_score'] = edge_betw_raw
    df_s1['eigenvector_lower'] = eig_s1_down
    df_s1['eigenvector_upper'] = eig_s1_up
    df_s1['eigenvector_total'] = eig_s1_total
    
    # S2 (Triángulos)
    df_s2 = sc.df_s2.copy()
    if N2 > 0:
        df_s2['dimension_k'] = 2
        df_s2['degree_adjacent'] = tri_deg
        df_s2['degree_normalized'] = norm_tri_deg
        df_s2['betweenness_score'] = tri_betw_raw
        df_s2['eigenvector_score'] = eig_s2
    else:
        df_s2['dimension_k'] = []
        df_s2['degree_adjacent'] = []
        df_s2['degree_normalized'] = []
        df_s2['betweenness_score'] = []
        df_s2['eigenvector_score'] = []

    return SimplicialCentralities(
        df_centrality_s0=df_s0,
        df_centrality_s1=df_s1,
        df_centrality_s2=df_s2
    )

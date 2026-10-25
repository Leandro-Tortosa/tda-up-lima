"""
Módulo de exportación de datos científicos y matrices de Hodge en formatos CSV y ZIP.
"""
import io
import zipfile
import pandas as pd
import numpy as np
import scipy.sparse as sp
from src.hodge_centrality import HodgeOperators, SimplicialCentralities
from src.simplicial import SimplicialComplex


def sparse_to_csv_bytes(matrix: sp.csr_matrix, row_prefix: str = "r", col_prefix: str = "c") -> bytes:
    """Convierte una matriz dispersa a un archivo CSV en memoria en bytes UTF-8."""
    df = pd.DataFrame(matrix.toarray())
    df.index = [f"{row_prefix}{i}" for i in range(matrix.shape[0])]
    df.columns = [f"{col_prefix}{j}" for j in range(matrix.shape[1])]
    
    buf = io.StringIO()
    df.to_csv(buf, index=True, encoding='utf-8')
    return buf.getvalue().encode('utf-8')


def build_hodge_matrices_zip(hodge: HodgeOperators) -> bytes:
    """
    Empaqueta en un archivo ZIP en memoria los 6 archivos CSV con las matrices fundamentales de Hodge:
    - B1.csv
    - B2.csv
    - L0.csv
    - L1_down.csv
    - L1_up.csv
    - L1.csv
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # B1 (N0 x N1)
        zip_file.writestr("B1_boundary_1.csv", sparse_to_csv_bytes(hodge.B1, row_prefix="node_", col_prefix="edge_"))
        
        # B2 (N1 x N2)
        zip_file.writestr("B2_boundary_2.csv", sparse_to_csv_bytes(hodge.B2, row_prefix="edge_", col_prefix="tri_"))
        
        # L0 (N0 x N0)
        zip_file.writestr("L0_node_laplacian.csv", sparse_to_csv_bytes(hodge.L0, row_prefix="node_", col_prefix="node_"))
        
        # L1_down (N1 x N1)
        zip_file.writestr("L1_down_edge_laplacian.csv", sparse_to_csv_bytes(hodge.L1_down, row_prefix="edge_", col_prefix="edge_"))
        
        # L1_up (N1 x N1)
        zip_file.writestr("L1_up_edge_laplacian.csv", sparse_to_csv_bytes(hodge.L1_up, row_prefix="edge_", col_prefix="edge_"))
        
        # L1 (N1 x N1)
        zip_file.writestr("L1_full_hodge_laplacian.csv", sparse_to_csv_bytes(hodge.L1, row_prefix="edge_", col_prefix="edge_"))
        
        # Metadatos del complejo
        meta_txt = (
            f"TDA Urban Simplicial Complex - Hodge Decomposition Summary\n"
            f"Chain Property Norm ||B1 * B2||_inf: {hodge.chain_property_norm:.10e}\n"
            f"Betti 0 (Connected Components): {hodge.betti_0}\n"
            f"Betti 1 (Non-triangulated 1-cycles / Holes): {hodge.betti_1}\n"
            f"Dimensions: N0={hodge.B1.shape[0]}, N1={hodge.B1.shape[1]}, N2={hodge.B2.shape[1]}\n"
        )
        zip_file.writestr("README_METADATA.txt", meta_txt.encode('utf-8'))
        
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def get_degree_centrality_export_csv(sc: SimplicialComplex, centralities: SimplicialCentralities) -> str:
    """Genera la tabla unificada de Simplicial Degree Centrality en formato CSV."""
    rows = []
    
    # k=0 Nodos
    for _, r in centralities.df_centrality_s0.iterrows():
        rows.append({
            'simplex_id': f"node_{int(r['simplex_id'])}",
            'dimension_k': 0,
            'vertices': f"[{int(r['osmid'])}]",
            'degree_raw_k1': r['degree_k1_edges'],
            'degree_raw_k2': r['degree_k2_triangles'],
            'degree_normalized': r['degree_normalized']
        })
        
    # k=1 Aristas
    for _, r in centralities.df_centrality_s1.iterrows():
        rows.append({
            'simplex_id': f"edge_{int(r['simplex_id'])}",
            'dimension_k': 1,
            'vertices': r['vertices'],
            'degree_raw_k1': 0,
            'degree_raw_k2': r['degree_upper'],
            'degree_normalized': r['degree_normalized']
        })
        
    # k=2 Triángulos
    for _, r in centralities.df_centrality_s2.iterrows():
        rows.append({
            'simplex_id': f"tri_{int(r['simplex_id'])}",
            'dimension_k': 2,
            'vertices': r['vertices'],
            'degree_raw_k1': 0,
            'degree_raw_k2': r['degree_adjacent'],
            'degree_normalized': r['degree_normalized']
        })
        
    df = pd.DataFrame(rows)
    return df.to_csv(index=False, encoding='utf-8')


def get_betweenness_centrality_export_csv(sc: SimplicialComplex, centralities: SimplicialCentralities) -> str:
    """Genera la tabla unificada de Simplicial Betweenness Centrality en formato CSV."""
    rows = []
    
    # k=0 Nodos
    for _, r in centralities.df_centrality_s0.iterrows():
        rows.append({
            'simplex_id': f"node_{int(r['simplex_id'])}",
            'dimension_k': 0,
            'vertices': f"[{int(r['osmid'])}]",
            'betweenness_score': r['betweenness_score'],
            'betweenness_triangles_agg': r['betweenness_triangles_agg']
        })
        
    # k=1 Aristas
    for _, r in centralities.df_centrality_s1.iterrows():
        rows.append({
            'simplex_id': f"edge_{int(r['simplex_id'])}",
            'dimension_k': 1,
            'vertices': r['vertices'],
            'betweenness_score': r['betweenness_score'],
            'betweenness_triangles_agg': 0.0
        })
        
    # k=2 Triángulos
    for _, r in centralities.df_centrality_s2.iterrows():
        rows.append({
            'simplex_id': f"tri_{int(r['simplex_id'])}",
            'dimension_k': 2,
            'vertices': r['vertices'],
            'betweenness_score': r['betweenness_score'],
            'betweenness_triangles_agg': 0.0
        })
        
    df = pd.DataFrame(rows)
    return df.to_csv(index=False, encoding='utf-8')


def get_eigenvector_centrality_export_csv(sc: SimplicialComplex, centralities: SimplicialCentralities) -> str:
    """Genera la tabla unificada de Simplicial Eigenvector Centrality en formato CSV."""
    rows = []
    
    # k=0 Nodos
    for _, r in centralities.df_centrality_s0.iterrows():
        rows.append({
            'simplex_id': f"node_{int(r['simplex_id'])}",
            'dimension_k': 0,
            'vertices': f"[{int(r['osmid'])}]",
            'eigenvector_lower': 0.0,
            'eigenvector_upper': 0.0,
            'eigenvector_total': r['eigenvector_node'],
            'eigenvector_simplicial_proj': r['eigenvector_simplicial_proj']
        })
        
    # k=1 Aristas
    for _, r in centralities.df_centrality_s1.iterrows():
        rows.append({
            'simplex_id': f"edge_{int(r['simplex_id'])}",
            'dimension_k': 1,
            'vertices': r['vertices'],
            'eigenvector_lower': r['eigenvector_lower'],
            'eigenvector_upper': r['eigenvector_upper'],
            'eigenvector_total': r['eigenvector_total'],
            'eigenvector_simplicial_proj': 0.0
        })
        
    # k=2 Triángulos
    for _, r in centralities.df_centrality_s2.iterrows():
        rows.append({
            'simplex_id': f"tri_{int(r['simplex_id'])}",
            'dimension_k': 2,
            'vertices': r['vertices'],
            'eigenvector_lower': 0.0,
            'eigenvector_upper': 0.0,
            'eigenvector_total': r['eigenvector_score'],
            'eigenvector_simplicial_proj': 0.0
        })
        
    df = pd.DataFrame(rows)
    return df.to_csv(index=False, encoding='utf-8')

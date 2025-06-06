import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import scanpy as sc
import anndata as ad
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import h5py
import json

class XeniumProcessor:
    def __init__(self):
        """
        Processor for Xenium spatial transcriptomics data
        Handles integration with scRNA-seq and tissue morphology
        """
        # Configure scanpy for spatial data
        sc.settings.verbosity = 3
        sc.settings.set_figure_params(dpi=80, facecolor='white')
        
        self.adata = None
        self.cell_metadata = None
        self.transcripts = None
        self.tissue_image = None
        
    def load_xenium_data(self, data_path):
        """
        Load Xenium data from the standard output directory
        
        Args:
            data_path: Path to Xenium output directory
            
        Expected files:
            - cell_feature_matrix.h5: Gene expression per cell
            - cells.parquet: Cell metadata with spatial coordinates
            - transcripts.parquet: Individual transcript locations
            - morphology.ome.tif: High-resolution tissue image
        """
        import os
        print(f"Loading Xenium data from: {data_path}")
        
        # Load cell x gene expression matrix
        matrix_path = os.path.join(data_path, "cell_feature_matrix.h5")
        if os.path.exists(matrix_path):
            self.adata = sc.read_10x_h5(matrix_path)
            self.adata.var_names_make_unique()
            print(f"Loaded expression matrix: {self.adata.shape}")
        else:
            raise FileNotFoundError(f"Cell feature matrix not found at {matrix_path}")
        
        # Load cell metadata with spatial coordinates
        cells_path = os.path.join(data_path, "cells.parquet")
        if os.path.exists(cells_path):
            self.cell_metadata = pd.read_parquet(cells_path)
            print(f"Loaded cell metadata: {self.cell_metadata.shape}")
            print(f"Spatial coordinates available: {'x_centroid' in self.cell_metadata.columns}")
        else:
            print("Warning: Cell metadata not found, will use cell IDs only")
        
        # Load transcript data
        transcripts_path = os.path.join(data_path, "transcripts.parquet")
        if os.path.exists(transcripts_path):
            self.transcripts = pd.read_parquet(transcripts_path)
            print(f"Loaded transcripts: {self.transcripts.shape}")
        else:
            print("Warning: Transcripts data not found")
        
        # Load tissue morphology image
        image_path = os.path.join(data_path, "morphology.ome.tif")
        if os.path.exists(image_path):
            # Note: For OME-TIF files, you might need specialized libraries
            # For now, we'll handle this as a placeholder
            self.tissue_image_path = image_path
            print(f"Found tissue image: {image_path}")
        else:
            print("Warning: Tissue morphology image not found")
        
        return self.adata, self.cell_metadata
    
    def integrate_spatial_coordinates(self):
        """
        Integrate spatial coordinates into the AnnData object
        """
        if self.cell_metadata is not None and 'x_centroid' in self.cell_metadata.columns:
            # Ensure cell order matches
            cell_order = [f"cell_{i}" for i in range(len(self.adata.obs))]
            
            # Add spatial coordinates to adata.obsm
            spatial_coords = self.cell_metadata[['x_centroid', 'y_centroid']].values
            self.adata.obsm['spatial'] = spatial_coords
            
            # Add cell areas if available
            if 'cell_area' in self.cell_metadata.columns:
                self.adata.obs['cell_area'] = self.cell_metadata['cell_area'].values
            
            print(f"Added spatial coordinates for {len(spatial_coords)} cells")
            print(f"Spatial range: X({spatial_coords[:, 0].min():.1f}, {spatial_coords[:, 0].max():.1f}), "
                  f"Y({spatial_coords[:, 1].min():.1f}, {spatial_coords[:, 1].max():.1f})")
        else:
            print("Warning: No spatial coordinates available")
    
    def preprocess_spatial_data(self, min_genes_per_cell=10, min_cells_per_gene=5):
        """
        Preprocess Xenium spatial data
        Less aggressive filtering than scRNA-seq due to targeted gene panels
        """
        print("Preprocessing spatial transcriptomics data...")
        
        # Calculate QC metrics
        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(self.adata, percent_top=None, log1p=False, inplace=True)
        
        print(f"Before filtering: {self.adata.shape}")
        
        # Filter cells and genes (less aggressive for spatial data)
        sc.pp.filter_cells(self.adata, min_genes=min_genes_per_cell)
        sc.pp.filter_genes(self.adata, min_cells=min_cells_per_gene)
        
        print(f"After filtering: {self.adata.shape}")
        
        # Normalize and log transform
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        # For spatial data, we often keep all genes (no HVG selection)
        # since we have a targeted panel
        
        # Scale data
        sc.pp.scale(self.adata, max_value=10)
        
        return self.adata
    
    def create_spatial_graph(self, radius=50, k_neighbors=6, method='radius'):
        """
        Create spatial neighborhood graph based on physical distance
        
        Args:
            radius: Distance threshold for radius-based neighbors
            k_neighbors: Number of neighbors for k-NN
            method: 'radius', 'knn', or 'delaunay'
        """
        if 'spatial' not in self.adata.obsm:
            raise ValueError("No spatial coordinates available. Run integrate_spatial_coordinates() first.")
        
        print(f"Creating spatial graph using {method} method...")
        
        spatial_coords = self.adata.obsm['spatial']
        n_cells = len(spatial_coords)
        
        if method == 'radius':
            # Create graph based on distance threshold
            from sklearn.neighbors import radius_neighbors_graph
            A = radius_neighbors_graph(spatial_coords, radius=radius, mode='connectivity')
            
        elif method == 'knn':
            # Create k-nearest neighbors graph
            A = kneighbors_graph(spatial_coords, n_neighbors=k_neighbors, mode='connectivity')
            
        elif method == 'delaunay':
            # Create Delaunay triangulation (natural neighbors)
            from scipy.spatial import Delaunay
            tri = Delaunay(spatial_coords)
            # Convert triangulation to adjacency matrix
            A = np.zeros((n_cells, n_cells))
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        A[simplex[i], simplex[j]] = 1
                        A[simplex[j], simplex[i]] = 1
            from scipy.sparse import csr_matrix
            A = csr_matrix(A)
        
        # Convert to edge list
        edge_index = []
        coo = A.tocoo()
        for i, j in zip(coo.row, coo.col):
            edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        print(f"Created spatial graph with {len(edge_index[0])} edges")
        print(f"Average neighbors per cell: {len(edge_index[0]) / n_cells:.1f}")
        
        return edge_index
    
    def create_multimodal_features(self, use_spatial=True, use_morphology=False):
        """
        Create multi-modal features combining gene expression and spatial information
        """
        features = []
        feature_names = []
        
        # Gene expression features
        if hasattr(self.adata, 'X'):
            gene_expr = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X
            features.append(gene_expr)
            feature_names.extend([f"gene_{name}" for name in self.adata.var_names])
        
        # Spatial coordinate features
        if use_spatial and 'spatial' in self.adata.obsm:
            spatial_coords = self.adata.obsm['spatial']
            # Normalize spatial coordinates
            spatial_coords_norm = StandardScaler().fit_transform(spatial_coords)
            features.append(spatial_coords_norm)
            feature_names.extend(['spatial_x', 'spatial_y'])
        
        # Cell morphology features
        if 'cell_area' in self.adata.obs:
            cell_area = self.adata.obs['cell_area'].values.reshape(-1, 1)
            cell_area_norm = StandardScaler().fit_transform(cell_area)
            features.append(cell_area_norm)
            feature_names.extend(['cell_area'])
        
        # Combine all features
        if features:
            combined_features = np.hstack(features)
            print(f"Created multi-modal features: {combined_features.shape}")
            print(f"Feature types: {len(feature_names)} total")
            return combined_features, feature_names
        else:
            raise ValueError("No features available")
    
    def create_spatial_torch_data(self, graph_method='radius', radius=50, k_neighbors=6):
        """
        Create PyTorch Geometric Data object with spatial graph
        """
        # Create spatial graph
        edge_index = self.create_spatial_graph(
            radius=radius, 
            k_neighbors=k_neighbors, 
            method=graph_method
        )
        
        # Create multi-modal features
        features, feature_names = self.create_multimodal_features()
        
        # Convert to torch tensors
        x = torch.tensor(features, dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index)
        
        # Add spatial coordinates for visualization
        if 'spatial' in self.adata.obsm:
            data.pos = torch.tensor(self.adata.obsm['spatial'], dtype=torch.float)
        
        print(f"Created spatial graph data: {data}")
        
        return data, feature_names
    
    def visualize_spatial_data(self, color_by='total_counts', figsize=(12, 8)):
        """
        Visualize cells in spatial context
        """
        if 'spatial' not in self.adata.obsm:
            print("No spatial coordinates available for visualization")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Spatial distribution of cells
        spatial_coords = self.adata.obsm['spatial']
        axes[0].scatter(spatial_coords[:, 0], spatial_coords[:, 1], 
                       s=1, alpha=0.6, c='blue')
        axes[0].set_title('Spatial Distribution of Cells')
        axes[0].set_xlabel('X coordinate')
        axes[0].set_ylabel('Y coordinate')
        axes[0].set_aspect('equal')
        
        # Plot 2: Colored by expression/metadata
        if color_by in self.adata.obs.columns:
            color_values = self.adata.obs[color_by]
        else:
            color_values = self.adata.obs['total_counts']
        
        scatter = axes[1].scatter(spatial_coords[:, 0], spatial_coords[:, 1], 
                                 c=color_values, s=1, alpha=0.6, cmap='viridis')
        axes[1].set_title(f'Cells colored by {color_by}')
        axes[1].set_xlabel('X coordinate')
        axes[1].set_ylabel('Y coordinate')
        axes[1].set_aspect('equal')
        plt.colorbar(scatter, ax=axes[1])
        
        plt.tight_layout()
        return fig
    
    def integrate_with_scrna(self, scrna_adata):
        """
        Integrate Xenium spatial data with scRNA-seq reference
        """
        print("Integrating Xenium with scRNA-seq reference...")
        
        # Find common genes
        common_genes = list(set(self.adata.var_names) & set(scrna_adata.var_names))
        print(f"Found {len(common_genes)} common genes")
        
        # Subset to common genes
        xenium_subset = self.adata[:, common_genes]
        scrna_subset = scrna_adata[:, common_genes]
        
        # Combine datasets for integration
        combined = ad.concat([xenium_subset, scrna_subset], 
                            label='dataset', 
                            keys=['xenium', 'scrna'])
        
        # Perform integration (simplified - you might want to use more sophisticated methods)
        sc.pp.neighbors(combined)
        sc.tl.leiden(combined, resolution=0.5)
        
        # Transfer labels back to Xenium data
        xenium_labels = combined.obs['leiden'][combined.obs['dataset'] == 'xenium']
        self.adata.obs['integrated_clusters'] = xenium_labels.values
        
        return self.adata 
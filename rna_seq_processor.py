import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

class scRNASeqProcessor:
    def __init__(self):
        # Configure scanpy settings
        sc.settings.verbosity = 3  # verbosity level
        sc.settings.set_figure_params(dpi=80, facecolor='white')
        
    def load_10x_h5(self, file_path):
        """
        Load 10x Genomics HDF5 file using scanpy
        
        Args:
            file_path (str): Path to the .h5 file
            
        Returns:
            AnnData: Loaded single-cell data
        """
        print(f"Loading 10x HDF5 file: {file_path}")
        
        try:
            # Load 10x HDF5 file
            adata = sc.read_10x_h5(file_path)
            
            # Make variable names unique (in case of duplicates)
            adata.var_names_unique()
            
            # Add basic info
            print(f"Loaded data shape: {adata.shape}")
            print(f"Number of cells: {adata.n_obs}")
            print(f"Number of genes: {adata.n_vars}")
            
            # Store raw data
            adata.raw = adata
            
            return adata
            
        except Exception as e:
            print(f"Error loading HDF5 file: {e}")
            raise
    
    def preprocess_data(self, adata, min_genes_per_cell=200, min_cells_per_gene=3, 
                       max_genes_per_cell=5000, max_mito_percent=20, n_top_genes=2000):
        """
        Standard single-cell RNA-seq preprocessing pipeline
        
        Args:
            adata: AnnData object
            min_genes_per_cell: Minimum genes per cell
            min_cells_per_gene: Minimum cells expressing each gene
            max_genes_per_cell: Maximum genes per cell (filter potential doublets)
            max_mito_percent: Maximum mitochondrial gene percentage
            n_top_genes: Number of highly variable genes to keep
            
        Returns:
            AnnData: Preprocessed data
        """
        print("Starting preprocessing...")
        
        # Calculate QC metrics
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        
        print(f"Before filtering: {adata.shape}")
        
        # Filter cells and genes
        sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
        
        # Filter cells with too many genes (potential doublets)
        adata = adata[adata.obs.n_genes_by_counts < max_genes_per_cell, :]
        
        # Filter cells with high mitochondrial percentage
        adata = adata[adata.obs.pct_counts_mt < max_mito_percent, :]
        
        print(f"After filtering: {adata.shape}")
        
        # Normalize to 10,000 reads per cell
        sc.pp.normalize_total(adata, target_sum=1e4)
        
        # Log transform
        sc.pp.log1p(adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        
        # Keep only highly variable genes
        adata.raw = adata  # Save full data
        adata = adata[:, adata.var.highly_variable]
        
        print(f"After HVG selection: {adata.shape}")
        
        # Scale data to unit variance
        sc.pp.scale(adata, max_value=10)
        
        return adata
    
    def reduce_dimensions(self, adata, n_pcs=50):
        """
        Perform PCA for dimensionality reduction
        
        Args:
            adata: Preprocessed AnnData object
            n_pcs: Number of principal components
            
        Returns:
            AnnData: Data with PCA coordinates
        """
        print("Performing PCA...")
        
        # Principal component analysis
        sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)
        
        print(f"PCA completed with {n_pcs} components")
        
        return adata
    
    def create_cell_graph(self, adata, n_neighbors=10, use_pca=True):
        """
        Create a k-NN graph between cells based on gene expression similarity
        
        Args:
            adata: AnnData object with PCA
            n_neighbors: Number of nearest neighbors
            use_pca: Whether to use PCA coordinates for graph construction
            
        Returns:
            torch_geometric.data.Data: Graph data object
        """
        print("Creating cell-cell similarity graph...")
        
        if use_pca and 'X_pca' in adata.obsm:
            # Use PCA coordinates for graph construction
            features = adata.obsm['X_pca']
            print(f"Using PCA features: {features.shape}")
        else:
            # Use scaled gene expression
            features = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            print(f"Using expression features: {features.shape}")
        
        # Create k-NN graph
        n_neighbors = min(n_neighbors, len(features) - 1)
        A = kneighbors_graph(features, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
        
        # Convert to edge list
        edge_index = []
        coo = A.tocoo()
        for i, j in zip(coo.row, coo.col):
            edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Use PCA coordinates as node features for the GNN
        if use_pca and 'X_pca' in adata.obsm:
            x = torch.tensor(adata.obsm['X_pca'][:, :20], dtype=torch.float)  # Use first 20 PCs
        else:
            x = torch.tensor(features, dtype=torch.float)
        
        print(f"Graph created with {len(edge_index[0])} edges")
        print(f"Node features shape: {x.shape}")
        
        return Data(x=x, edge_index=edge_index), features
    
    def add_dummy_labels(self, adata, method='leiden', n_clusters=5):
        """
        Add dummy labels for demonstration (since we don't have true cancer subtypes)
        
        Args:
            adata: AnnData object
            method: 'leiden', 'kmeans', or 'random'
            n_clusters: Number of clusters
            
        Returns:
            numpy.array: Cell labels
        """
        print(f"Generating dummy labels using {method}...")
        
        if method == 'leiden':
            # Compute the neighborhood graph
            sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
            
            # Leiden clustering
            sc.tl.leiden(adata, resolution=0.5)
            labels = adata.obs['leiden'].astype(int).values
            
        elif method == 'kmeans':
            from sklearn.cluster import KMeans
            features = adata.obsm['X_pca'][:, :20]  # Use first 20 PCs
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(features)
            
        else:  # random
            labels = np.random.randint(0, n_clusters, size=adata.n_obs)
        
        print(f"Generated {len(np.unique(labels))} unique labels")
        return labels
    
    def get_summary_stats(self):
        """
        Get summary statistics of the processed data
        """
        if self.adata is None:
            return "No data loaded"
        
        stats = {
            'n_cells': self.adata.n_obs,
            'n_genes': self.adata.n_vars,
            'n_hvg': sum(self.adata.var.highly_variable) if 'highly_variable' in self.adata.var else 0,
            'median_genes_per_cell': np.median(self.adata.obs['n_genes_by_counts']),
            'median_counts_per_cell': np.median(self.adata.obs['total_counts'])
        }
        
        return stats 
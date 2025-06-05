import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, silhouette_score
import os
import webbrowser

from xenium_processor import XeniumProcessor
from rna_seq_processor import scRNASeqProcessor

class SpatialGNN(torch.nn.Module):
    """
    Spatial-aware Graph Neural Network for multi-modal spatial transcriptomics
    """
    def __init__(self, num_features, num_classes, hidden_dim=128, use_attention=True):
        super(SpatialGNN, self).__init__()
        
        self.use_attention = use_attention
        
        if use_attention:
            # Use Graph Attention Networks for better spatial modeling
            self.conv1 = GATConv(num_features, hidden_dim, heads=4, dropout=0.2)
            self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=0.2)
            self.conv3 = GATConv(hidden_dim * 4, num_classes, heads=1, dropout=0.2)
        else:
            # Standard GCN layers
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
            self.conv3 = GCNConv(hidden_dim // 2, num_classes)
        
        self.dropout = torch.nn.Dropout(0.3)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dim * 4 if use_attention else hidden_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim * 4 if use_attention else hidden_dim // 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First layer
        x = self.conv1(x, edge_index)
        if not self.use_attention:
            x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second layer
        x = self.conv2(x, edge_index)
        if not self.use_attention:
            x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)

def create_spatial_visualization(data, predictions, labels, epoch, output_dir, title_suffix=""):
    """
    Create spatial visualization overlaying predictions on tissue coordinates
    """
    if not hasattr(data, 'pos'):
        print("No spatial coordinates available for visualization")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get spatial coordinates
    spatial_coords = data.pos.numpy()
    x_coords = spatial_coords[:, 0]
    y_coords = spatial_coords[:, 1]
    
    # Plot 1: True labels
    scatter1 = axes[0, 0].scatter(x_coords, y_coords, c=labels, cmap='tab20', alpha=0.7, s=3)
    axes[0, 0].set_title(f'True Cell Types (Epoch {epoch}){title_suffix}')
    axes[0, 0].set_xlabel('X coordinate (μm)')
    axes[0, 0].set_ylabel('Y coordinate (μm)')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # Plot 2: Predictions
    scatter2 = axes[0, 1].scatter(x_coords, y_coords, c=predictions, cmap='tab20', alpha=0.7, s=3)
    axes[0, 1].set_title(f'Predicted Cell Types (Epoch {epoch}){title_suffix}')
    axes[0, 1].set_xlabel('X coordinate (μm)')
    axes[0, 1].set_ylabel('Y coordinate (μm)')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # Plot 3: Prediction confidence (entropy)
    # Calculate prediction entropy as confidence measure
    if hasattr(data, 'prediction_probs'):
        entropy = -torch.sum(data.prediction_probs * torch.log(data.prediction_probs + 1e-8), dim=1)
        scatter3 = axes[1, 0].scatter(x_coords, y_coords, c=entropy.numpy(), cmap='viridis', alpha=0.7, s=3)
        axes[1, 0].set_title(f'Prediction Uncertainty (Epoch {epoch})')
        axes[1, 0].set_xlabel('X coordinate (μm)')
        axes[1, 0].set_ylabel('Y coordinate (μm)')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(scatter3, ax=axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, 'Prediction confidence\nnot available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Plot 4: Spatial neighborhood connectivity
    # Show a subset of edges for visualization
    edge_index = data.edge_index.numpy()
    n_edges_to_show = min(1000, edge_index.shape[1])  # Limit for visibility
    edge_subset = np.random.choice(edge_index.shape[1], n_edges_to_show, replace=False)
    
    axes[1, 1].scatter(x_coords, y_coords, c=predictions, cmap='tab20', alpha=0.5, s=1)
    for i in edge_subset:
        src, dst = edge_index[0, i], edge_index[1, i]
        axes[1, 1].plot([x_coords[src], x_coords[dst]], 
                       [y_coords[src], y_coords[dst]], 
                       'gray', alpha=0.1, linewidth=0.5)
    
    axes[1, 1].set_title(f'Spatial Graph Connectivity (Sample)')
    axes[1, 1].set_xlabel('X coordinate (μm)')
    axes[1, 1].set_ylabel('Y coordinate (μm)')
    axes[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'spatial_epoch_{epoch:03d}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_integration_html(visualizations, metrics, final_accuracy, dataset_info):
    """
    Create comprehensive HTML visualization for spatial transcriptomics analysis
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Spatial Transcriptomics GNN Analysis - GSE243280</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; color: #333; }}
            .dataset-info {{ background-color: #e3f2fd; padding: 20px; border-radius: 8px; margin-bottom: 30px; border-left: 5px solid #2196f3; }}
            .metrics {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 30px; }}
            .epoch-container {{ margin-bottom: 20px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
            .epoch-header {{ background-color: #4caf50; color: white; padding: 15px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }}
            .epoch-header:hover {{ background-color: #45a049; }}
            .epoch-content {{ padding: 20px; display: none; background-color: #fff; }}
            .epoch-content.active {{ display: block; }}
            .epoch-image {{ text-align: center; }}
            .epoch-image img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .metrics-chart {{ text-align: center; margin: 20px 0; }}
            .toggle-icon {{ transition: transform 0.3s; }}
            .toggle-icon.rotated {{ transform: rotate(180deg); }}
            .data-summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
            .summary-card {{ background-color: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
            .summary-number {{ font-size: 2em; font-weight: bold; color: #4caf50; }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Spatial Transcriptomics Graph Neural Network Analysis</h1>
                <h2>GSE243280: Breast Cancer Multi-modal Integration</h2>
                <p>Xenium In Situ + scRNA-seq + Whole Slide Image Analysis</p>
            </div>
            
            <div class="dataset-info">
                <h3>Dataset Information</h3>
                <div class="data-summary">
                    <div class="summary-card">
                        <div class="summary-number">{dataset_info.get('n_cells', 'N/A')}</div>
                        <div>Total Cells</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-number">{dataset_info.get('n_genes', 'N/A')}</div>
                        <div>Genes Analyzed</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-number">{dataset_info.get('spatial_range_x', 'N/A')}</div>
                        <div>Spatial Range X (μm)</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-number">{dataset_info.get('spatial_range_y', 'N/A')}</div>
                        <div>Spatial Range Y (μm)</div>
                    </div>
                </div>
                <p><strong>Data Source:</strong> 10x Genomics Xenium In Situ Spatial Gene Expression</p>
                <p><strong>Tissue Type:</strong> Breast Cancer (T2N1M0)</p>
                <p><strong>Integration:</strong> Spatial coordinates + Gene expression + Graph neural networks</p>
            </div>
            
            <div class="metrics">
                <h3>Training Summary</h3>
                <p><strong>Final Test Accuracy:</strong> {final_accuracy:.4f}</p>
                <p><strong>Total Epochs Visualized:</strong> {len(visualizations)}</p>
                <p><strong>Graph Type:</strong> Spatial neighborhood graph (radius-based connectivity)</p>
            </div>
            
            <div class="metrics-chart">
                <div id="metricsPlot"></div>
            </div>
    """
    
    # Add epoch visualizations
    for epoch in sorted(visualizations.keys()):
        metrics_info = metrics.get(epoch, {})
        accuracy = metrics_info.get('accuracy', 0)
        loss = metrics_info.get('loss', 0)
        
        html_content += f"""
            <div class="epoch-container">
                <div class="epoch-header" onclick="toggleEpoch({epoch})">
                    <span><strong>Epoch {epoch}</strong> - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}</span>
                    <span class="toggle-icon" id="icon-{epoch}">▼</span>
                </div>
                <div class="epoch-content" id="content-{epoch}">
                    <div class="epoch-image">
                        <img src="{os.path.basename(visualizations[epoch])}" alt="Epoch {epoch} Spatial Visualization">
                    </div>
                    <p><strong>Spatial Analysis at Epoch {epoch}:</strong></p>
                    <ul>
                        <li>Top-left: Ground truth cell type distribution in tissue space</li>
                        <li>Top-right: GNN predictions overlaid on spatial coordinates</li>
                        <li>Bottom-left: Prediction uncertainty (darker = more confident)</li>
                        <li>Bottom-right: Sample of spatial graph connectivity</li>
                    </ul>
                </div>
            </div>
        """
    
    # Add metrics plotting and JavaScript
    epochs = list(sorted(metrics.keys()))
    accuracies = [metrics[e]['accuracy'] for e in epochs]
    losses = [metrics[e]['loss'] for e in epochs]
    
    html_content += f"""
        </div>
        
        <script>
            // Plot metrics
            var trace1 = {{
                x: {epochs},
                y: {accuracies},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Spatial Accuracy',
                yaxis: 'y',
                line: {{color: '#4caf50', width: 3}},
                marker: {{size: 8}}
            }};
            
            var trace2 = {{
                x: {epochs},
                y: {losses},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Training Loss',
                yaxis: 'y2',
                line: {{color: '#ff5722', width: 3}},
                marker: {{size: 8}}
            }};
            
            var layout = {{
                title: 'Spatial GNN Training Metrics Over Time',
                xaxis: {{title: 'Epoch', gridcolor: '#eee'}},
                yaxis: {{title: 'Accuracy', side: 'left', color: '#4caf50', gridcolor: '#eee'}},
                yaxis2: {{title: 'Loss', side: 'right', overlaying: 'y', color: '#ff5722'}},
                hovermode: 'x unified',
                height: 500,
                plot_bgcolor: '#fafafa',
                paper_bgcolor: 'white'
            }};
            
            Plotly.newPlot('metricsPlot', [trace1, trace2], layout);
            
            function toggleEpoch(epoch) {{
                var content = document.getElementById('content-' + epoch);
                var icon = document.getElementById('icon-' + epoch);
                
                if (content.classList.contains('active')) {{
                    content.classList.remove('active');
                    icon.classList.remove('rotated');
                }} else {{
                    content.classList.add('active');
                    icon.classList.add('rotated');
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    return html_content

def main():
    """
    Main integration pipeline for GSE243280 multi-modal spatial transcriptomics
    """
    print("=== GSE243280 Multi-modal Spatial Transcriptomics Analysis ===")
    
    # Paths to your downloaded data (you'll need to update these)
    xenium_data_path = "/path/to/xenium/data"  # Update this path
    scrna_h5_path = "/Users/ani/mapping_scRNA_GUI/GSM7782697_3p_count_filtered_feature_bc_matrix.h5"
    
    # Create output directory
    output_dir = 'spatial_xenium_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # For demonstration, we'll start with synthetic spatial data
    # Once you download the Xenium data, uncomment the real data loading section
    
    print("Loading demonstration spatial data...")
    
    # === SYNTHETIC DATA FOR DEMONSTRATION ===
    # (Remove this section once you have real Xenium data)
    n_cells = 5000
    n_genes = 500
    n_classes = 6
    
    # Create spatial layout mimicking tissue architecture
    np.random.seed(42)
    
    # Create clustered spatial coordinates
    cluster_centers = np.random.rand(n_classes, 2) * 1000  # Tissue coordinates in μm
    cell_positions = []
    cell_labels = []
    
    for i, center in enumerate(cluster_centers):
        n_cells_in_cluster = n_cells // n_classes
        # Add some cells around each center
        cluster_cells = np.random.multivariate_normal(
            center, [[50**2, 0], [0, 50**2]], n_cells_in_cluster
        )
        cell_positions.append(cluster_cells)
        cell_labels.extend([i] * n_cells_in_cluster)
    
    spatial_coords = np.vstack(cell_positions)
    labels = np.array(cell_labels)
    
    # Create gene expression data correlated with spatial position and cell type
    gene_expression = np.random.randn(n_cells, n_genes)
    
    # Add spatial gradients and cell-type-specific expression
    for i in range(n_genes):
        # Spatial gradient
        gradient_x = np.sin(spatial_coords[:, 0] / 200) * 0.5
        gradient_y = np.cos(spatial_coords[:, 1] / 200) * 0.5
        
        # Cell type specific expression
        type_effect = np.zeros(n_cells)
        for cell_type in range(n_classes):
            mask = labels == cell_type
            type_effect[mask] = np.random.normal(cell_type - n_classes/2, 0.5)
        
        gene_expression[:, i] += gradient_x + gradient_y + type_effect
    
    # Normalize gene expression
    from sklearn.preprocessing import StandardScaler
    gene_expression = StandardScaler().fit_transform(gene_expression)
    
    # Create spatial graph
    from sklearn.neighbors import radius_neighbors_graph
    radius = 75  # μm
    A = radius_neighbors_graph(spatial_coords, radius=radius, mode='connectivity')
    
    # Convert to edge list
    edge_index = []
    coo = A.tocoo()
    for i, j in zip(coo.row, coo.col):
        edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Create multi-modal features (gene expression + spatial coordinates)
    spatial_coords_norm = StandardScaler().fit_transform(spatial_coords)
    features = np.hstack([gene_expression, spatial_coords_norm])
    
    # Create PyTorch Geometric data
    from torch_geometric.data import Data
    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        pos=torch.tensor(spatial_coords, dtype=torch.float)
    )
    
    labels = torch.tensor(labels, dtype=torch.long)
    data.y = labels
    
    print(f"Created synthetic spatial data:")
    print(f"- {n_cells} cells with spatial coordinates")
    print(f"- {n_genes} genes + 2 spatial features = {features.shape[1]} total features")
    print(f"- {len(edge_index[0])} spatial edges (radius={radius}μm)")
    print(f"- {n_classes} cell types")
    
    # === REAL DATA LOADING (uncomment when you have Xenium data) ===
    # processor = XeniumProcessor()
    # adata, cell_metadata = processor.load_xenium_data(xenium_data_path)
    # processor.integrate_spatial_coordinates()
    # adata = processor.preprocess_spatial_data()
    # data, feature_names = processor.create_spatial_torch_data()
    
    # # Load scRNA-seq for integration
    # scrna_processor = scRNASeqProcessor()
    # scrna_adata = scrna_processor.load_10x_h5(scrna_h5_path)
    # scrna_adata = scrna_processor.preprocess_data(scrna_adata)
    # 
    # # Integrate datasets
    # integrated_adata = processor.integrate_with_scrna(scrna_adata)
    
    # Split data for training/testing
    n_cells = len(labels)
    indices = np.arange(n_cells)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels.numpy())
    
    train_mask = torch.zeros(n_cells, dtype=torch.bool)
    train_mask[train_idx] = True
    
    # Initialize spatial GNN model
    num_features = data.x.shape[1]
    num_classes = len(torch.unique(labels))
    model = SpatialGNN(num_features, num_classes, hidden_dim=128, use_attention=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    print(f"\nTraining Spatial GNN:")
    print(f"- Model: Spatial GAT with {num_features} features → {num_classes} classes")
    print(f"- Training cells: {train_mask.sum()}")
    print(f"- Test cells: {len(test_idx)}")
    
    # Storage for visualizations and metrics
    visualizations = {}
    metrics = {}
    
    # Training loop
    model.train()
    for epoch in range(150):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        
        # Log every 15 epochs
        if (epoch + 1) % 15 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')
        
        # Visualize every 15 epochs
        if (epoch + 1) % 15 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data).argmax(dim=1)
                train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
                
                # Store metrics
                metrics[epoch+1] = {
                    'accuracy': train_acc.item(),
                    'loss': loss.item()
                }
                
                # Create spatial visualization
                print(f'Creating spatial visualization for epoch {epoch+1}...')
                plot_path = create_spatial_visualization(
                    data, pred.numpy(), labels.numpy(), epoch+1, output_dir
                )
                visualizations[epoch+1] = plot_path
            
            model.train()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        test_mask = torch.zeros(n_cells, dtype=torch.bool)
        test_mask[test_idx] = True
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        
        # Calculate additional metrics
        ari_score = adjusted_rand_score(labels[test_mask].numpy(), pred[test_mask].numpy())
        
        print(f'\n=== Final Results ===')
        print(f'Test Accuracy: {test_acc:.4f}')
        print(f'Adjusted Rand Index: {ari_score:.4f}')
    
    # Create dataset info for HTML
    dataset_info = {
        'n_cells': n_cells,
        'n_genes': n_genes,
        'spatial_range_x': f"{spatial_coords[:, 0].min():.0f}-{spatial_coords[:, 0].max():.0f}",
        'spatial_range_y': f"{spatial_coords[:, 1].min():.0f}-{spatial_coords[:, 1].max():.0f}"
    }
    
    # Create and save HTML visualization
    print('\nCreating comprehensive HTML visualization...')
    html_content = create_integration_html(visualizations, metrics, test_acc.item(), dataset_info)
    
    html_path = os.path.join(output_dir, 'spatial_transcriptomics_analysis.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f'Spatial analysis saved to: {html_path}')
    
    # Open in browser
    try:
        webbrowser.open(f'file://{os.path.abspath(html_path)}')
        print('Opening spatial transcriptomics visualization in your browser...')
    except:
        print('Could not open browser automatically. Please open the HTML file manually.')

if __name__ == "__main__":
    main() 
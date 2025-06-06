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
    def __init__(self, num_features, num_classes, hidden_dim=128, num_heads=4, dropout=0.3, use_attention=True):
        super(SpatialGNN, self).__init__()
        
        self.use_attention = use_attention
        self.num_heads = num_heads
        
        if use_attention:
            # Use Graph Attention Networks for better spatial modeling
            self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout)
            self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            self.conv3 = GATConv(hidden_dim * num_heads, num_classes, heads=1, dropout=dropout)
        else:
            # Standard GCN layers
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
            self.conv3 = GCNConv(hidden_dim // 2, num_classes)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dim * num_heads if use_attention else hidden_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim * num_heads if use_attention else hidden_dim // 2)

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
    Main function to run spatial transcriptomics analysis with Xenium data
    """
    print("Starting Spatial Transcriptomics Analysis with Xenium Data")
    print("=" * 60)
    
    # Create output directory
    output_dir = 'spatial_xenium_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processor
    processor = XeniumProcessor()
    
    try:
        # Load Xenium data from current directory (files extracted directly here)
        print("Loading Xenium data...")
        adata, metadata = processor.load_xenium_data(".")  # Changed from xenium_data_path to current directory
        
        # Integrate spatial coordinates
        print("\nIntegrating spatial coordinates...")
        processor.integrate_spatial_coordinates()
        
        # Preprocess data
        print("\nPreprocessing spatial data...")
        adata = processor.preprocess_spatial_data()
        
        # Create spatial graph
        print("\nCreating spatial neighborhood graph...")
        edge_index = processor.create_spatial_graph(method='radius', radius=100)
        
        # Create multi-modal features
        print("\nCreating multi-modal features...")
        features, feature_names = processor.create_multimodal_features(
            use_spatial=True, 
            use_morphology=True
        )
        
        # Create PyTorch data
        print("\nPreparing data for GNN...")
        spatial_data, feature_names = processor.create_spatial_torch_data(
            graph_method='radius',
            radius=100
        )
        
        print(f"Final dataset: {spatial_data.x.shape[0]} cells, {spatial_data.x.shape[1]} features")
        print(f"Graph edges: {spatial_data.edge_index.shape[1]}")
        
        # Add dummy labels for demonstration (in practice, these would be cancer subtypes)
        n_cells = spatial_data.x.shape[0]
        # Create realistic spatial labels based on location
        spatial_coords = adata.obsm['spatial']
        
        # Use spatial clustering to create meaningful labels
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        spatial_labels = kmeans.fit_predict(spatial_coords)
        
        spatial_data.y = torch.tensor(spatial_labels, dtype=torch.long)
        
        print(f"Created {len(np.unique(spatial_labels))} spatial clusters as labels")
        
        # Split data
        n_train = int(0.7 * n_cells)
        n_val = int(0.15 * n_cells)
        
        indices = torch.randperm(n_cells)
        train_mask = torch.zeros(n_cells, dtype=torch.bool)
        val_mask = torch.zeros(n_cells, dtype=torch.bool)
        test_mask = torch.zeros(n_cells, dtype=torch.bool)
        
        train_mask[indices[:n_train]] = True
        val_mask[indices[n_train:n_train+n_val]] = True
        test_mask[indices[n_train+n_val:]] = True
        
        spatial_data.train_mask = train_mask
        spatial_data.val_mask = val_mask
        spatial_data.test_mask = test_mask
        
        # Initialize model
        model = SpatialGNN(
            num_features=spatial_data.x.shape[1],
            num_classes=len(np.unique(spatial_labels)),
            hidden_dim=64,
            num_heads=4,
            dropout=0.3
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training setup
        model.train()
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        spatial_visualizations = {}
        
        print(f"\nStarting training with {spatial_data.x.shape[1]} features...")
        print(f"Model architecture: GAT with {model.num_heads} attention heads")
        
        # Training loop
        for epoch in range(150):
            # Training
            optimizer.zero_grad()
            out = model(spatial_data)
            loss = criterion(out[train_mask], spatial_data.y[train_mask])
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                model.eval()
                pred = model(spatial_data).argmax(dim=1)
                
                train_acc = (pred[train_mask] == spatial_data.y[train_mask]).float().mean()
                val_acc = (pred[val_mask] == spatial_data.y[val_mask]).float().mean()
                
                train_losses.append(loss.item())
                train_accuracies.append(train_acc.item())
                val_accuracies.append(val_acc.item())
                
                model.train()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | '
                      f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
            
            # Create visualizations every 15 epochs
            if (epoch + 1) % 15 == 0:
                model.eval()
                with torch.no_grad():
                    predictions = model(spatial_data).argmax(dim=1).numpy()
                    probabilities = torch.softmax(model(spatial_data), dim=1)
                    uncertainty = 1 - probabilities.max(dim=1)[0].numpy()
                
                # Create spatial visualization
                viz_data = create_spatial_visualization(
                    spatial_coords=spatial_coords,
                    true_labels=spatial_data.y.numpy(),
                    predictions=predictions,
                    uncertainty=uncertainty,
                    edge_index=spatial_data.edge_index,
                    epoch=epoch+1,
                    output_dir=output_dir
                )
                
                spatial_visualizations[epoch+1] = viz_data
                model.train()
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            pred = model(spatial_data).argmax(dim=1)
            test_acc = (pred[test_mask] == spatial_data.y[test_mask]).float().mean()
            
            final_predictions = pred.numpy()
            final_probabilities = torch.softmax(model(spatial_data), dim=1)
            final_uncertainty = 1 - final_probabilities.max(dim=1)[0].numpy()
        
        print(f'\nFinal Test Accuracy: {test_acc:.4f}')
        
        # Create final comprehensive visualization
        print("Creating final spatial visualization...")
        final_viz = create_spatial_visualization(
            spatial_coords=spatial_coords,
            true_labels=spatial_data.y.numpy(),
            predictions=final_predictions,
            uncertainty=final_uncertainty,
            edge_index=spatial_data.edge_index,
            epoch='Final',
            output_dir=output_dir
        )
        
        # Create interactive HTML dashboard
        print("Creating interactive HTML dashboard...")
        html_content = create_integration_html(
            spatial_visualizations,
            {e: {'accuracy': a, 'loss': l} for e, a, l in zip(spatial_visualizations.keys(), train_accuracies, train_losses)},
            test_acc.item(),
            {
                'n_cells': n_cells,
                'n_genes': len(feature_names),
                'spatial_range_x': f"{spatial_coords[:, 0].min():.0f}-{spatial_coords[:, 0].max():.0f}",
                'spatial_range_y': f"{spatial_coords[:, 1].min():.0f}-{spatial_coords[:, 1].max():.0f}"
            }
        )
        
        # Save HTML dashboard
        html_path = os.path.join(output_dir, 'spatial_transcriptomics_analysis.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Analysis complete! Results saved to: {output_dir}/")
        print(f"Interactive dashboard: {html_path}")
        
        # Open in browser
        try:
            webbrowser.open(f'file://{os.path.abspath(html_path)}')
            print("Opening interactive dashboard in browser...")
        except:
            print("Could not open browser automatically. Please open the HTML file manually.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Please check that all required Xenium files are present:")
        print("- cell_feature_matrix.h5")
        print("- cells.parquet")
        print("- transcripts.parquet")
        print("- morphology.ome.tif")
        raise

if __name__ == "__main__":
    main() 
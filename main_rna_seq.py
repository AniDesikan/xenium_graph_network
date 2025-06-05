import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.model_selection import train_test_split
import os
import webbrowser
import matplotlib.pyplot as plt
from rna_seq_processor import scRNASeqProcessor

# Enhanced GNN for scRNA-seq data
class scRNAGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64):
        super(scRNAGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, num_classes)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First Graph Convolution Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third Graph Convolution Layer
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)

def create_rna_seq_visualization(features, predictions, labels, epoch, output_dir):
    """
    Create scatter plot visualization for scRNA-seq data using PCA coordinates
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Use first two principal components for visualization
    x_coords = features[:, 0]
    y_coords = features[:, 1]
    
    # Plot true labels
    scatter1 = ax1.scatter(x_coords, y_coords, c=labels, cmap='tab10', alpha=0.7, s=20)
    ax1.set_title(f'True Cell Clusters (Epoch {epoch})')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    plt.colorbar(scatter1, ax=ax1)
    
    # Plot predictions
    scatter2 = ax2.scatter(x_coords, y_coords, c=predictions, cmap='tab10', alpha=0.7, s=20)
    ax2.set_title(f'Predicted Cell Types (Epoch {epoch})')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'epoch_{epoch:03d}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_html_visualization(visualizations, metrics, final_accuracy):
    """
    Create HTML page with all visualizations and metrics
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>scRNA-seq GNN Training Visualization</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; color: #333; }}
            .metrics {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 30px; }}
            .epoch-container {{ margin-bottom: 20px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
            .epoch-header {{ background-color: #007bff; color: white; padding: 15px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }}
            .epoch-header:hover {{ background-color: #0056b3; }}
            .epoch-content {{ padding: 20px; display: none; background-color: #fff; }}
            .epoch-content.active {{ display: block; }}
            .epoch-image {{ text-align: center; }}
            .epoch-image img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .metrics-chart {{ text-align: center; margin: 20px 0; }}
            .toggle-icon {{ transition: transform 0.3s; }}
            .toggle-icon.rotated {{ transform: rotate(180deg); }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>scRNA-seq Graph Neural Network Training Results</h1>
                <p>Single-cell RNA sequencing data classification using Graph Convolutional Networks</p>
            </div>
            
            <div class="metrics">
                <h3>Training Summary</h3>
                <p><strong>Final Test Accuracy:</strong> {final_accuracy:.4f}</p>
                <p><strong>Total Epochs Visualized:</strong> {len(visualizations)}</p>
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
                        <img src="{os.path.basename(visualizations[epoch])}" alt="Epoch {epoch} Visualization">
                    </div>
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
                name: 'Accuracy',
                yaxis: 'y',
                line: {{color: '#1f77b4'}}
            }};
            
            var trace2 = {{
                x: {epochs},
                y: {losses},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Loss',
                yaxis: 'y2',
                line: {{color: '#ff7f0e'}}
            }};
            
            var layout = {{
                title: 'Training Metrics Over Time',
                xaxis: {{title: 'Epoch'}},
                yaxis: {{title: 'Accuracy', side: 'left', color: '#1f77b4'}},
                yaxis2: {{title: 'Loss', side: 'right', overlaying: 'y', color: '#ff7f0e'}},
                hovermode: 'x unified',
                height: 400
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
    # File path to your HDF5 file
    h5_file_path = "/Users/ani/mapping_scRNA_GUI/GSM7782697_3p_count_filtered_feature_bc_matrix.h5"
    
    # Create output directory
    output_dir = 'scrna_gnn_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting scRNA-seq GNN analysis...")
    
    # Initialize processor
    processor = scRNASeqProcessor()
    
    try:
        # Load and process real scRNA-seq data
        print("Loading scRNA-seq data...")
        adata = processor.load_10x_h5(h5_file_path)
        
        # Preprocess the data
        adata = processor.preprocess_data(adata)
        
        # Reduce dimensions
        adata = processor.reduce_dimensions(adata)
        
        # Create cell graph
        graph_data, pca_features = processor.create_cell_graph(adata)
        
        # Add dummy labels for demonstration
        labels = processor.add_dummy_labels(adata, method='leiden')
        
        print(f"Data loaded successfully!")
        print(f"Number of cells: {len(labels)}")
        print(f"Number of features: {graph_data.x.shape[1]}")
        print(f"Number of unique labels: {len(np.unique(labels))}")
        
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Generating synthetic data for demonstration...")
        
        # Generate synthetic data as fallback
        n_cells = 1000
        n_features = 20
        n_classes = 4
        
        # Create synthetic PCA-like features
        pca_features = np.random.randn(n_cells, n_features)
        labels = np.random.randint(0, n_classes, n_cells)
        
        # Create simple k-NN graph
        from sklearn.neighbors import kneighbors_graph
        A = kneighbors_graph(pca_features, n_neighbors=10, mode='connectivity', include_self=False)
        edge_index = []
        coo = A.tocoo()
        for i, j in zip(coo.row, coo.col):
            edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        from torch_geometric.data import Data
        graph_data = Data(x=torch.tensor(pca_features, dtype=torch.float), edge_index=edge_index)
    
    # Convert labels to torch tensor
    labels = torch.tensor(labels, dtype=torch.long)
    graph_data.y = labels
    
    # Split data for training/testing
    n_cells = len(labels)
    indices = np.arange(n_cells)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels.numpy())
    
    # Create training mask
    train_mask = torch.zeros(n_cells, dtype=torch.bool)
    train_mask[train_idx] = True
    
    # Initialize model
    num_features = graph_data.x.shape[1]
    num_classes = len(torch.unique(labels))
    model = scRNAGNN(num_features, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print(f"Training GNN with {num_features} features and {num_classes} classes...")
    
    # Storage for visualizations and metrics
    visualizations = {}
    metrics = {}
    
    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(graph_data)
        loss = F.nll_loss(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        
        # Log every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')
        
        # Visualize every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(graph_data).argmax(dim=1)
                train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
                
                # Store metrics
                metrics[epoch+1] = {
                    'accuracy': train_acc.item(),
                    'loss': loss.item()
                }
                
                # Create visualization
                print(f'Creating visualization for epoch {epoch+1}...')
                plot_path = create_rna_seq_visualization(
                    pca_features, pred.numpy(), labels.numpy(), epoch+1, output_dir
                )
                visualizations[epoch+1] = plot_path
            
            model.train()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred = model(graph_data).argmax(dim=1)
        test_mask = torch.zeros(n_cells, dtype=torch.bool)
        test_mask[test_idx] = True
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        print(f'Final Test Accuracy: {test_acc:.4f}')
    
    # Create and save HTML visualization
    print('Creating HTML visualization...')
    html_content = create_html_visualization(visualizations, metrics, test_acc.item())
    
    html_path = os.path.join(output_dir, 'scrna_training_visualization.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f'Visualization saved to: {html_path}')
    
    # Open in browser
    try:
        webbrowser.open(f'file://{os.path.abspath(html_path)}')
        print('Opening visualization in your default browser...')
    except:
        print('Could not open browser automatically. Please open the HTML file manually.')

if __name__ == "__main__":
    main() 
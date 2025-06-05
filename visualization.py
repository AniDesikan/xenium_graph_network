import matplotlib.pyplot as plt
import networkx as nx
import base64
from io import BytesIO
import json

def visualize_graph(data, predictions, epoch, true_labels=None):
    """Create a visualization of the graph with node colors based on predictions"""
    # Convert to NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(len(data.x)):
        G.add_node(i)
    
    # Add edges
    edge_list = data.edge_index.t().numpy()
    for edge in edge_list:
        G.add_edge(edge[0], edge[1])
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Color nodes based on predictions
    colors = ['red', 'green', 'blue']
    node_colors = [colors[pred] for pred in predictions]
    
    # Draw the graph
    nx.draw(G, pos, node_color=node_colors, node_size=300, 
            with_labels=True, font_size=8, font_weight='bold')
    
    # Add title
    if true_labels is not None:
        accuracy = (predictions == true_labels).mean()
        plt.title(f'Epoch {epoch} - Graph Neural Network\nAccuracy: {accuracy:.3f}')
    else:
        plt.title(f'Epoch {epoch} - Graph Neural Network')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=colors[i], markersize=10, 
                                label=f'Class {i}') for i in range(3)]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Save to string buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def create_metrics_plot(epochs, accuracies, losses):
    """Create a plot showing accuracy and loss over epochs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy plot
    ax1.plot(epochs, accuracies, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training Accuracy Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Loss plot
    ax2.plot(epochs, losses, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Over Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to string buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def create_html_visualization(visualizations, metrics, final_accuracy):
    """Create an HTML file with collapsible epoch cards and metrics visualization"""
    
    # Extract epochs, accuracies, and losses for the metrics plot
    epochs = sorted(metrics.keys())
    accuracies = [metrics[epoch]['accuracy'] for epoch in epochs]
    losses = [metrics[epoch]['loss'] for epoch in epochs]
    
    # Create metrics plot
    metrics_plot = create_metrics_plot(epochs, accuracies, losses)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GNN Training Visualization</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
                line-height: 1.6;
            }}
            .header {{
                text-align: center;
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0 0 10px 0;
                font-size: 2.5em;
            }}
            .header p {{
                margin: 5px 0;
                font-size: 1.1em;
            }}
            .visualization-container {{
                display: flex;
                flex-direction: column;
                gap: 20px;
                margin-bottom: 30px;
                max-width: 800px;
                margin-left: auto;
                margin-right: auto;
            }}
            .epoch-card {{
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                overflow: hidden;
                transition: all 0.3s ease;
            }}
            .epoch-card:hover {{
                box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
            }}
            .epoch-header {{
                padding: 20px;
                background-color: #34495e;
                color: white;
                cursor: pointer;
                user-select: none;
                transition: background-color 0.3s ease;
            }}
            .epoch-header:hover {{
                background-color: #2c3e50;
            }}
            .epoch-header h3 {{
                margin: 0 0 10px 0;
                font-size: 1.4em;
            }}
            .epoch-stats {{
                display: flex;
                justify-content: space-between;
                margin-top: 10px;
            }}
            .stat {{
                text-align: center;
            }}
            .stat-label {{
                font-size: 0.9em;
                opacity: 0.8;
            }}
            .stat-value {{
                font-size: 1.1em;
                font-weight: bold;
            }}
            .epoch-content {{
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease;
                padding: 0 20px;
            }}
            .epoch-content.expanded {{
                max-height: 800px;
                padding: 20px;
            }}
            .epoch-image {{
                width: 100%;
                border-radius: 5px;
                margin-top: 10px;
            }}
            .metrics-section {{
                background-color: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 30px;
            }}
            .metrics-title {{
                font-size: 1.8em;
                margin-bottom: 20px;
                text-align: center;
                color: #2c3e50;
            }}
            .metrics-image {{
                width: 100%;
                border-radius: 5px;
            }}
            .final-stats {{
                background-color: #27ae60;
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 1.1em;
            }}
            .expand-indicator {{
                float: right;
                transition: transform 0.3s ease;
            }}
            .expand-indicator.rotated {{
                transform: rotate(180deg);
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Graph Neural Network Training Visualization</h1>
            <p>Iris Classification using Custom GNN</p>
            <p>Final Test Accuracy: <strong>{final_accuracy:.3f}</strong></p>
        </div>
        
        <div class="metrics-section">
            <h2 class="metrics-title">Training Metrics Over Time</h2>
            <img src="data:image/png;base64,{metrics_plot}" class="metrics-image" alt="Training metrics visualization">
        </div>
        
        <div class="visualization-container">
    """
    
    # Add epoch cards
    for epoch in sorted(visualizations.keys()):
        accuracy = metrics[epoch]['accuracy']
        loss = metrics[epoch]['loss']
        
        html_content += f"""
            <div class="epoch-card">
                <div class="epoch-header" onclick="toggleCard(this)">
                    <h3>Epoch {epoch} <span class="expand-indicator">▼</span></h3>
                    <div class="epoch-stats">
                        <div class="stat">
                            <div class="stat-label">Accuracy</div>
                            <div class="stat-value">{accuracy:.3f}</div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Loss</div>
                            <div class="stat-value">{loss:.4f}</div>
                        </div>
                    </div>
                </div>
                <div class="epoch-content">
                    <img src="data:image/png;base64,{visualizations[epoch]}" class="epoch-image" alt="Epoch {epoch} visualization">
                </div>
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="final-stats">
            <h2>Training Complete!</h2>
            <p>The GNN successfully learned to classify the Iris dataset.</p>
            <p>Click on any epoch card above to see the network's predictions at that time.</p>
        </div>
        
        <script>
            function toggleCard(header) {
                const content = header.nextElementSibling;
                const indicator = header.querySelector('.expand-indicator');
                
                if (content.classList.contains('expanded')) {
                    content.classList.remove('expanded');
                    indicator.classList.remove('rotated');
                } else {
                    content.classList.add('expanded');
                    indicator.classList.add('rotated');
                }
            }
            
            // Add hover effects
            document.querySelectorAll('.epoch-card').forEach(card => {
                card.addEventListener('mouseenter', function() {
                    this.style.transform = 'translateY(-2px)';
                });
                card.addEventListener('mouseleave', function() {
                    this.style.transform = 'translateY(0)';
                });
            });
        </script>
    </body>
    </html>
    """
    
    return html_content 
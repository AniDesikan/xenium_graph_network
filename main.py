import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import webbrowser
from visualization import visualize_graph, create_html_visualization

# Define the GNN model
class SimpleGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First Graph Convolution Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

def create_graph_from_features(features, labels):
    # Create a k-NN graph instead of fully connected for better visualization
    from sklearn.neighbors import kneighbors_graph
    
    # Create k-nearest neighbors graph
    k = min(5, len(features) - 1)  # Connect each node to 5 nearest neighbors
    A = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=False)
    edge_index = []
    
    # Convert adjacency matrix to edge list
    coo = A.tocoo()
    for i, j in zip(coo.row, coo.col):
        edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y)

def main():
    # Create directory for outputs
    os.makedirs('gnn_visualization', exist_ok=True)
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create graph data
    train_data = create_graph_from_features(X_train, y_train)
    test_data = create_graph_from_features(X_test, y_test)
    
    # Initialize model
    model = SimpleGNN(num_features=4, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Storage for visualizations and metrics
    visualizations = {}
    metrics = {}
    
    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(train_data)
        loss = F.nll_loss(out, train_data.y)
        loss.backward()
        optimizer.step()
        
        # Log every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')
        
        # Visualize every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Get predictions for training data
                train_pred = model(train_data).argmax(dim=1).numpy()
                
                # Calculate accuracy
                accuracy = (train_pred == train_data.y.numpy()).mean()
                
                # Store metrics
                metrics[epoch+1] = {
                    'accuracy': accuracy,
                    'loss': loss.item()
                }
                
                # Create visualization
                print(f'Creating visualization for epoch {epoch+1}...')
                viz = visualize_graph(train_data, train_pred, epoch+1, train_data.y.numpy())
                visualizations[epoch+1] = viz
            
            model.train()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred = model(test_data).argmax(dim=1)
        correct = int((pred == test_data.y).sum())
        acc = correct / len(test_data.y)
        print(f'Final Test Accuracy: {acc:.4f}')
    
    # Create and save HTML visualization
    print('Creating HTML visualization...')
    html_content = create_html_visualization(visualizations, metrics, acc)
    
    html_path = os.path.join('gnn_visualization', 'training_visualization.html')
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

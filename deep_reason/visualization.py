import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional

def plot_community_size_histogram(communities_parquet_path: str, output_path: Optional[str] = None) -> None:
    """
    Plot an interactive histogram of community sizes from a communities parquet file.
    
    Args:
        communities_parquet_path (str): Path to the parquet file containing community data
        output_path (Optional[str]): Path to save the plot. If None, the plot will be displayed in browser.
    """
    # Read communities from parquet file
    communities_df = pd.read_parquet(communities_parquet_path)
    
    # Calculate community sizes and filter out communities with 2 or 3 entities
    community_sizes = [len(entity_ids) for entity_ids in communities_df['entity_ids'] if len(entity_ids) > 3]
    
    # Calculate statistics
    stats = {
        'Total Communities': len(community_sizes),
        'Mean Size': np.mean(community_sizes),
        'Median Size': np.median(community_sizes),
        'Min Size': min(community_sizes),
        'Max Size': max(community_sizes),
        'Communities Removed': len(communities_df) - len(community_sizes)
    }
    
    # Create the plot
    fig = go.Figure()
    
    # Calculate bin edges for fixed width of 20
    max_size = max(community_sizes)
    bin_edges = list(range(0, max_size + 20, 20))
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=community_sizes,
        xbins=dict(
            start=0,
            end=max_size + 20,
            size=20
        ),
        name='Community Sizes',
        marker_color='#1f77b4',
        opacity=0.75
    ))
    
    # Update layout
    fig.update_layout(
        title='Distribution of Community Sizes',
        xaxis_title='Number of Entities in Community',
        yaxis_title='Number of Communities',
        showlegend=False,
        template='plotly_white',
        hovermode='x unified',
        annotations=[
            dict(
                x=0.95,
                y=0.95,
                xref='paper',
                yref='paper',
                text='<br>'.join([f'{k}: {v:.1f}' if isinstance(v, float) else f'{k}: {v}' 
                                 for k, v in stats.items()]),
                showarrow=False,
                bgcolor='white',
                bordercolor='black',
                borderwidth=1,
                borderpad=4,
                align='right'
            )
        ]
    )
    
    # Save or show the plot
    if output_path:
        fig.write_html(output_path.replace('.png', '.html'))
    else:
        fig.show()

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from . import util
import numpy as np
from plotly.subplots import make_subplots

def plot_volume(data_dict, title=None):
    """
    Plot volume of submissions and comments over time using a stacked bar chart.
    
    Args:
        data_dict (dict): Dictionary containing 'submissions' and 'comments' DataFrames
        title (str, optional): Custom title for the plot. If None, will use "Reddit content volume"
    """
    # Create counts DataFrame
    counts = {}
    total_counts = {}
    
    for key in ['submissions', 'comments']:
        if key in data_dict and not data_dict[key].empty:
            counts[key] = data_dict[key]['dataset'].value_counts()
            total_counts[key] = int(len(data_dict[key]))
    
    if not counts:
        print("No data to plot")
        return
        
    # Combine into DataFrame and sort chronologically
    counts_df = pd.DataFrame(counts).fillna(0)
    counts_df = counts_df.sort_index()
    
    # Generate plot title
    if title is None:
        title = "Reddit content volume"
    
    # Add counts to title
    title_counts = []
    for key, count in total_counts.items():
        title_counts.append(f"{count:,} {key}")
    title = f"{title} ({', '.join(title_counts)})"
    
    # Create stacked bar chart
    fig = px.bar(
        counts_df,
        title=title,
        barmode='stack',
        color_discrete_map={
            'submissions': tab10_to_hex('tab:orange'),
            'comments': tab10_to_hex('tab:blue')
        }
    )
    
    return fig

def plot_metrics(data_dict, metrics, title=None, quantiles=(0.25, 0.75), center='mean'):
    """
    Plot metrics over time with shaded quantile regions.
    
    Args:
        data_dict (dict): Dictionary containing 'submissions' and 'comments' DataFrames
        metrics (list): List of metric column names to plot
        title (str, optional): Custom title for the plot
        quantiles (tuple, optional): Lower and upper quantiles for shaded regions. Defaults to (0.25, 0.75)
        center (str, optional): Central tendency measure to plot. Either 'mean' or 'median'. Defaults to 'mean'
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    if center not in ['mean', 'median']:
        raise ValueError("center must be either 'mean' or 'median'")
    
    # Colors for submissions and comments
    sub_color = tab10_to_hex('tab:orange')
    com_color = tab10_to_hex('tab:blue')
    
    # Create subplots with moderate vertical spacing
    fig = make_subplots(rows=len(metrics), cols=1, subplot_titles=metrics, vertical_spacing=0.1)
    
    # For each metric
    for i, metric in enumerate(metrics, 1):
        # For each data type (submissions/comments)
        for data_type, color in [('submissions', sub_color), ('comments', com_color)]:
            if data_type in data_dict and not data_dict[data_type].empty:
                df = data_dict[data_type]
                
                # Calculate central tendency and quantiles by dataset
                if center == 'mean':
                    stats = df.groupby('dataset')[metric].agg(['mean'])
                else:  # median
                    stats = df.groupby('dataset')[metric].agg(['median'])
                stats = stats.sort_index()
                
                quant_low = df.groupby('dataset')[metric].quantile(quantiles[0])
                quant_high = df.groupby('dataset')[metric].quantile(quantiles[1])
                
                # Add central tendency line
                fig.add_trace(
                    go.Scatter(
                        x=stats.index,
                        y=stats[center],
                        name=f'{data_type} {center}',
                        line=dict(color=color),
                        legendgroup=f'{data_type}_{metric}',
                        showlegend=True,
                        legendgrouptitle_text=metric
                    ),
                    row=i, col=1
                )
                
                # Add shaded region for quantiles
                fig.add_trace(
                    go.Scatter(
                        x=stats.index,
                        y=quant_high,
                        fill=None,
                        mode='lines',
                        line=dict(width=0),
                        legendgroup=f'{data_type}_{metric}',
                        showlegend=False,
                        name=f'{data_type} quantiles'
                    ),
                    row=i, col=1
                )
                
                # Extract RGB values from hex color
                rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                
                fig.add_trace(
                    go.Scatter(
                        x=stats.index,
                        y=quant_low,
                        fill='tonexty',
                        mode='lines',
                        line=dict(width=0),
                        fillcolor=f'rgba{rgb + (0.2,)}',  # Add transparency
                        legendgroup=f'{data_type}_{metric}',
                        showlegend=True,
                        name=f'{data_type} {quantiles[0]:.0%}-{quantiles[1]:.0%} quantiles'
                    ),
                    row=i, col=1
                )
    
    # Update layout
    fig.update_layout(
        height=300 * len(metrics),  # Slightly increased height per subplot
        title_text=title,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left", 
            x=1.02,
            tracegroupgap=30  # Add gap between legend groups
        )
    )
    
    return fig

def plot_metrics_thresholded(data_dict, metrics, thresholds, title=None, as_percentage=True):
    """
    Plot count or percentage of metric values above thresholds over time.
    
    Args:
        data_dict (dict): Dictionary containing 'submissions' and 'comments' DataFrames
        metrics (list): List of metric column names to plot
        thresholds (dict): Dictionary mapping metric names to threshold values
        title (str, optional): Custom title for the plot
        as_percentage (bool): If True, show as percentage, otherwise as count. Defaults to True
    """

    
    fig = go.Figure()
    
    # Get colors from plotly express qualitative color sequence
    colors = px.colors.qualitative.Set1
    
    # For each metric
    for i, metric in enumerate(metrics):
        threshold = thresholds[metric]
        color = colors[i % len(colors)]  # Cycle through colors if more metrics than colors
        
        # For each data type (submissions/comments)
        for data_type in ['submissions', 'comments']:
            if data_type in data_dict and not data_dict[data_type].empty:
                df = data_dict[data_type]
                
                # Calculate counts/percentages by dataset
                total = df.groupby('dataset').size()
                above_threshold = df[df[metric] > threshold].groupby('dataset').size()
                
                if as_percentage:
                    values = (above_threshold / total * 100).fillna(0)
                else:
                    values = above_threshold.fillna(0)
                
                # Add line with same color per metric but thicker for comments
                fig.add_trace(
                    go.Scatter(
                        x=values.index,
                        y=values,
                        name=f'{metric} > {threshold} ({data_type})',
                        line=dict(color=color, width=3 if data_type == 'comments' else 1),
                        mode='lines'
                    )
                )
    
    # Update layout with height scaled by number of metrics
    fig.update_layout(
        title_text=title,
        yaxis_title='Percentage' if as_percentage else 'Count',
        showlegend=True,
        height=max(400, 30 * len(metrics)),  # Scale height with metrics but keep minimum size
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02
        )
    )

    return fig

def plot_goemotions(data_dict, emotions = 'all', title=None, as_percentage=True):
    thresholds = util.get_goemotions_thresholds()
    if emotions != 'all':
        thresholds = {k: v for k, v in thresholds.items() if k in emotions}
    return plot_metrics_thresholded(data_dict, list(thresholds.keys()), thresholds, title, as_percentage)

def plot_distributions(data_dict, metrics, periods, title=None):
    """
    Plot distributions of metrics across different time periods.
    
    Args:
        data_dict (dict): Dictionary containing 'submissions' and/or 'comments' DataFrames
        metrics (list): List of metric names to plot
        periods (list): List of tuples containing (start_year, end_year) in YYYY format
        title (str): Optional title for the plot
    
    Returns:
        plotly.graph_objects.Figure
    """

    
    # Generate period names from years
    period_names = [f"{start}-{end}" for start, end in periods]
    
    # Create subplots
    fig = make_subplots(
        rows=len(metrics), 
        cols=len(periods),
        subplot_titles=[name for name in period_names] * len(metrics),
        vertical_spacing=0.1
    )
    
    colors = {
        'submissions': tab10_to_hex('tab:orange'),
        'comments': tab10_to_hex('tab:blue')
    }
    
    # Pre-compute bins for each metric
    metric_bins = {}
    for metric in metrics:
        all_data = []
        for df in data_dict.values():
            if not df.empty:
                all_data.extend(df[metric].values)
        if all_data:
            min_val = np.min(all_data)
            max_val = np.max(all_data)
            metric_bins[metric] = np.linspace(min_val, max_val, 51)  # 50 bins
    
    for i, metric in enumerate(metrics, 1):
        for j, ((start_year, end_year), period_name) in enumerate(zip(periods, period_names), 1):
            
            for data_type, df in data_dict.items():
                if df.empty:
                    continue
                    
                # Create year column from dataset (YYYY-MM format)
                df['year'] = df['dataset'].str[:4].astype(int)
                
                # Filter data for period
                mask = (df['year'] >= int(start_year)) & (df['year'] <= int(end_year))
                period_data = df[mask][metric]
                
                # Pre-compute histogram using the metric's global bins
                counts, _ = np.histogram(period_data, bins=metric_bins[metric], density=True)
                
                # Add bar trace (more efficient than histogram)
                fig.add_trace(
                    go.Bar(
                        x=metric_bins[metric][:-1],
                        y=counts,
                        name=data_type.capitalize(),
                        opacity=0.6,
                        showlegend=True if (i==1 and j==1) else False,
                        legendgroup=data_type,
                        marker_color=colors[data_type],
                        width=metric_bins[metric][1] - metric_bins[metric][0]  # Set bar width to bin size
                    ),
                    row=i, col=j
                )
                
                # Drop temporary year column
                df.drop('year', axis=1, inplace=True)
            
            # Update axes labels and ranges
            fig.update_xaxes(
                title_text=metric.capitalize(), 
                row=i, col=j,
                range=[metric_bins[metric][0], metric_bins[metric][-1]]  # Set same range for all periods
            )
            fig.update_yaxes(title_text='Probability' if j==1 else None, row=i, col=j)
    
    # Update layout
    fig.update_layout(
        height=300 * len(metrics),
        width=400 * len(periods),
        title_text=title or 'Distribution of content metrics across time periods',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02
        ),
        barmode='overlay'
    )
    
    return fig

def tab10_to_hex(tab_color):
    """
    Convert a tab10 color name to its hex value for plotly.
    
    Args:
        tab_color (str): Color name in tab10 format (e.g. 'tab:blue', 'tab:orange')
        
    Returns:
        str: Hex color string (e.g. '#1f77b4')
        
    Raises:
        ValueError: If the provided color name is not in tab10 palette
    """
    tab10_colors = {
        'tab:blue': '#1f77b4',
        'tab:orange': '#ff7f0e',
        'tab:green': '#2ca02c', 
        'tab:red': '#d62728',
        'tab:purple': '#9467bd',
        'tab:brown': '#8c564b',
        'tab:pink': '#e377c2',
        'tab:gray': '#7f7f7f',
        'tab:olive': '#bcbd22',
        'tab:cyan': '#17becf'
    }
    
    if tab_color not in tab10_colors:
        raise ValueError(f"Color '{tab_color}' not found in tab10 palette")
        
    return tab10_colors[tab_color]

def survey_dashboard(df):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Trends Over Time', 'Demographic Comparison',
                        'Score Heatmap', 'Year Comparison'),
        specs=[[{"type": "scatter"}, {"type": "box"}],
               [{"type": "heatmap"}, {"type": "box"}]],
        vertical_spacing=0.12,  # Reduce space between rows
        horizontal_spacing=0.1  # Adjust space between columns
    )
    
    # Add traces for the time trends plot
    metrics = ['overall', 'relatable', 'inspiring', 'likable', 'relevant']
    for metric in metrics:
        yearly_means = df.groupby('year')[metric].mean()
        fig.add_trace(go.Scatter(
            x=yearly_means.index,
            y=yearly_means.values,
            name=metric.capitalize(),
            mode='lines+markers'
        ), row=1, col=1)
    
    # Add traces for the demographic comparison plot
    fig.add_trace(go.Box(
        x=df['demo'],
        y=df['overall'],
        name='Overall',
        marker_color='blue'
    ), row=1, col=2)
    
    # Add traces for the heatmap
    pivot_table = df.pivot_table(
        values=metrics,
        index='demo',
        aggfunc='mean'
    )
    fig.add_trace(go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdBu',
        text=np.round(pivot_table.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        showscale=True,
        colorbar=dict(
            len=0.5,         # Length of colorbar
            y=0.25,          # Position of colorbar
            yanchor='middle' # Anchor point for y position
        )
    ), row=2, col=1)
    
    # Add traces for the year comparison plot
    fig.add_trace(go.Box(
        x=df['year'],
        y=df['overall'],
        name='Overall',
        marker_color='green'
    ), row=2, col=2)
    
    # Update layout with better spacing and formatting
    fig.update_layout(
        height=800,           # Reduced height
        width=1200,
        title_text="Survey Results Dashboard",
        showlegend=True,
        legend=dict(
            orientation="h",   # Horizontal legend
            yanchor="bottom",
            y=1.02,           # Position legend above the plots
            xanchor="right",
            x=1
        )
    )
    
    # Update axes labels and tick angles
    fig.update_xaxes(tickangle=45)  # Angle the x-axis labels
    
    return fig
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the data
f = np.load("/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/labels_for_vis_purposes.npz", allow_pickle=True)

length_to_plot = 10000

spec = f["original_spectogram"][:length_to_plot]
vocalization = f["vocalization"][:length_to_plot]
labels = f["hdbscan_labels"]
colors = f["hdbscan_colors"]
embeddings = f["embedding_outputs"]

# Process labels
temp = np.asarray(vocalization)
indexes = np.where(temp == 1)[0]
for i, index in enumerate(indexes):
    temp[index] = labels[i]
labels = temp[:length_to_plot]

# Process embedding
extended_embeddings = np.zeros((length_to_plot, 2))
for i, index in enumerate(indexes):
    extended_embeddings[index] = embeddings[i]

# Create mapping from UMAP index to original data index
umap_to_original_mapping = {str(i): int(index) for i, index in enumerate(range(10000))}

# Convert the mapping to a list of [key, value] pairs
umap_to_original_list = [[str(k), v] for k, v in umap_to_original_mapping.items()]

print("UMAP to Original Mapping (first 10 items):")
print(dict(list(umap_to_original_mapping.items())[:10]))
print("Total mapping items:", len(umap_to_original_mapping))

# Create the Plotly figure for spectrogram, labels, and selection bar
fig = make_subplots(rows=3, cols=1, row_heights=[0.6, 0.2, 0.2], shared_xaxes=True, vertical_spacing=0.02)

# Add spectrogram
fig.add_trace(
    go.Heatmap(z=spec.T, colorscale='Viridis', zmin=spec.min(), zmax=spec.max(), showscale=False),
    row=1, col=1
)

# Add labels
fig.add_trace(
    go.Heatmap(z=[labels], colorscale=colors, zmin=labels.min(), zmax=labels.max(), showscale=False),
    row=2, col=1
)

# Add selection bar
selection_bar = np.zeros(length_to_plot)
fig.add_trace(
    go.Heatmap(z=[selection_bar], colorscale='Greys', zmin=0, zmax=1, showscale=False),
    row=3, col=1
)

# Create the Plotly figure for UMAP visualization
fig_umap = go.Figure()

# Add UMAP visualization
fig_umap.add_trace(
    go.Scatter(
        x=extended_embeddings[:, 0],
        y=extended_embeddings[:, 1],
        mode='markers',
        marker=dict(
            color=labels,
            colorscale=colors,
            size=3,
            showscale=False
        ),
        name='UMAP'
    )
)

# Update layout for both figures
fig.update_layout(
    title='Spectrogram, HDBSCAN Labels, and Selection Bar',
    height=900,
    margin=dict(l=20, r=20, t=40, b=20),
    yaxis=dict(title='Frequency'),
    yaxis2=dict(title='Labels'),
    yaxis3=dict(title='Selection')
)

fig_umap.update_layout(
    title='UMAP Visualization',
    height=800,
    width=800,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(title='UMAP Dimension 1'),
    yaxis=dict(title='UMAP Dimension 2')
)

# Create HTML file
html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Spectrogram, Labels, and UMAP</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .container {{
            display: flex;
        }}
        .left {{
            flex: 2;
        }}
        .right {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="left">
            <div id="plot1"></div>
        </div>
        <div class="right">
            <div id="plot2"></div>
        </div>
    </div>
    <script>
        var plotData1 = {fig.to_json()};
        Plotly.newPlot('plot1', plotData1.data, plotData1.layout);
        var plotData2 = {fig_umap.to_json()};
        Plotly.newPlot('plot2', plotData2.data, plotData2.layout);

        var umapToOriginalList = {json.dumps(umap_to_original_list)};
        var umapToOriginalMapping = new Map(umapToOriginalList);
        console.log('UMAP to Original Mapping size:', umapToOriginalMapping.size);
        console.log('First 10 items of UMAP to Original Mapping:', 
            Array.from(umapToOriginalMapping.entries()).slice(0, 10));

        var plot2 = document.getElementById('plot2');
        plot2.on('plotly_selected', function(eventData) {{
            if (eventData) {{
                var selectedIndexes = eventData.points.map(point => point.pointIndex);
                console.log('Selected indexes:', selectedIndexes);
                var selectionBarUpdate = new Array(10000).fill(0);
                var updatedIndexes = [];
                selectedIndexes.forEach(index => {{
                    var originalIndex = umapToOriginalMapping.get(index.toString());
                    console.log('UMAP index:', index, 'Original index:', originalIndex);
                    if (originalIndex !== undefined && originalIndex < 10000) {{
                        selectionBarUpdate[originalIndex] = 1;
                        updatedIndexes.push(originalIndex);
                    }}
                }});
                console.log('Updated indexes:', updatedIndexes);
                console.log('Selection bar update:', selectionBarUpdate);
                var update = {{
                    z: [selectionBarUpdate]
                }};
                console.log('Update object:', update);
                Plotly.restyle('plot1', update, 2).then(function() {{
                    console.log('Restyle completed successfully');
                    var updatedData = Plotly.d3.select('#plot1').data()[2].z[0];
                    console.log('Updated selection bar data:', updatedData);
                }}).catch(function(err) {{
                    console.error('Error in restyle:', err);
                }});
            }}
        }});

        setInterval(function() {{
            var plot1 = document.getElementById('plot1');
            if (plot1 && plot1.data && plot1.data[2] && plot1.data[2].z) {{
                var selectionBarData = plot1.data[2].z[0];
                if (Array.isArray(selectionBarData)) {{
                    var nonZeroCount = selectionBarData.filter(x => x !== 0).length;
                    console.log('Current selection bar data (non-zero count):', nonZeroCount);
                }} else {{
                    console.log('Selection bar data is not an array:', selectionBarData);
                    console.log('Full data[2].z:', plot1.data[2].z);
                }}
            }} else {{
                console.log('Selection bar data not available');
                if (plot1 && plot1.data) {{
                    console.log('Number of traces:', plot1.data.length);
                    console.log('Trace 2 data:', plot1.data[2]);
                }}
            }}
        }}, 5000);
    </script>
</body>
</html>
'''

# Save the HTML file
with open('interactive_plot.html', 'w') as f:
    f.write(html_content)

print("HTML file 'interactive_plot.html' has been created.")
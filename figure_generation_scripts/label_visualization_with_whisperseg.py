import numpy as np
import json
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, row, layout
from bokeh.models import ColorBar, LinearColorMapper, ColumnDataSource, CustomJS, TapTool, LassoSelectTool, Range1d
from bokeh.transform import linear_cmap
from bokeh.io import show

# Load the data
f = np.load("/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/labels_new_whisperseg_test_pitch.npz", allow_pickle=True)

length_to_plot = 50000
downsample_factor = 5  # Adjust this value to balance between performance and detail

spec = f["s"][:length_to_plot]
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

# Downsample data for UMAP plot
downsampled_embeddings = extended_embeddings[::downsample_factor]
downsampled_labels = labels[::downsample_factor]

# Create Bokeh figures
output_file("interactive_plot.html")

# Shared x-range for synchronization
shared_x_range = Range1d(0, length_to_plot)

# Spectrogram figure
spec_fig = figure(title="Spectrogram", x_axis_label='Time', y_axis_label='Frequency', height=300, x_range=shared_x_range, sizing_mode='stretch_both')
spec_fig.image(image=[spec.T], x=0, y=0, dw=length_to_plot, dh=spec.shape[1], palette="Viridis256")

# Labels figure
labels_fig = figure(title="Labels", x_axis_label='Time', y_axis_label='Labels', height=200, x_range=shared_x_range, sizing_mode='stretch_both')
labels_fig.image(image=[labels.reshape(1, -1)], x=0, y=0, dw=length_to_plot, dh=1, palette="Viridis256")

# UMAP figure with WebGL
umap_fig = figure(title="UMAP Visualization", x_axis_label='UMAP Dimension 1', y_axis_label='UMAP Dimension 2', 
                  height=800, aspect_ratio=1, tools="lasso_select,tap,pan,wheel_zoom,box_zoom,reset", output_backend="webgl", sizing_mode='stretch_both')
umap_source = ColumnDataSource(data=dict(x=downsampled_embeddings[:, 0], y=downsampled_embeddings[:, 1], color=downsampled_labels))
umap_fig.scatter('x', 'y', color=linear_cmap('color', 'Viridis256', min(downsampled_labels), max(downsampled_labels)), 
                 size=3, source=umap_source)

# Selected indexes figure
selected_indexes_fig = figure(title="Selected Indexes", x_axis_label='Time', y_axis_label='Selected', height=100, x_range=shared_x_range, sizing_mode='stretch_both')
selected_indexes_source = ColumnDataSource(data=dict(x=[]))
selected_indexes_fig.rect(x='x', y=0.5, width=1, height=1, color='red', source=selected_indexes_source)

# JavaScript callback for updating selected indexes
callback = CustomJS(args=dict(selected_source=selected_indexes_source, downsample_factor=downsample_factor), code="""
    var selected_indexes = cb_obj.indices;
    console.log('Selected indexes array:', selected_indexes);
    var x_coords = [];
    for (var i = 0; i < selected_indexes.length; i++) {
        x_coords.push(selected_indexes[i] * downsample_factor);
    }
    console.log('X coordinates:', x_coords);
    selected_source.data = {x: x_coords};
    selected_source.change.emit();
""")

umap_source.selected.js_on_change('indices', callback)
umap_fig.add_tools(LassoSelectTool())

# Layout
left_column = column(spec_fig, labels_fig, selected_indexes_fig, sizing_mode='stretch_both')
main_layout = layout([
    [left_column, umap_fig]
], sizing_mode='stretch_both')

save(main_layout)

print("HTML file 'interactive_plot.html' has been created.")
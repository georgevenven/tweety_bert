import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QDoubleSpinBox, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QGraphicsRectItem
import numpy as np
import os
import json

# Set the directory containing spectrogram files
directory = '/media/george-vengrovski/Extreme SSD/sham lesioned birds/arhive/USA5271_no_threshold_no_norm_no_norm_test/'

# Load all npz files in the directory
spectrogram_files = [f for f in os.listdir(directory) if f.endswith('.npz')]
spectrogram_files.sort()

# Initialize variables
current_file_index = 0
thresholds = {}
initial_threshold = 0
# Function to load spectrogram data from file
def load_spectrogram(file_index):
    file_path = os.path.join(directory, spectrogram_files[file_index])
    data = np.load(file_path)
    spectrogram = data['s']  # Keep original orientation
    labels = data['labels']
    
    # Calculate vocal envelope from 20:50
    spectrogram_segment = spectrogram[20:216]
    
    # Z-score normalization
    spectrogram_mean = np.mean(spectrogram_segment)
    spectrogram_std = np.std(spectrogram_segment)
    spectrogram_segment = (spectrogram_segment - spectrogram_mean) / spectrogram_std
    
    # Calculate vocal envelope
    mu_s = np.mean(spectrogram_segment, axis=0)  # Average power over frequency components
    max_s = np.max(spectrogram_segment, axis=0)  # Maximum power over frequency components
    
    # Use absolute values to avoid sqrt of negative numbers
    Envelope_values = np.sqrt(np.abs(mu_s)) * max_s  # Vocal envelope
    
    return spectrogram, labels, Envelope_values

# Load the initial spectrogram and calculate initial envelope values
spectrogram, labels, Envelope_values = load_spectrogram(current_file_index)
num_time_bins = spectrogram.shape[1]  # Number of time bins is along the second axis
mu_values = np.mean(spectrogram, axis=0)  # Calculate mean across frequency bins (axis=0)
max_power_value = np.max(spectrogram)
Envelope_values = mu_values * max_power_value

# Create a QApplication instance
app = QApplication([])

# Main window layout
win = QWidget()
layout = QVBoxLayout()
win.setLayout(layout)
win.setWindowTitle("Spectrogram Threshold Adjustment Tool")

# Graphics layout for plots
graphics_layout = pg.GraphicsLayoutWidget()
graphics_layout.ci.layout.setSpacing(0)  # Remove spacing between plots
layout.addWidget(graphics_layout)

# Plotting area for the spectrogram (top plot)
p1 = graphics_layout.addPlot(row=0, col=0, title="Spectrogram")
img = pg.ImageItem()
p1.addItem(img)
img.setImage(spectrogram.T)  # Transpose only for display
p1.invertY(False)  # Do not invert Y axis

# Plotting area for the labels (middle plot)
p_labels = graphics_layout.addPlot(row=1, col=0)
p_labels.setMaximumHeight(50)  # Limit the height to 50 pixels
p_labels.setXLink(p1)  # Link x-axis to p1
p_labels.hideAxis('left')
p_labels.hideAxis('bottom')

# Plotting area for the envelope (bottom plot)
p2 = graphics_layout.addPlot(row=2, col=0, title="Thresholded Sound Envelope")
p2.setXLink(p1)  # Link x-axis to p1
curve = p2.plot(Envelope_values, pen='b')

# Add a red line to indicate the threshold
threshold_line = pg.InfiniteLine(pos=initial_threshold, angle=0, pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine))
p2.addItem(threshold_line)

# Replace the QSpinBox with QDoubleSpinBox
threshold_input = QDoubleSpinBox()
threshold_input.setRange(-50.0, 50.0)
threshold_input.setSingleStep(0.1)  # Set step size for decimal increments
threshold_input.setDecimals(2)  # Set number of decimal places
threshold_input.setValue(initial_threshold)
layout.addWidget(threshold_input)

def update_plot(threshold, file_index):
    global spectrogram, labels, Envelope_values

    # Load spectrogram data for the current file
    spectrogram, labels, Envelope_values = load_spectrogram(file_index)

    # Update the spectrogram plot
    img.setImage(spectrogram.T)  # Transpose only for display

    # Ensure x-axis alignment
    p1.setXRange(0, num_time_bins, padding=0)

    # Update the envelope plot and show labels above threshold
    above_threshold = Envelope_values > threshold
    label_mask = labels > 0  # Assuming 0 is background/no label
    visible_mask = label_mask & above_threshold

    curve.setData(Envelope_values)
    p2.clear()
    p2.addItem(curve)
    p2.addItem(threshold_line)

    # Set fixed y-range for the envelope plot
    y_min = min(threshold - 1, np.min(Envelope_values))
    y_max = max(np.max(Envelope_values), threshold + 1)
    p2.setYRange(y_min, y_max)

    # Add rectangles for labels above threshold
    for start, end in get_contiguous_regions(visible_mask):
        rect = QGraphicsRectItem(start, threshold, end - start, y_max - threshold)
        rect.setBrush(pg.mkBrush(color='g', alpha=50))
        rect.setPen(pg.mkPen(None))
        p2.addItem(rect)

    # Update red labels bar
    p_labels.clear()
    for start, end in get_contiguous_regions(above_threshold):
        rect = QGraphicsRectItem(start, 0, end - start, 1)
        rect.setBrush(pg.mkBrush('r'))
        rect.setPen(pg.mkPen(None))
        p_labels.addItem(rect)

    threshold_line.setPos(threshold)
    p2.setTitle(f"File: {spectrogram_files[file_index]} - Threshold: {threshold:.2f}")

def get_contiguous_regions(mask):
    """Find contiguous True regions of the boolean array "mask"."""
    regions = []
    idx_start = None

    for idx, val in enumerate(mask):
        if val and idx_start is None:
            idx_start = idx
        elif not val and idx_start is not None:
            regions.append((idx_start, idx))
            idx_start = None

    if idx_start is not None:
        regions.append((idx_start, len(mask)))

    return regions

def threshold_changed():
    threshold = threshold_input.value()
    update_plot(threshold, current_file_index)

threshold_input.valueChanged.connect(threshold_changed)

# Function to save thresholds to a file
def save_thresholds():
    with open('thresholds.json', 'w') as f:
        json.dump(thresholds, f)
    print("Thresholds saved to 'thresholds.json'.")

# Function to handle key press events for file navigation
def key_pressed(evt):
    global current_file_index, initial_threshold
    if evt.key() == QtCore.Qt.Key_Right:
        if current_file_index < len(spectrogram_files) - 1:
            thresholds[spectrogram_files[current_file_index]] = threshold_input.value()
            current_file_index += 1
            initial_threshold = thresholds.get(spectrogram_files[current_file_index], initial_threshold)
            threshold_input.setValue(initial_threshold)
            update_plot(initial_threshold, current_file_index)

    elif evt.key() == QtCore.Qt.Key_Left:
        if current_file_index > 0:
            thresholds[spectrogram_files[current_file_index]] = threshold_input.value()
            current_file_index -= 1
            initial_threshold = thresholds.get(spectrogram_files[current_file_index], initial_threshold)
            threshold_input.setValue(initial_threshold)
            update_plot(initial_threshold, current_file_index)

    elif evt.key() == QtCore.Qt.Key_Q:
        thresholds[spectrogram_files[current_file_index]] = threshold_input.value()
        save_thresholds()
        app.quit()

# Add this function to load thresholds from a file
def load_thresholds():
    global thresholds
    try:
        with open('thresholds.json', 'r') as f:
            thresholds = json.load(f)
        print("Thresholds loaded from 'thresholds.json'.")
    except FileNotFoundError:
        print("No existing thresholds file found. Starting with empty thresholds.")

win.keyPressEvent = key_pressed

win.show()

if __name__ == '__main__':
    load_thresholds()  # Load existing thresholds
    initial_threshold = thresholds.get(spectrogram_files[current_file_index], initial_threshold)
    threshold_input.setValue(initial_threshold)
    update_plot(initial_threshold, current_file_index)
    app.exec_()
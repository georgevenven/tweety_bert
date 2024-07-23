#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 17:52:41 2023

@author: AnanyaKapoor
"""

"""
Code to plot just an audio/behavioral and embedding

"""

# Libraries/Settings
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from matplotlib import cm
from PyQt5.QtCore import Qt
import numpy as np
from PyQt5.QtCore import QPointF
import matplotlib.pyplot as plt
# -------------
from pyqtgraph import DateAxisItem, AxisItem, QtCore

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
class DataPlotter(QWidget):
    def __init__(self):
        QWidget.__init__(self) # TODO? parent needed? #awkward


        # Setup main window/layout
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.app = pg.mkQApp()


        # Instantiate window
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('Embedding Analysis')

        # Behave plot
        self.behavePlot = self.win.addPlot()


        # Define bottom plot 
        self.win.nextRow()
        self.embPlot = self.win.addPlot()


        self.setupPlot()
        
        self.win.scene().sigMouseClicked.connect(self.update2)    

        self.additionList = []
        self.additionCount = 0.05


    def setupPlot(self):

        # Setup behave plot for img
        self.imgBehave = pg.ImageItem()
        self.behavePlot.addItem(self.imgBehave)
        self.behavePlot.hideAxis('left')
        # self.behavePlot.hideAxis('bottom')
        self.behavePlot.setMouseEnabled(x=True, y=False)  # Enable x-axis and disable y-axis interaction


        # # Setup emb plot
        # self.embPlot.hideAxis('left')
        # self.embPlot.hideAxis('bottom')

    def clear_plots(self):
        self.embPlot.clear()
        self.behavePlot.clear()


    # Change to general one day?
    def set_behavioral_image(self,image_array,colors_per_timepoint, **kwargs):
        # print('replot')


        self.behave_array = image_array
        # if 'addition' in kwargs.keys():
        #     self.imgBehave.setImage(self.behave_array + kwargs['addition'])
        # else:
        #     self.imgBehave.setImage(self.behave_array)
            # self.imgBehave.setImage(np.stack([self.behave_array] * 3, axis=-1))

        filterSpec = image_array
        # Normalize the numeric array to the [0, 1] range
        normalized_array = (filterSpec - np.min(filterSpec)) / (np.max(filterSpec) - np.min(filterSpec))

        # Apply the colormap to the normalized array

        rgb_array = plt.cm.get_cmap('inferno')(normalized_array)
        rgb_add = np.zeros_like(image_array)
        
        colors = np.concatenate((colors_per_timepoint, np.ones((colors_per_timepoint.shape[0],1))), axis = 1)
        # colors*=255

        if 'addition' in kwargs.keys():
            relList = kwargs['addition']
            rgb_add1 = colors.copy()
            reshaped_colors = np.tile(rgb_add1, (rgb_array.shape[0], 1, 1))
            rgb_add1 = reshaped_colors.reshape(rgb_array.shape[0], rgb_array.shape[1], rgb_array.shape[2])
            print(rgb_add1.shape)
            for img in relList:
                print("IMAGE'S SHAPE")
                print(img.shape)
                
                # Some musings on how to extract the column for the first nonzero element (start of the windows) and the last nonzero element (end of the windows)
                
                # Assuming 'array' is your 2D NumPy array
                # flattened_array = img.flatten()
                
                # Find the index of the first nonzero element
                # first_nonzero_idx = np.argmax(flattened_array != 0)
                
                # Find the index of the last nonzero element
                # We reverse the array for this operation
                # last_nonzero_idx = flattened_array.size - 1 - np.argmax(flattened_array[::-1] != 0)
                
                # If needed, convert these indices to 2D coordinates
                # first_nonzero_2d = np.unravel_index(first_nonzero_idx, img.shape)[1]
                # last_nonzero_2d = np.unravel_index(last_nonzero_idx, img.shape)[1]
                
                # print(first_nonzero_2d)
                # print(last_nonzero_2d)
                
                # print("UNIQUE COLORS")
                # col_arr = self.colors_per_timepoint[first_nonzero_2d:last_nonzero_2d+1,:]
                # unique_rows, counts = np.unique(col_arr, axis=0, return_counts=True)
                # print(unique_rows)
                # print(counts)
                
                # Assuming 'array' is your 2D NumPy array
                # unique_rows, counts = np.unique(col_arr, axis=0, return_counts=True)
                # print(np.argmax(counts))
                # color_to_replace = np.append(unique_rows[np.argmax(counts)], 1)
                # print(color_to_replace)
                # rgb_add1[:,first_nonzero_2d:last_nonzero_2d+1, :] = color_to_replace
                # rgb_add[:,first_nonzero_2d:last_nonzero_2d+1] = unique_rows[np.argmax(counts)]
                # print('UNIQUE COUNTS')
                # print(counts)
                
                # Choose the unique row for which you want to find the indices
                # For example, let's say you want the indices of the first unique row
                # target_row = unique_rows[0]
                
                # Find the indices where this row occurs
                # indices = np.where((col_arr == target_row).all(axis=1))[0]
                # print(indices)
                # print(indices.shape)
                
                # unique_rows, unique_indices, occurrence_indices = np.unique(col_arr, axis=0, return_index=True, return_inverse=True)
                # print(unique_rows)
                # print(unique_indices)
                # print(occurrence_indices)
                # indices_of_chosen_unique = np.where(occurrence_indices == 0)[0]

                
                # # I want to fill the spots where there are silences with the other color
                # zero_columns_within_slice = np.all(img[:,first_nonzero_2d:last_nonzero_2d+1] == 0, axis = 0)
                # zero_columns_indices = np.where(zero_columns_within_slice)[0]
                # print("SILENCE INDICES")
                # print(zero_columns_indices)
                

                # print()
                # zero_columns = np.all(img == 0, axis=0)
                # zero_columns_indices = np.where(zero_columns)[0]
                # # print(zero_columns_indices)
                # print(np.diff(zero_columns_indices))
                # print(np.where(np.diff(zero_columns_indices)!=1))
                rgb_add += img 
            zero_columns = np.all(rgb_add == 0, axis=0)
            zero_columns_indices = np.where(zero_columns)[0]
            # zero_columns_indices = np.delete(zero_columns_indices, np.arange(first_nonzero_2d, last_nonzero_2d+1), axis = 0)

            rgb_add1[:,zero_columns_indices,:] = 0
            # rgb_add1[zero_columns_indices == 0,:] = 0

        else:
            rgb_add1 = np.zeros_like(colors)


        # rgb_add1 = colors.copy()

        # rgb_add1 = plt.cm.get_cmap('hsv')(rgb_add)
        # zero_columns = np.all(rgb_add == 0, axis=0)
        # rgb_add1[zero_columns == 0,:] = 0

        self.imgBehave.setImage(rgb_array + rgb_add1)


    def update(self):
        rgn = self.region.getRegion()

        findIndices = np.where(np.logical_and(self.startEndTimes[0,:] > rgn[0], self.startEndTimes[1,:] < rgn[1]))[0]
    
        self.newScatter.setData(pos = self.emb[findIndices,:])



        self.embPlot.setXRange(np.min(self.emb[:,0]) - 1, np.max(self.emb[:,0] + 1), padding=0)
        self.embPlot.setYRange(np.min(self.emb[:,1]) - 1, np.max(self.emb[:,1] + 1), padding=0)





    # Load the embedding and start times into scatter plot
    def accept_embedding(self,embedding,startEndTimes, mean_colors_for_minispec):

        self.emb = embedding
        self.startEndTimes = startEndTimes
        self.mean_colors_for_minispec = mean_colors_for_minispec


        # self.cmap = cm.get_cmap('hsv')
        # norm_times = np.arange(self.emb.shape[0])/self.emb.shape[0]
        
        colors = np.concatenate((mean_colors_for_minispec, np.ones((mean_colors_for_minispec.shape[0],1))), axis = 1)
        colors*=255
        # colors = self.cmap(norm_times) * 255
        self.defaultColors = colors.copy()
        self.scatter = pg.ScatterPlotItem(pos=embedding, size=5, brush=colors)
        self.embPlot.addItem(self.scatter)
        
        self.newScatter = pg.ScatterPlotItem(pos=embedding[0:10,:], size=10, brush=pg.mkBrush(255, 255, 255, 200))
        self.embPlot.addItem(self.newScatter)


        # Scale imgBehave 
        height,width = self.behave_array.shape

        x_start, x_end, y_start, y_end = 0, self.startEndTimes[1,-1], 0, height
        pos = [x_start, y_start]
        scale = [float(x_end - x_start) / width, float(y_end - y_start) / height]

        self.imgBehave.setPos(*pos)
        tr = QtGui.QTransform()
        self.imgBehave.setTransform(tr.scale(scale[0], scale[1]))
        self.behavePlot.getViewBox().setLimits(yMin=y_start, yMax=y_end)
        self.behavePlot.getViewBox().setLimits(xMin=x_start, xMax=x_end)

        # print(self.startEndTimes)
        self.region = pg.LinearRegionItem(values=(0, self.startEndTimes[0,-1] / 10))
        self.region.setZValue(10)

        
        self.region.sigRegionChanged.connect(self.update)

        self.behavePlot.addItem(self.region)

        # consider where    
        self.embMaxX = np.max(self.emb[:,0])
        self.embMaxY = np.max(self.emb[:,1])


        self.embMinX = np.min(self.emb[:,0])
        self.embMinY = np.min(self.emb[:,1])

        self.embPlot.setXRange(self.embMinX - 1, self.embMaxX + 1, padding=0)
        self.embPlot.setYRange(self.embMinY - 1, self.embMaxY + 1, padding=0)

    def plot_file(self,filePath):

        self.clear_plots()
        self.setupPlot()

        A = np.load(filePath)

        self.startEndTimes = A['embStartEnd']
        self.colors_per_timepoint = A['colors_per_timepoint']
        self.behavioralArr = A['behavioralArr']
        plotter.set_behavioral_image(A['behavioralArr'], A['colors_per_timepoint'])

        # feed it (N by 2) embedding and length N list of times associated with each point
        plotter.accept_embedding(A['embVals'],A['embStartEnd'], A['mean_colors_per_minispec'])

    def addROI(self):
        self.r1 = pg.EllipseROI([0, 0], [self.embMaxX/5, self.embMaxY/5], pen=(3,9))
        # r2a = pg.PolyLineROI([[0,0], [0,self.embMaxY/5], [self.embMaxX/5,self.embMaxY/5], [self.embMaxX/5,0]], closed=True)
        self.embPlot.addItem(self.r1)

        #self.r1.sigRegionChanged.connect(self.update2)

    # Manage key press events
    def keyPressEvent(self,evt):
        print('key is ',evt.key())

        if evt.key() == 65: # stick with numbers for now
            self.update()

    def update2(self):
        print('called')
        ellipse_size = self.r1.size()
        ellipse_center = self.r1.pos() + ellipse_size/2

        try:
            self.outCircles = np.vstack((self.outCircles,np.array([ellipse_center[0],ellipse_center[1],ellipse_size[0],ellipse_size[1]])))
        except:
            self.outCircles = np.array([ellipse_center[0],ellipse_center[1],ellipse_size[0],ellipse_size[1]])

        # print(self.outCircles)
        np.savez('bounds.npz',bounds = self.outCircles)
        # Print the center and size
        # print("Ellipse Center:", ellipse_center)
        # print("Ellipse Size:", ellipse_size)
        # print(self.r1)
        # print(ellipse_size[0])

        #manual distnace
        bound = np.square(self.emb[:,0] - ellipse_center[0])/np.square(ellipse_size[0]/2) +  np.square(self.emb[:,1] - ellipse_center[1])/np.square(ellipse_size[1]/2)
        indices_in_roi = np.where(bound < 1)[0]
        # print(f'The number of indices in the ROI is {indices_in_roi.shape}')
        print(indices_in_roi)





        # # points_in_roi = [QPointF(x, y) for x, y in self.emb if self.r1.contains(QPointF(x, y))]
        # # print(points_in_roi)
        # print('does it contian 0,0')
        # if self.r1.contains(QPointF(0,0)):
        #     print('yes')

        # # indices_in_roi = [pt for pt in self.emb if roiShape.contains(pt)]
        # # print(roiShape.pos())
        # indices_in_roi = [index for index, (x, y) in enumerate(self.emb) if self.r1.contains(QPointF(x, y))]
        # print(indices_in_roi)
        # # print(indices_in_roi)
        # # print(self.emb.shape)

        tempImg = self.behave_array.copy()*0
        presumedTime = np.linspace(self.startEndTimes[0,0],self.startEndTimes[1,-1],num = tempImg.shape[1])
        # print(f'The Presumed Time: {presumedTime}')
        # print(self.startEndTimes.shape)
        # print(self.behave_array.shape)


        for index in indices_in_roi:
            # For each index in the ROI, extract the associated spec slice
            mask = (presumedTime < self.startEndTimes[1,index]) & (presumedTime > self.startEndTimes[0,index])
            print("MASK SHAPE")
            print(mask.shape)
            relPlace = np.where(mask)[0]
            print("RELPLACE")
            print(relPlace)
            print(relPlace.shape)
            tempImg[:,relPlace] = self.additionCount
            # print("WHAT IS ADDITION COUNT")
            # print(self.additionCount)

        self.additionList.append(tempImg)
        # print(f'The Shape of the Temporary Image: {tempImg.shape}')
        # print(f'The Length of the Addition List: {len(self.additionList)}')
        self.additionCount += .05

        self.set_behavioral_image(self.behave_array,self.colors_per_timepoint, addition = self.additionList)


        # self.newScatter.setData(pos = self.emb[indices_in_roi,:])



        # self.embPlot.setXRange(np.min(self.emb[:,0]) - 1, np.max(self.emb[:,0] + 1), padding=0)
        # self.embPlot.setYRange(np.min(self.emb[:,1]) - 1, np.max(self.emb[:,1] + 1), padding=0)

    def show(self):
        self.win.show()
        self.app.exec_()


# IDEA (iterate through bouts..)
if __name__ == '__main__':
    app = QApplication([])
    # Instantiate the plotter    
    plotter = DataPlotter()

    # Accept folder of data
    #plotter.accept_folder('SortedResults/B119-Jul28')
    #/Users/ethanmuchnik/Desktop/Series_GUI/SortedResults/Pk146-Jul28/1B.npz
    #plotter.plot_file('/Users/ethanmuchnik/Desktop/Series_GUI/SortedResults/Pk146-Jul28/1B.npz')
    plotter.plot_file('/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/umap_dict_file.npz')
    # plotter.plot_file('working.npz')

    plotter.addROI()

    # Show
    plotter.show()
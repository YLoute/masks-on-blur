######
# Created to generate masks with Laplacian Pyramid method on blurred areas for close-range Structure from Motion (SfM)
# Bonus: refine detection of AprilTags centers and write it in csv files
#
# Author: Yannick FAURE
# Licence: GPL v3
#
# Caution : This python code is not optimized and ressembles as a draft, however it works as intended.
######
# Main file to run GUI to compute masks and detect apriltags 36h11 on entire folder, with variables.
# Contains functions to run on entire folder to compute masks and tags
######


# Qt User Interface Compiler version 6.4.3
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient, 
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QLabel, QMainWindow,
    QMenu, QMenuBar, QPlainTextEdit, QSizePolicy, QPushButton, QCheckBox,
    QVBoxLayout, QHBoxLayout, QSlider, QFrame,
    QSpinBox, QDoubleSpinBox, QStatusBar, QWidget)

from plyer import filechooser
from os import listdir
from os.path import isfile, join, basename
import cv2
import numpy as np
from pathlib import Path
import time

import modules.on_crop_compute_sobel as on_crop_compute_sobel
import modules.exif_changer as exif_changer

import modules.laplacian_pyramids_and_morpho as LP_morpho # TODO 
import modules.april_tags_36h11 as ap

class Ui_MainWindow(object):
    user_selected_folder = ""
    image_files = []
    LaplacianPy_defaultValue = 1
    BlurThreshold_defaultValue = 5
    """dilate_kernel_size_1_defaultValue = 100
    erode_kernel_size_2_defaultValue = 95
    dilate_kernel_size_3_defaultValue = 30"""
    dilate_kernel_size_1_defaultValue = 51
    erode_kernel_size_2_defaultValue = 52
    dilate_kernel_size_3_defaultValue = 0
    Opacity_defaultValue = 40
    Opacity2_defaultValue = 0 ### TODO change to 20
    Gamma2_defaultValue = 0
    tags = False
    tags_all_mask = []
    tags_all_mask_params = []
    current_tag_index = False
    
    def setupUi(self, MainWindow):
        self.window_width = 1900
        self.window_height = 1000
        """
MainWindow """
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(self.window_width, self.window_height)
        icon = QIcon('icon_69_alpha_crop.png')
        MainWindow.setWindowIcon(icon)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        
        """
> centralwidget (in MainWindow (under potential Menu (QAction)) (QWidget) """
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        
        """
> centralwidget (QWidget)
    > verticalLayoutWidget (QWidget) """
        self.verticalLayoutWidget = QWidget(self.centralwidget)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(0, 0, self.window_width, self.window_height))
        """self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)"""
        
        """
> centralwidget (QWidget)
    > verticalLayoutWidget (QWidget) 
        > hz1LayoutWidget (QWidget) """
        self.hz1LayoutWidget = QWidget(self.verticalLayoutWidget)
        self.hz1LayoutWidget.setObjectName(u"hz1LayoutWidget")
        self.hz1LayoutWidget.setGeometry(QRect(0, 0, self.window_width, 20))
        
        # Variables which will evolve
        bar_pos_x = -1
        bar_pos_y = -1
        bar_size_x = 0
        bar_size_y= 22
        bar_space_x = 25
        
        """
> MainWindow (QWidget)
    > centralwidget (QComboBox)
        > hz1LayoutWidget (QWidget)
            > pushButton_Browse (QPushButton) """
        self.pushButton_Browse = QPushButton(self.hz1LayoutWidget)
        self.pushButton_Browse.setObjectName(u"pushButton_Browse")
        bar_size_x = 110
        self.pushButton_Browse.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        
        """
> centralwidget (QWidget)
    > verticalLayoutWidget (QWidget)
        > hz1LayoutWidget (QWidget)
            > comboBox_BrowseFiles (QComboBox)"""
        self.comboBox_BrowseFiles = QComboBox(self.hz1LayoutWidget)
        self.comboBox_BrowseFiles.setObjectName(u"comboBox_BrowseFiles")
        bar_pos_x += bar_size_x
        bar_size_x = 200
        self.comboBox_BrowseFiles.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        
        
        self.pushButton_ExifR = QPushButton(self.hz1LayoutWidget)
        self.pushButton_ExifR.setObjectName(u"pushButton_ExifR")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 80
        self.pushButton_ExifR.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.pushButton_ExifR.setText("Clean Rot Exif")
        self.pushButton_ExifR.setStatusTip("/!\\ SAVE YOUR ORIGINAL FILES ELSEWHERE BEFORE /!\\"
        + "Clean Rotation tag from Exif and save new files without loosing in quality. "
        + "If button is red, there are some rotation tags, so clean it before generating anything else.")
        
        
        """
> centralwidget (QWidget)
    > verticalLayoutWidget (QWidget)
        > hz1LayoutWidget (QWidget)
            > ...
            > label_LaplacianPy (QWidget)
            > spinBox_LaplacianPy (QSplinBox) """        
        
        self.label_LaplacianPy = QLabel(self.hz1LayoutWidget)
        self.label_LaplacianPy.setObjectName(u"label_LaplacianPy")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 132
        self.label_LaplacianPy.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_LaplacianPy = QSpinBox(self.hz1LayoutWidget)
        self.spinBox_LaplacianPy.setObjectName(u"spinBox_LaplacianPy")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_LaplacianPy.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_LaplacianPy.setMinimum(1)
        
        self.label_BlurThreshold = QLabel(self.hz1LayoutWidget)
        self.label_BlurThreshold.setObjectName(u"label_BlurThreshold")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 121
        self.label_BlurThreshold.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_BlurThreshold = QSpinBox(self.hz1LayoutWidget)
        self.spinBox_BlurThreshold.setObjectName(u"spinBox_BlurThreshold")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_BlurThreshold.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_BlurThreshold.setMinimum(0)
        self.spinBox_BlurThreshold.setMaximum(255)
        
        self.label_dilate1 = QLabel(self.hz1LayoutWidget)
        self.label_dilate1.setObjectName(u"label_dilate1")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 75
        self.label_dilate1.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_dilate1 = QSpinBox(self.hz1LayoutWidget)
        self.spinBox_dilate1.setObjectName(u"spinBox_dilate_final")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_dilate1.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_dilate1.setMinimum(0)
        self.spinBox_dilate1.setMaximum(1000)
        self.spinBox_dilate1.setSingleStep(5)
        
        self.label_erode2 = QLabel(self.hz1LayoutWidget)
        self.label_erode2.setObjectName(u"label_erode2")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 81
        self.label_erode2.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_erode2 = QSpinBox(self.hz1LayoutWidget)
        self.spinBox_erode2.setObjectName(u"spinBox_erode2")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_erode2.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_erode2.setMinimum(0)
        self.spinBox_erode2.setMaximum(1000)
        self.spinBox_erode2.setSingleStep(5)
        
        self.label_dilate3 = QLabel(self.hz1LayoutWidget)
        self.label_dilate3.setObjectName(u"label_dilate3")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 77
        self.label_dilate3.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_dilate3 = QSpinBox(self.hz1LayoutWidget)
        self.spinBox_dilate3.setObjectName(u"spinBox_dilate3")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_dilate3.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_dilate3.setMinimum(0)
        self.spinBox_dilate3.setMaximum(1000)
        self.spinBox_dilate3.setSingleStep(5)
        
        self.label_bilan_dilate = QLabel(self.hz1LayoutWidget)
        self.label_bilan_dilate.setObjectName(u"label_bilan_dilate")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 20
        self.label_bilan_dilate.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        
        
        
        # Text only
        self.label_units = QLabel(self.hz1LayoutWidget)
        self.label_units.setObjectName(u"label_units")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 50
        fontbold = QFont()
        fontbold.setBold(True)
        self.label_units.setFont(fontbold)
        self.label_units.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        
        # Back to variables
        self.label_Opacity = QLabel(self.hz1LayoutWidget)
        self.label_Opacity.setObjectName(u"label_Opacity")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 48
        self.label_Opacity.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_Opacity = QSpinBox(self.hz1LayoutWidget)
        self.spinBox_Opacity.setObjectName(u"spinBox_Opacity")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_Opacity.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_Opacity.setMinimum(0)
        self.spinBox_Opacity.setMaximum(100)
        self.spinBox_Opacity.setSingleStep(10)
        
        self.label_Opacity2 = QLabel(self.hz1LayoutWidget)
        self.label_Opacity2.setObjectName(u"label_Opacity2")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 68
        self.label_Opacity2.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_Opacity2 = QSpinBox(self.hz1LayoutWidget)
        self.spinBox_Opacity2.setObjectName(u"spinBox_Opacity2")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_Opacity2.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_Opacity2.setMinimum(0)
        self.spinBox_Opacity2.setMaximum(100)
        self.spinBox_Opacity2.setSingleStep(5)
        
        self.label_Gamma2 = QLabel(self.hz1LayoutWidget)
        self.label_Gamma2.setObjectName(u"label_Gamma2")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 68
        self.label_Gamma2.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_Gamma2 = QSpinBox(self.hz1LayoutWidget)
        self.spinBox_Gamma2.setObjectName(u"spinBox_Gamma2")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_Gamma2.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_Gamma2.setMinimum(0)
        self.spinBox_Gamma2.setMaximum(100)
        self.spinBox_Gamma2.setSingleStep(10)
        
        self.pushButton_Show_l1 = QPushButton(self.hz1LayoutWidget)
        self.pushButton_Show_l1.setObjectName(u"pushButton_Show")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 50
        self.pushButton_Show_l1.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        
        self.pushButton_Reset_l1 = QPushButton(self.hz1LayoutWidget)
        self.pushButton_Reset_l1.setObjectName(u"pushButton_Reset")
        bar_pos_x += bar_size_x
        bar_size_x = 40
        self.pushButton_Reset_l1.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
    
        # Line 2 : AprilTags
        """
> centralwidget (QWidget)
    > verticalLayoutWidget (QWidget) 
        > hz1LayoutWidget (QWidget) """
        self.hz2LayoutWidget = QWidget(self.verticalLayoutWidget)
        self.hz2LayoutWidget.setObjectName(u"hz2LayoutWidget")
        self.hz2LayoutWidget.setGeometry(QRect(0, 20, self.window_width, 20))
        
        # Variables which will evolve
        bar_pos_x = -1
        bar_pos_y = -1
        bar_size_x = 0
        bar_size_y= 22
        bar_space_x = 25
        
        """
> centralwidget (QWidget)
    > verticalLayoutWidget (QWidget)
        > hz2LayoutWidget (QWidget)
            > pushButton_Detect (QPushButton) """
        self.pushButton_DetectAprilTags = QPushButton(self.hz2LayoutWidget)
        self.pushButton_DetectAprilTags.setObjectName(u"pushButton_DetectAprilTags")
        bar_size_x = 95
        self.pushButton_DetectAprilTags.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.pushButton_DetectAprilTags.setStatusTip("/!\\ Dedicated to work on AprilTags 36h11 with checkerboard pattern at the center. Either it will generate false detections /!\\")
        
        self.pushButton_Reset_l2 = QPushButton(self.hz2LayoutWidget)
        self.pushButton_Reset_l2.setObjectName(u"pushButton_Reset_l2")
        bar_pos_x += bar_size_x
        bar_size_x = 70
        self.pushButton_Reset_l2.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        
        self.label_refineEdges = QLabel(self.hz2LayoutWidget)
        self.label_refineEdges.setObjectName(u"label_refineEdges")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 67
        self.label_refineEdges.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.label_refineEdges.setText("refineEdges")
        self.label_refineEdges.setStatusTip(u"When true, the edges of the each quad are adjusted to \"snap to\" strong gradients nearby. This is useful "+
                                            "when decimation is employed, as it can increase the quantity of the initial quad estimate substantially. "+
                                            "Generally (Default) recommended to be on (True). Trying False on purpose to not induce False detections of not flat targets.")
        
        self.checkBox_refineEdges = QCheckBox(self.hz2LayoutWidget)
        self.checkBox_refineEdges.setObjectName(u"checkBox_refineEdges")
        bar_pos_x += bar_size_x
        bar_size_x = 22
        self.checkBox_refineEdges.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        
        self.label_decodeSharpening = QLabel(self.hz2LayoutWidget)
        self.label_decodeSharpening.setObjectName(u"label_decodeSharpening")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 106
        self.label_decodeSharpening.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.label_decodeSharpening.setText("decodeSharpening")
        self.label_decodeSharpening.setStatusTip(u"How much sharpening should be done to decoded images. This can help decode small tags but may or may not "+
                                                 "help in odd lighting conditions or low light conditions. Default is 0.25. Max sharpened?=1 ?")
        
        self.spinBox_decodeSharpening = QDoubleSpinBox(self.hz2LayoutWidget)
        self.spinBox_decodeSharpening.setObjectName(u"spinBox_decodeSharpening")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_decodeSharpening.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_decodeSharpening.setDecimals(2)
        self.spinBox_decodeSharpening.setSingleStep(0.05)
        
        self.label_quadDecimate = QLabel(self.hz2LayoutWidget)
        self.label_quadDecimate.setObjectName(u"label_quadDecimate")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 83
        self.label_quadDecimate.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.label_quadDecimate.setText("quadDecimate")
        self.label_quadDecimate.setFont(fontbold)
        self.label_quadDecimate.setStatusTip(u"It's the resize factor before detection (not impacting binary decoding (=payload) still full resolution. "+
                                             "High value = quick but will miss some targets. Default is 2.")
        self.spinBox_quadDecimate = QDoubleSpinBox(self.hz2LayoutWidget)
        self.spinBox_quadDecimate.setObjectName(u"spinBox_quadDecimate")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_quadDecimate.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_quadDecimate.setDecimals(1)
        self.spinBox_quadDecimate.setSingleStep(0.1)
        
        self.label_quadSigma = QLabel(self.hz2LayoutWidget)
        self.label_quadSigma.setObjectName(u"label_quadSigma")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 67
        self.label_quadSigma.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.label_quadSigma.setText("quadSigma")
        self.label_quadSigma.setStatusTip(u"What Gaussian blur should be applied to the segmented image (used for quad detection). "+
                                          "Very noisy images benefit from non-zero values (e.g. 0.8) Default is 0.0.")
        self.spinBox_quadSigma = QDoubleSpinBox(self.hz2LayoutWidget)
        self.spinBox_quadSigma.setObjectName(u"spinBox_quadSigma")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_quadSigma.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_quadSigma.setDecimals(1)
        self.spinBox_quadSigma.setSingleStep(0.1)
        
        self.label_quadThreshold_criticalAngle = QLabel(self.hz2LayoutWidget)
        self.label_quadThreshold_criticalAngle.setObjectName(u"label_quadThreshold_criticalAngle")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 154
        self.label_quadThreshold_criticalAngle.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.label_quadThreshold_criticalAngle.setText("quadThreshold.criticalAngle")
        self.label_quadThreshold_criticalAngle.setFont(fontbold)
        self.label_quadThreshold_criticalAngle.setStatusTip(u"The detector will reject quads where pairs of edges have angles that are close to straight "+
                                                            "or close to 180 degrees. Zero means that no quads are rejected. Default is π/4 = 0.785")
        self.spinBox_quadThreshold_criticalAngle = QDoubleSpinBox(self.hz2LayoutWidget)
        self.spinBox_quadThreshold_criticalAngle.setObjectName(u"spinBox_quadThreshold_criticalAngle")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_quadThreshold_criticalAngle.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_quadThreshold_criticalAngle.setDecimals(3)
        self.spinBox_quadThreshold_criticalAngle.setSingleStep(0.098)
        self.spinBox_quadThreshold_criticalAngle.setMaximum(1.571)
        
        self.label_maxLineFitMSE = QLabel(self.hz2LayoutWidget)
        self.label_maxLineFitMSE.setObjectName(u"label_maxLineFitMSE")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 90
        self.label_maxLineFitMSE.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.label_maxLineFitMSE.setText(".maxLineFitMSE")
        self.label_maxLineFitMSE.setStatusTip(u"When fitting lines to the contours, the maximum mean squared error allowed. "+
                                              "This is useful in rejecting contours that are far from being quad shaped; "+
                                              "rejecting these quads \"early\" saves expensive decoding processing. Default is 10.0.")
        self.spinBox_maxLineFitMSE = QDoubleSpinBox(self.hz2LayoutWidget)
        self.spinBox_maxLineFitMSE.setObjectName(u"spinBox_maxLineFitMSE")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_maxLineFitMSE.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_maxLineFitMSE.setDecimals(1)
        self.spinBox_maxLineFitMSE.setSingleStep(0.5)
        
        self.label_maxNumMaxima = QLabel(self.hz2LayoutWidget)
        self.label_maxNumMaxima.setObjectName(u"label_maxNumMaxima")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 100
        self.label_maxNumMaxima.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.label_maxNumMaxima.setText(".maxNumMaxima")
        self.label_maxNumMaxima.setStatusTip(u"How many corner candidates to consider when segmenting a group of pixels into a quad. Default is 10.")
        self.spinBox_maxNumMaxima = QSpinBox(self.hz2LayoutWidget)
        self.spinBox_maxNumMaxima.setObjectName(u"spinBox_mmaxNumMaxima")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_maxNumMaxima.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_maxNumMaxima.setSingleStep(1)
        
        self.label_minClusterPixels = QLabel(self.hz2LayoutWidget)
        self.label_minClusterPixels.setObjectName(u"label_minClusterPixels")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 96
        self.label_minClusterPixels.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.label_minClusterPixels.setText(".minClusterPixels")
        self.label_minClusterPixels.setStatusTip(u"Min number of pixels inside a quad to consider it as one. "+
                                                 "Threshold used to reject quads containing too few pixels. Default is 300 pixels.")
        self.spinBox_minClusterPixels = QSpinBox(self.hz2LayoutWidget)
        self.spinBox_minClusterPixels.setObjectName(u"spinBox_minClusterPixels")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_minClusterPixels.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_minClusterPixels.setSingleStep(1)
        self.spinBox_minClusterPixels.setMaximum(999)
        
        self.label_minWhiteBlackDiff = QLabel(self.hz2LayoutWidget)
        self.label_minWhiteBlackDiff.setObjectName(u"label_minWhiteBlackDiff")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 100
        self.label_minWhiteBlackDiff.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.label_minWhiteBlackDiff.setText(".maxNumMaxima")
        self.label_minWhiteBlackDiff.setStatusTip(u"Minimum brightness offset/difference W/B to detect a tag?. When we build our model of black & white pixels, we add an extra check "+
                                                  "that the white model must be (overall) brighter than the black model. How much brighter (in pixel value, [0,255]). "+
                                                  "Default is 5.")
        self.spinBox_minWhiteBlackDiff = QSpinBox(self.hz2LayoutWidget)
        self.spinBox_minWhiteBlackDiff.setObjectName(u"spinBox_minWhiteBlackDiff")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_minWhiteBlackDiff.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_minWhiteBlackDiff.setMinimum(0)
        self.spinBox_minWhiteBlackDiff.setMaximum(255)
        self.spinBox_minWhiteBlackDiff.setSingleStep(5)
        
        
        
        #bar_pos_x = 1500
        bar_pos_x += bar_size_x + bar_space_x * 3
        bar_size_x = 90
        self.label_AprilTag_current = QLabel(self.hz2LayoutWidget)
        self.label_AprilTag_current.setObjectName(u"label_AprilTag_current")
        self.label_AprilTag_current.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        
        self.pushButton_AprilTag_previous = QPushButton(self.hz2LayoutWidget)
        self.pushButton_AprilTag_previous.setObjectName(u"AprilTag_previous")
        bar_pos_x += bar_size_x
        bar_size_x = 25
        self.pushButton_AprilTag_previous.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.pushButton_AprilTag_next = QPushButton(self.hz2LayoutWidget)
        self.pushButton_AprilTag_next.setObjectName(u"AprilTag_next")
        bar_pos_x += bar_size_x
        bar_size_x = 25
        self.pushButton_AprilTag_next.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        
        # Line 3 : save params, load, run on entire folder with these params
        """
> centralwidget (QWidget)
    > verticalLayoutWidget (QWidget) 
        > hz1LayoutWidget (QWidget) """
        self.hz3LayoutWidget = QWidget(self.verticalLayoutWidget)
        self.hz3LayoutWidget.setObjectName(u"hz3LayoutWidget")
        self.hz3LayoutWidget.setGeometry(QRect(0, 40, self.window_width, 20))
        
        # Variables which will evolve
        bar_pos_x = 3
        bar_pos_y = -1
        bar_size_x = 0
        bar_size_y= 22
        bar_space_x = 25
        """
> MainWindow (QWidget)
    > centralwidget (QComboBox)
        > hz1LayoutWidget (QWidget)
            > pushButton_Browse (QPushButton) """
        # Last parameters for tag generation
        self.label_CropBoxFactor = QLabel(self.hz3LayoutWidget)
        self.label_CropBoxFactor.setObjectName(u"label_CropBoxFactor")
        bar_size_x = 136
        self.label_CropBoxFactor.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.label_CropBoxFactor.setText("Box crop factor to refine")
        self.label_CropBoxFactor.setStatusTip(u"Search zone: factor of the box found by corner. 0.1 is good. There is an absolute minimal value of 30 pixels in each direction anyway.")
        self.spinBox_CropBoxFactor = QDoubleSpinBox(self.hz3LayoutWidget)
        self.spinBox_CropBoxFactor.setObjectName(u"spinBox_CropBoxFactor")
        bar_pos_x += bar_size_x
        bar_size_x = 42
        self.spinBox_CropBoxFactor.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.spinBox_CropBoxFactor.setMinimum(0.01)
        self.spinBox_CropBoxFactor.setMaximum(1)
        self.spinBox_CropBoxFactor.setDecimals(2)
        self.spinBox_CropBoxFactor.setSingleStep(0.01)
        
        self.pushButton_SaveMasksParams = QPushButton(self.hz3LayoutWidget)
        self.pushButton_SaveMasksParams.setObjectName(u"pushButton_SaveMasksParams")
        bar_pos_x += bar_size_x + bar_space_x * 2
        bar_size_x = 130
        self.pushButton_SaveMasksParams.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.pushButton_SaveMasksParams.setText(u"Save masks params")
        
        self.pushButton_LoadMasksParams = QPushButton(self.hz3LayoutWidget)
        self.pushButton_LoadMasksParams.setObjectName(u"pushButton_LoadMasksParams")
        bar_pos_x += bar_size_x
        bar_size_x = 130
        self.pushButton_LoadMasksParams.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.pushButton_LoadMasksParams.setText(u"Load masks params")
        
        self.pushButton_SaveAprilTagsParams = QPushButton(self.hz3LayoutWidget)
        self.pushButton_SaveAprilTagsParams.setObjectName(u"pushButton_SaveAprilTagsParams")
        bar_pos_x += bar_size_x + bar_space_x
        bar_size_x = 130
        self.pushButton_SaveAprilTagsParams.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.pushButton_SaveAprilTagsParams.setText(u"Save AprilTags params")
        
        self.pushButton_LoadAprilTagsParams = QPushButton(self.hz3LayoutWidget)
        self.pushButton_LoadAprilTagsParams.setObjectName(u"pushButton_LoadAprilTagsParams")
        bar_pos_x += bar_size_x
        bar_size_x = 130
        self.pushButton_LoadAprilTagsParams.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.pushButton_LoadAprilTagsParams.setText(u"Load AprilTags params")
        
        self.pushButton_RunMasks = QPushButton(self.hz3LayoutWidget)
        self.pushButton_RunMasks.setObjectName(u"pushButton_RunMasks")
        bar_pos_x += bar_size_x + bar_space_x *2
        bar_size_x = 200
        self.pushButton_RunMasks.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.pushButton_RunMasks.setText(u"Generate masks on entire folder")
        self.pushButton_RunMasks.setFont(fontbold)
        
        self.pushButton_RunAprilTags = QPushButton(self.hz3LayoutWidget)
        self.pushButton_RunAprilTags.setObjectName(u"pushButton_RunAprilTags")
        bar_pos_x += bar_size_x
        bar_size_x = 200
        self.pushButton_RunAprilTags.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.pushButton_RunAprilTags.setText(u"Gene tags coords on entire folder")
        self.pushButton_RunAprilTags.setFont(fontbold)
        
        self.pushButton_RunSobel = QPushButton(self.hz3LayoutWidget)
        self.pushButton_RunSobel.setObjectName(u"pushButton_Sobel")
        bar_pos_x += bar_size_x
        bar_size_x = 200
        self.pushButton_RunSobel.setGeometry(QRect(bar_pos_x, bar_pos_y, bar_size_x, bar_size_y))
        self.pushButton_RunSobel.setText(u"Gene Sobel on entire folder")
        self.pushButton_RunSobel.setFont(fontbold)
        
        # Info
        self.label_Info = QLabel(self.centralwidget)
        self.label_Info.setObjectName(u"label_Info")
        self.label_Info.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        self.label_Info.setGeometry(QRect(4, 70, 1200, self.window_height - 90))
        fontbigbold = QFont("Times", 10, QFont.Bold)
        self.label_Info.setFont(fontbigbold)
        
        
        
        """
> centralwidget (QWidget)
    > Pictures... """
        self.label_OriginalPicture = QLabel(self.centralwidget)
        self.label_OriginalPicture.setObjectName(u"label_OriginalPicture")
        self.label_OriginalPicture.setGeometry(QRect(0, 60, 1350, 900))
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label_OriginalPicture.sizePolicy().hasHeightForWidth())
        self.label_OriginalPicture.setSizePolicy(sizePolicy1)
        self.label_OriginalPicture.setPixmap(QPixmap(u"../photogrammetrie/chapelle_bonnebos_musee_de_la_resistance_20240417/IMG_9473.JPG"))
        self.label_OriginalPicture.setScaledContents(True)
        
        self.label_OriginalPicture_Scaled = QLabel(self.centralwidget)
        self.label_OriginalPicture_Scaled.setObjectName(u"label_OriginalPicture_Scaled")
        self.label_OriginalPicture_Scaled.setGeometry(QRect(1350+15, 60, 1900-1350-15, 900))
        sizePolicy1.setHeightForWidth(self.label_OriginalPicture_Scaled.sizePolicy().hasHeightForWidth())
        self.label_OriginalPicture_Scaled.setSizePolicy(sizePolicy1)
        self.label_OriginalPicture_Scaled.setFocusPolicy(Qt.NoFocus)
        self.label_OriginalPicture_Scaled.setPixmap(QPixmap(u"../photogrammetrie/chapelle_bonnebos_musee_de_la_resistance_20240417/IMG_9473.JPG"))
        self.label_OriginalPicture_Scaled.setAlignment(Qt.AlignCenter)
        
        
        self.verticalSlider = QSlider(self.centralwidget)
        self.verticalSlider.setObjectName(u"verticalSlider")
        self.verticalSlider.setGeometry(QRect(1900-40, 100, 22, 160))
        self.verticalSlider.setOrientation(Qt.Vertical)
        self.verticalSlider.setValue(50) # 50 / 100, 0 is bottom
        
        self.hzSlider = QSlider(self.centralwidget)
        self.hzSlider.setObjectName(u"hzSlider")
        self.hzSlider.setGeometry(QRect(1900-280, 80, 240, 22))
        self.hzSlider.setOrientation(Qt.Horizontal)
        self.hzSlider.setValue(50) # 50 / 100, 0 is bottom
        
        self.reset_l1()
        self.reset_l2()
        
        
        MainWindow.setCentralWidget(self.centralwidget)

        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)
        
    # setupUi
    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Analysis tool - Choose your masks parameters & detect AprilTags 36h11 - by Yannick Faure", None))
        
        self.pushButton_Browse.setText(QCoreApplication.translate("MainWindow", u"Choose your folder", None))
        self.pushButton_Browse.clicked.connect(self.open_file_menu)
        self.pushButton_ExifR.clicked.connect(self.exifCleanRot)
        
        self.label_LaplacianPy.setText(QCoreApplication.translate("MainWindow", u"Laplacian pyramid level", None))
        self.label_BlurThreshold.setText(QCoreApplication.translate("MainWindow", u"Blur Threshold [0;255]", None))
        self.label_dilate1.setText(QCoreApplication.translate("MainWindow", u"1st dilate size", None))
        self.label_erode2.setText(QCoreApplication.translate("MainWindow", u"2nd erode size", None))
        self.label_dilate3.setText(QCoreApplication.translate("MainWindow", u"3rd dilate size", None))
        self.label_bilan_dilate.setText(str(self.spinBox_dilate1.value() - self.spinBox_erode2.value() + self.spinBox_dilate3.value()))
        self.label_units.setText(QCoreApplication.translate("MainWindow", u"[px] | [%]", None))
        self.label_Opacity.setText(QCoreApplication.translate("MainWindow", u"Opacity", None))
        self.label_Opacity2.setText(QCoreApplication.translate("MainWindow", u"Opacity 1:1", None))
        self.label_Gamma2.setText(QCoreApplication.translate("MainWindow", u"Gamma 1:1", None))
        
        self.spinBox_Opacity.valueChanged.connect(self.opacity_change_left)
        self.spinBox_Opacity2.valueChanged.connect(self.opacity_change_right)
        self.spinBox_Gamma2.valueChanged.connect(self.opacity_change_right)
        
        self.pushButton_Show_l1.setText(QCoreApplication.translate("MainWindow", u"Show", None))
        self.pushButton_Show_l1.clicked.connect(self.on_parameters_change)
        self.pushButton_Reset_l1.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.pushButton_Reset_l1.clicked.connect(self.reset_l1)
        
        self.label_OriginalPicture.setText("")
        self.label_OriginalPicture_Scaled.setText("")
        
        
        self.pushButton_DetectAprilTags.setText(QApplication.translate("MainWindow", u"Detect AprilTags", None))
        self.pushButton_DetectAprilTags.clicked.connect(self.detect_apriltags)
        self.pushButton_Reset_l2.setText(QCoreApplication.translate("MainWindow", u"Res.params", None))
        self.pushButton_Reset_l2.clicked.connect(self.reset_l2)
        
        
        
        self.label_AprilTag_current.setText(QCoreApplication.translate("MainWindow", u"None (center)", None))
        self.pushButton_AprilTag_previous.setText(QApplication.translate("MainWindow", u"<", None))
        self.pushButton_AprilTag_previous.clicked.connect(self.apriltag_previous)
        self.pushButton_AprilTag_next.setText(QApplication.translate("MainWindow", u">", None))
        self.pushButton_AprilTag_next.clicked.connect(self.apriltag_next)
        
        self.verticalSlider.valueChanged.connect(self.move_zoom)
        self.hzSlider.valueChanged.connect(self.move_zoom)

        
        self.comboBox_BrowseFiles.currentTextChanged.connect(self.first_layering)
        
        
        self.pushButton_RunMasks.clicked.connect(self.runMasks)
        self.pushButton_RunAprilTags.clicked.connect(self.runTags)
        self.pushButton_RunSobel.clicked.connect(self.runSobel)
        
        

    def open_file_menu(self, *args):
        filechooser.choose_dir(on_selection=self.handle_selection)
        
    def handle_selection(self, selection):
        if selection:
            self.user_selected_folder = selection[0]
            self.load_image_files()
            self.update_comboBox() # Liste déroulante

    def load_image_files(self):
        # Load images names from chosen folder
        if self.user_selected_folder:
            self.image_files = [f for f in listdir(self.user_selected_folder) if isfile(join(self.user_selected_folder, f)) and f.lower().endswith(('.jpg', '.JPG'))]
                
    def update_comboBox(self):
        # Update comboBox with images names from the chose folder, clear it first
        self.comboBox_BrowseFiles.clear()
        self.comboBox_BrowseFiles.insertItems(0, self.image_files) # (index, string)
        self.exifRotTest()
        
    def first_layering(self):
        self.tags = False
        self.tags_all_mask = []
        self.tags_all_mask_params = []
        self.current_tag_index = False
        self.label_AprilTag_current.setText(QCoreApplication.translate("MainWindow", u"None (center)", None))
        self.verticalSlider.setValue(50)
        self.hzSlider.setValue(50)
        self.layering()
        self.label_Info.setText("")
        self.label_Info.raise_()
    
    def mask_create(self, folder, image):
        image_path = join(folder, image)
        img_base = cv2.imread(image_path)
        
        im_nb = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
        
        self.masks_parameters() # Get parameters from GUI
        
        laplacian_pyramid_list = LP_morpho.create_Laplacian_Pyramid(im_nb,self.pyramid_levels)
        laplacian_merged_image_grayscale = LP_morpho.merge_pyramid(laplacian_pyramid_list)
        
        _, mask_threshold = cv2.threshold(laplacian_merged_image_grayscale, self.blur_threshold, 255, cv2.THRESH_BINARY)
        mask_threshold = mask_threshold.astype(np.uint8)
        
        if self.dilate_kernel_size_1 != 0:
            mask_threshold = LP_morpho.dilate_white_zones(mask_threshold, self.dilate_kernel_size_1)
        if self.erode_kernel_size_2 != 0:
            mask_threshold = LP_morpho.erode_white_zones(mask_threshold, self.erode_kernel_size_2)
        if self.dilate_kernel_size_3 != 0:
            mask_threshold = LP_morpho.dilate_white_zones(mask_threshold, self.dilate_kernel_size_3)
            
        mask_threshold = cv2.cvtColor(mask_threshold, cv2.COLOR_GRAY2BGR)
        
        return img_base, im_nb, mask_threshold, laplacian_merged_image_grayscale
    
    
    def layering(self):
        if self.comboBox_BrowseFiles.currentText() != "":
            self.img_base, self.im_nb, self.mask_threshold, self.laplacian_merged_image_grayscale = self.mask_create(self.user_selected_folder, self.comboBox_BrowseFiles.currentText())
        
            # Overlay images
            self.opacity_change_left()
            self.opacity_change_right()
            
        
    def masks_parameters(self):
        self.pyramid_levels = self.spinBox_LaplacianPy.value()
        self.blur_threshold = self.spinBox_BlurThreshold.value()
        self.dilate_kernel_size_1 = self.spinBox_dilate1.value()
        self.erode_kernel_size_2 = self.spinBox_erode2.value()
        self.dilate_kernel_size_3 = self.spinBox_dilate3.value()
        self.Opacity = self.spinBox_Opacity.value()
        self.Opacity2 = self.spinBox_Opacity2.value()
        self.Gamma2 = self.spinBox_Gamma2.value()
        
    def apriltags_parameters(self):
        if self.checkBox_refineEdges.checkState() == Qt.CheckState.Checked:
            self.refineEdges = 1
        else:
            self.refineEdges = 0
        self.decodeSharpening = self.spinBox_decodeSharpening.value()
        self.quadDecimate = self.spinBox_quadDecimate.value()
        self.quadSigma = self.spinBox_quadSigma.value()
        self.criticalAngle = self.spinBox_quadThreshold_criticalAngle.value()
        self.maxLineFitMSE = self.spinBox_maxLineFitMSE.value()
        self.maxNumMaxima = self.spinBox_maxNumMaxima.value()
        self.minClusterPixels = self.spinBox_minClusterPixels.value()
        self.minWhiteBlackDiff = self.spinBox_minWhiteBlackDiff.value()
        self.CropBoxFactor = self.spinBox_CropBoxFactor.value()
        
    def reset_l1(self):
        self.spinBox_LaplacianPy.setValue(self.LaplacianPy_defaultValue)
        self.spinBox_BlurThreshold.setValue(self.BlurThreshold_defaultValue)
        self.spinBox_dilate1.setValue(self.dilate_kernel_size_1_defaultValue)
        self.spinBox_erode2.setValue(self.erode_kernel_size_2_defaultValue)
        self.spinBox_dilate3.setValue(self.dilate_kernel_size_3_defaultValue)
        self.spinBox_Opacity.setValue(self.Opacity_defaultValue)
        self.spinBox_Opacity2.setValue(self.Opacity2_defaultValue)
        self.spinBox_Gamma2.setValue(self.Gamma2_defaultValue)
        self.on_parameters_change()
        
    def reset_l2(self):
        self.checkBox_refineEdges.setCheckState(Qt.CheckState.Checked)
        self.spinBox_decodeSharpening.setValue(0.25)
        self.spinBox_quadDecimate.setValue(2)
        self.spinBox_quadSigma.setValue(0.0)
        self.spinBox_quadThreshold_criticalAngle.setValue(0.7853981633974483)
        self.spinBox_maxLineFitMSE.setValue(10.0)
        self.spinBox_maxNumMaxima.setValue(10)
        self.spinBox_minClusterPixels.setValue(300)
        self.spinBox_minWhiteBlackDiff.setValue(5)
        self.spinBox_CropBoxFactor.setValue(0.1)

        
    def on_parameters_change(self):
        self.label_bilan_dilate.setText(str(self.spinBox_dilate1.value() - self.spinBox_erode2.value() + self.spinBox_dilate3.value()))
        if self.comboBox_BrowseFiles.currentText() != "":
            self.layering()
    
    
    
    def convert_OpenCV_to_QImage(self, im): # convert OpenCV image to QImage
        height, width, channel = im.shape
        bytes_per_line = 3 * width
        qimage = QImage(im.data, width, height, bytes_per_line, QImage.Format_BGR888) # If RGB
        return QPixmap.fromImage(qimage) # Convert QImage to QPixmap
    
    def opacity_change_left(self):
        self.masks_parameters()
        beta = self.Opacity / 100 # Weight for the mask
        alpha = 1 - beta # Weight for original image
        gamma = 0 # ? Doesn't change a thing
        
        im = cv2.addWeighted(self.img_base, alpha, self.mask_threshold, beta, gamma)
        # Call for image of the targets if there are any and layer it
        if self.tags != False:
            imTags = ap.drawTags(im, self.tags, self.refined_tags, thickness=10)
            #im = cv2.addWeighted(im, .5, imTags, .5, gamma)
            mask = np.any(imTags != [0, 0, 0], axis=-1)
            self.tags_all_mask.append(mask)
            im[mask] = imTags[mask]
        pixmap = self.convert_OpenCV_to_QImage(im)
        self.label_OriginalPicture.setPixmap(QPixmap(pixmap))
        
    def opacity_change_right(self):
        self.masks_parameters()
        beta2 = self.Opacity2 / 100 # Weight for the mask
        alpha2 = 1 - beta2 + self.Gamma2 / 100  # Weight for original image, artificially lighten by gamma
        gamma2 = 0 # Doesn't change a thing
        
        self.im_right = cv2.addWeighted(self.img_base, alpha2, self.mask_threshold, beta2, gamma2)
        # Call for image of the targets if there are any and layer it
        if self.tags != False:
            imTags = ap.drawTags(self.im_right, self.tags, self.refined_tags, thickness=1)
            mask = np.any(imTags != [0, 0, 0], axis=-1)
            self.im_right[mask] = imTags[mask]
            
        # Draw Canny lines by overlaying canny_img
        if self.tags != False:
            mask = np.where(self.canny_img > 0)
            canny_img = cv2.cvtColor(self.canny_img,cv2.COLOR_GRAY2BGR)
            self.im_right[mask] = canny_img[mask]
            
        # Export image cv2.imread(join(self.user_selected_folder, jpg_file))
        writepath = Path(join(self.user_selected_folder,"test_current_targets.png"))
        cv2.imwrite(writepath,self.im_right)
        
        # Export basic Laplacian Pyramid image with desired level for experimental purpose
        writepath_LP = Path(join(self.user_selected_folder,"test_current_LP.png"))
        cv2.imwrite(writepath_LP,self.laplacian_merged_image_grayscale)
        
        self.move_zoom() # Crop and focussing another center
        
    def move_zoom(self):
        # Crop and focussing another center ? 
        # Reminder: self.label_OriginalPicture_Scaled.setGeometry(QRect(1440+15, 40, 1900-1440-15, 980))
        crop_size_x = self.label_OriginalPicture_Scaled.size().width()
        crop_size_y = self.label_OriginalPicture_Scaled.size().height()
        slider_x = self.hzSlider.value()
        slider_y = self.verticalSlider.value() # [0; 99]
        """print("")
        print(slider_x)
        print(slider_y)"""
        # Center of the cropped image
        self.center_zoom_x = int(((self.im_right.shape[1] - crop_size_x) / 99) * slider_x + crop_size_x / 2)
        self.center_zoom_y = int(((crop_size_y - self.im_right.shape[0]) / 99) * slider_y + self.im_right.shape[0] - crop_size_y / 2)
        gauche = max(0, self.center_zoom_x - int(crop_size_x / 2))
        droite = min(self.im_right.shape[1], self.center_zoom_x + int(crop_size_x / 2))
        haut = max(0, self.center_zoom_y - int(crop_size_y / 2))
        bas = min(self.im_right.shape[0], self.center_zoom_y + int(crop_size_y / 2))
        im_crop = self.im_right[haut:bas, gauche:droite, :]
        im_crop = np.ascontiguousarray(im_crop)
        # Scale bar in px
        pos_scale_from_angles = 25
        cv2.line(im_crop,(pos_scale_from_angles, int(crop_size_y - pos_scale_from_angles)),
                 (pos_scale_from_angles + 50, int(crop_size_y - pos_scale_from_angles)),(40,40,255),1)
        cv2.line(im_crop,(pos_scale_from_angles + 50, int(crop_size_y - pos_scale_from_angles - 5)),
                 (pos_scale_from_angles + 150, int(crop_size_y - pos_scale_from_angles - 5)),(40,40,255),1)
        cv2.putText(im_crop, str(50), (pos_scale_from_angles, crop_size_y - pos_scale_from_angles - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,20,255), 1, cv2.LINE_AA)
        cv2.putText(im_crop, str(100), (pos_scale_from_angles + 50, crop_size_y - pos_scale_from_angles - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,20,255), 1, cv2.LINE_AA)
        
        #print(im_crop.shape)
        pixmap = self.convert_OpenCV_to_QImage(im_crop)
        self.label_OriginalPicture_Scaled.setPixmap(QPixmap(pixmap))
        
        # Print tag currently on the show
        if self.tags != False:
            target_tag = None
            tag_searched = self.label_AprilTag_current.text()
            for tag in self.refined_tags:
                if str(tag[0]) == tag_searched:
                    target_tag = tag
                    break
            if target_tag:
                tag_id, rtag_x, rtag_y = target_tag
                print("AprilTag "+str(tag_id)+": refined coords: "+str(rtag_x)+", "+str(rtag_y)+
                      " on "+self.comboBox_BrowseFiles.currentText())
                self.label_Info.setText("AprilTag "+str(tag_id)+": refined coords: "+str(rtag_x)+", "+str(rtag_y)+
                                        " on "+self.comboBox_BrowseFiles.currentText())
                self.label_Info.setStyleSheet("""color: #000000;""")
                self.label_Info.raise_()
            else:
                print("/!\\ AprilTag "+tag_searched+" wasn't refined on "+self.comboBox_BrowseFiles.currentText()+" /!\\")
                self.label_Info.setText("/!\\ AprilTag "+tag_searched+" wasn't refined on "+self.comboBox_BrowseFiles.currentText()+" /!\\")
                self.label_Info.setStyleSheet("""color: #000000;""")
                self.label_Info.raise_()
        
    
    def apriltag_previous(self):
        if self.tags != False:
            if self.label_AprilTag_current.text() == "None (center)":
                self.current_tag_index = len(self.tags) - 1
            else:
                if self.current_tag_index <= 0:
                    self.current_tag_index = len(self.tags) - 1
                else:
                    self.current_tag_index -= 1
            self.label_AprilTag_current.setText(str(self.tags[self.current_tag_index].getId()))
            #self.label_AprilTag_current.setText(str(self.tags[self.current_tag_index].tag_id))
            self.calc_slider_pos()
    
    def apriltag_next(self):
        if self.tags != False:
            if self.label_AprilTag_current.text() == "None (center)":
                self.current_tag_index = 0
            else:
                if self.current_tag_index >= len(self.tags) - 1:
                    self.current_tag_index = 0
                else:
                    self.current_tag_index += 1
            self.label_AprilTag_current.setText(str(self.tags[self.current_tag_index].getId()))
            #self.label_AprilTag_current.setText(str(self.tags[self.current_tag_index].tag_id))
            self.calc_slider_pos()
            
    def calc_slider_pos(self):
        crop_size_x = self.label_OriginalPicture_Scaled.size().width()
        crop_size_y = self.label_OriginalPicture_Scaled.size().height()
        # x
        self.hzSlider.setValue(
            (self.tags[self.current_tag_index].getCenter().x - crop_size_x/2) * 99 /
            #(self.tags[self.current_tag_index].center[0] - crop_size_x/2) * 99 /
            (self.im_right.shape[1] - crop_size_x)
            )
        # y
        self.verticalSlider.setValue(
            (self.tags[self.current_tag_index].getCenter().y + crop_size_y / 2 - self.im_right.shape[0]) * 99 /
            #(self.tags[self.current_tag_index].center[1] + crop_size_y / 2 - self.im_right.shape[0]) * 99 /
            (crop_size_y - self.im_right.shape[0])
            )

        
    def detect_apriltags(self):
        #print(self.tags)
        self.apriltags_parameters()
        print(self.refineEdges)
        print(f"decodeSharpening {self.decodeSharpening}, quadDecimate {self.quadDecimate}, quadSigma {self.quadSigma}")
        print(f"criticalAngle {self.criticalAngle}, qmaxLineFitMSE {self.maxLineFitMSE}, maxNumMaxima {self.maxNumMaxima}, minClusterPixels {self.minClusterPixels}, minWhiteBlackDiff {self.minWhiteBlackDiff}")
        self.tags, self.refined_tags, self.canny_img, self.cropzones_img = ap.detectTagsFilter(self.im_nb, self.refineEdges, self.decodeSharpening, self.quadDecimate, self.quadSigma,
                                        self.criticalAngle, self.maxLineFitMSE, self.maxNumMaxima,
                                        self.minClusterPixels, self.minWhiteBlackDiff,
                                        self.CropBoxFactor)
        """def detectTags(img_nb, refineEdges=True, decodeSharpening=0.25, quadDecimate=2, quadSigma=0.0,
               criticalAngle=10, maxLineFitMSE=10.0, maxNumMaxima=10, minClusterPixels=5, minWhiteBlackDiff=5)"""
        """if self.tags:
            for tag in self.tags:
                print(tag.getId())"""
        
        self.opacity_change_left()
        self.opacity_change_right()
        
    # Compute on entire folder either mask or tags
    def runMasks(self):
        self.label_Info.setText("WAIT... Creating masks")
        self.label_Info.setStyleSheet("""color: #FF0000;""")
        self.label_Info.raise_()
        QApplication.processEvents()
        
        exec_date_time = time.strftime("%Y%m%d_%H%M", time.localtime())
        start_time = time.time()
        runtime1 = False
        folder = Path(self.user_selected_folder)
        for jpg_file in folder.glob("*.[jJ][pP][gG]"):
            _, _, mask_threshold, laplacian_merged_image_grayscale = self.mask_create(self.user_selected_folder, jpg_file)
            
            if runtime1 == False:
                runtime1 = round(time.time() - start_time, 2)
                
            mask_path_threshold = Path(self.user_selected_folder) / f"{exec_date_time}-{self.pyramid_levels}_{self.blur_threshold}_{self.dilate_kernel_size_1},{self.erode_kernel_size_2},{self.dilate_kernel_size_3}_dilate_erode_dilate_{runtime1}s" / f"{jpg_file.stem}.mask.png"
            mask_path_threshold.parent.mkdir(parents=True, exist_ok=True)
            print(mask_path_threshold)
            
            self.label_Info.setText("WAIT... Creating masks\n[...]\n"+str(mask_path_threshold))
            self.label_Info.setStyleSheet("""color: #FF0000;""")
            self.label_Info.raise_()
            QApplication.processEvents()
        
            cv2.imwrite(mask_path_threshold,mask_threshold)
            
        exec_total_time = round(time.time() - start_time, 2)
        print("Masks generated in "+str(exec_total_time)+" s")
        self.label_Info.setText("Masks generated in "+str(exec_total_time)+" s in subdir of "+self.user_selected_folder)
        self.label_Info.setStyleSheet("""color: #44CC44;""")
        self.label_Info.raise_()
    
    def runTags(self):
        self.label_Info.setText("WAIT... Creating Tags list")
        self.label_Info.setStyleSheet("""color: #FF0000;""")
        self.label_Info.raise_()
        QApplication.processEvents()
        
        exec_date_time = time.strftime("%Y%m%d_%H%M", time.localtime())
        start_time = time.time()
        runtime1 = False
        folder = Path(self.user_selected_folder)
        for jpg_file in folder.glob("*.[jJ][pP][gG]"):
            self.apriltags_parameters()
            #print(jpg_file)
            #print(join(self.user_selected_folder, jpg_file))
            #print(os.path.exists(join(self.user_selected_folder, jpg_file)))
            #print(self.refineEdges)
            #print(f"decodeSharpening {self.decodeSharpening}, quadDecimate {self.quadDecimate}, quadSigma {self.quadSigma}")
            #print(f"criticalAngle {self.criticalAngle}, qmaxLineFitMSE {self.maxLineFitMSE}, maxNumMaxima {self.maxNumMaxima}, minClusterPixels {self.minClusterPixels}, minWhiteBlackDiff {self.minWhiteBlackDiff}")
            im = cv2.imread(join(self.user_selected_folder, jpg_file))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            _, refined_tags, _, _ = ap.detectTagsFilter(im, self.refineEdges, self.decodeSharpening, self.quadDecimate, self.quadSigma,
                                        self.criticalAngle, self.maxLineFitMSE, self.maxNumMaxima,
                                        self.minClusterPixels, self.minWhiteBlackDiff,
                                        self.CropBoxFactor)
            
            if runtime1 == False:
                runtime1 = round(time.time() - start_time, 2)
                
            if self.refineEdges == True:
                refineEdges = 1
            else:
                refineEdges = 0
            
            apriltags_relative_path = Path(self.user_selected_folder) / f"AprilTags_36h11_on_images" / f"{exec_date_time}-36h11_relative-{refineEdges},{self.decodeSharpening},{round(self.quadDecimate,1)},{self.quadSigma}_{round(self.criticalAngle,3)},{self.maxLineFitMSE},{self.maxNumMaxima},{self.minClusterPixels},{self.minWhiteBlackDiff}.txt"
            apriltags_absolute_path = Path(self.user_selected_folder) / f"AprilTags_36h11_on_images" / f"{exec_date_time}-36h11_absolute-{refineEdges},{self.decodeSharpening},{round(self.quadDecimate,1)},{self.quadSigma}_{round(self.criticalAngle,3)},{self.maxLineFitMSE},{self.maxNumMaxima},{self.minClusterPixels},{self.minWhiteBlackDiff}.txt"

            #apriltags_absolute_path_monge = Path(self.user_selected_folder) / f"AprilTags_36h11_on_images" / f"{exec_date_time}-36h11_absolute-{refineEdges},{self.decodeSharpening},{self.quadDecimate},{self.quadSigma}_{round(self.criticalAngle,3)},{self.maxLineFitMSE},{self.maxNumMaxima},{self.minClusterPixels},self.minWhiteBlackDiff_Monge.txt"
            
            apriltags_relative_path.parent.mkdir(parents=True, exist_ok=True)
            apriltags_absolute_path.parent.mkdir(parents=True, exist_ok=True)
            #apriltags_absolute_path_monge.parent.mkdir(parents=True, exist_ok=True)
            
            if refined_tags != False:
                for tag in refined_tags:
                    with open(apriltags_relative_path, "a") as f:
                        f.write(str(jpg_file.name)+", "+str(tag[0])+", "+str(tag[1])+", "+str(tag[2])+"\n")
                    with open(apriltags_absolute_path, "a") as f:
                        f.write(str(jpg_file)+", "+str(tag[0])+", "+str(tag[1])+", "+str(tag[2])+"\n")
            """if monge_tags != False:
                for tag in monge_tags:
                    with open(apriltags_absolute_path_monge, "a") as f:
                        f.write(str(jpg_file)+", "+str(tag[0])+", "+str(tag[1])+", "+str(tag[2])+"\n")"""
            
            
            """self.label_Info.setText(self.label_Info.text()+"\n"+str(mask_path_threshold))
            self.palette.setColor(QPalette.WindowText, QColor(255,0,0))
            self.label_Info.setPalette(self.palette)
            self.label_Info.raise_()
            QApplication.processEvents()"""
        
            #cv2.imwrite(mask_path_threshold,mask_threshold)
            
        exec_total_time = round(time.time() - start_time, 2)
        print("List of 36h11 AprilTags generated in "+str(exec_total_time)+" s in subdir of "+self.user_selected_folder)
        self.label_Info.setText("Tags generated in "+str(exec_total_time)+" s in subdir of "+self.user_selected_folder)
        self.label_Info.setStyleSheet("""color: #44CC44;""")
        self.label_Info.raise_()
        
    
    def runSobel(self):
        self.label_Info.setText("WAIT... Creating Sobel images")
        self.label_Info.setStyleSheet("""color: #FF0000;""")
        self.label_Info.raise_()
        QApplication.processEvents()
        
        exec_date_time = time.strftime("%Y%m%d_%H%M", time.localtime())
        start_time = time.time()
        runtime1 = False
        folder = Path(self.user_selected_folder)
        for jpg_file in folder.glob("*.[jJ][pP][gG]"):
            image_path = join(folder, jpg_file)
            img_base = cv2.imread(image_path)
        
            #im_nb = img_base.copy()
            im_nb = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
        
            #self.masks_parameters() # Get parameters from GUI (not needed right now, TODO if gaussian blur applied with kernel changeable)
            sobel = on_crop_compute_sobel.sobel(im_nb)
            #_, _, mask_threshold, laplacian_merged_image_grayscale = self.mask_create(self.user_selected_folder, jpg_file)
            
            # Trying separating and merging channels
            """im_blue, im_green, im_red = cv2.split(img_base)
            sobel_blue = on_crop_compute_sobel.sobel(im_blue)
            sobel_green = on_crop_compute_sobel.sobel(im_green)
            sobel_red = on_crop_compute_sobel.sobel(im_red)
            sobel_merge = cv2.merge((sobel_blue, sobel_green, sobel_red))"""
            
            if runtime1 == False:
                runtime1 = round(time.time() - start_time, 2)
                
            sobel_path = Path(self.user_selected_folder) / f"{exec_date_time}-sobel" / f"{jpg_file.stem}.sobel.jpg"
            #sobel_merge_path = Path(self.user_selected_folder) / f"{exec_date_time}-sobel" / f"{jpg_file.stem}.sobel_merge.jpg"
            sobel_path.parent.mkdir(parents=True, exist_ok=True)
            print(sobel_path)
            if self.label_Info.text().count("\n") > 55:
                self.label_Info.setText("...\n"+str(sobel_path))
            else:
                self.label_Info.setText(self.label_Info.text()+"\n"+str(sobel_path))
            self.label_Info.setStyleSheet("""color: #FF0000;""")
            self.label_Info.raise_()
            QApplication.processEvents()
        
            cv2.imwrite(sobel_path,sobel)
            #cv2.imwrite(sobel_merge_path,sobel_merge)
            
        exec_total_time = round(time.time() - start_time, 2)
        print("Sobel images generated in "+str(exec_total_time)+" s in subdir of "+self.user_selected_folder)
        self.label_Info.setText("Sobel images generated in "+str(exec_total_time)+" s in subdir of "+self.user_selected_folder)
        self.label_Info.setStyleSheet("""color: #44CC44;""")
        self.label_Info.raise_()
        
    
    def exifRotTest(self):
        folder = Path(self.user_selected_folder)
        if folder != None:
            exif = exif_changer.Clean_some_exif()
            exif.list_files(folder)
            if exif.ifRotExif() != False:
                self.pushButton_ExifR.setStyleSheet("""color: red; text-decoration: none;""")
            
                self.label_Info.setText("EXIF Orientation tags does exist, see console to know the pictures which have some. "+
                                    "Clean it before generating masks, tags and so on.")
                self.label_Info.setStyleSheet("""color: #FF0000;""")
                self.label_Info.raise_()
            else:
                self.pushButton_ExifR.setStyleSheet("""color: #969696; text-decoration: line-through;""")
                
                self.label_Info.setText("")
                self.label_Info.raise_()
    
    
    def exifCleanRot(self):
        folder = Path(self.user_selected_folder)
        if folder != None:
            self.label_Info.setText("WAIT: Currently cleaning EXIF orientations on current folder...")
            self.label_Info.setStyleSheet("""color: #FF0000;""")
            self.label_Info.raise_()
            QApplication.processEvents()
            
            exif = exif_changer.Clean_some_exif()
            exif.list_files(folder)
            exif.clean_orientation_tag()
            
            self.load_image_files()
            self.update_comboBox()
        
            self.label_Info.setText("EXIF orientations cleaned.")
            self.label_Info.setStyleSheet("""color: #00FF00;""")
            self.label_Info.raise_()
            
        


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    app.exec()
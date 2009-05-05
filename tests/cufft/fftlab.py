#! /usr/bin/env python
#
# fftlab.py --- Demonstrates 2d ffts and convolutions
#

import sys, os, random

from qt import *
from matplotlib.numerix import arange, sin, pi
QApplication.setColorSpec(QApplication.NormalColor)
app = QApplication(sys.argv)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy import *
from scipy import signal
from pylab import fftshift
import numpy as np

# Required for PyQt
TRUE  = 1
FALSE = 0

PROGNAME = "fftlab"
PROG_VERSION = "0.1"

class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()
        
        FigureCanvas.__init__(self, self.fig)
        self.reparent(parent, QPoint(0, 0))

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def sizeHint(self):
        w, h = self.get_width_height()
        return QSize(w, h)

    def minimumSizeHint(self):
        return QSize(10, 10)

class ImageCanvas(MplCanvas):
    """Simple canvas for matshow'ing a 2d image array"""
    def __init__(self, numpy_array, parent=None, width=5, height=4, dpi=100):
        assert numpy_array.ndim == 2
        self.data = numpy_array
        super(ImageCanvas,self).__init__(parent, width, height, dpi)

    def compute_initial_figure(self):
        self.axes.matshow(self.data)

class FFTLab(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self, None,
                             "FFTLab Main Window",
                             Qt.WType_TopLevel | Qt.WDestructiveClose)

        self.file_menu = QPopupMenu(self)
        self.file_menu.insertItem('&Quit', self.file_quit, Qt.CTRL + Qt.Key_Q)
        self.menuBar().insertItem('&File', self.file_menu)

        self.help_menu = QPopupMenu(self)
        self.menuBar().insertSeparator()
        self.menuBar().insertItem('&Help', self.help_menu)

        self.help_menu.insertItem('&About', self.about)

        self.main_widget = QWidget(self, "Main widget")

        lena_orig = ImageCanvas(lena()/255.,self.main_widget)
        lena_fft = ImageCanvas(signal.fft2(lena()/255.).real,self.main_widget)
        lena_power = ImageCanvas(fftshift(log(abs(signal.fftn(lena()/255.)))),self.main_widget)
        lena_convolved = ImageCanvas(signal.fftconvolve(lena()/255., np.ones((11,11))), self.main_widget)

        g = QGridLayout(self.main_widget)
        g.addWidget(lena_orig,0,0)
        g.addWidget(lena_convolved,0,1)
        g.addWidget(lena_fft,1,0)
        g.addWidget(lena_power,1,1)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().message("%s - v%s" % (PROGNAME, PROG_VERSION) , 2000)

    def file_quit(self):
        qApp.exit(0)

    def about(self):
        QMessageBox.about(self, "About %s" % PROGNAME,
u"""%(prog)s version %(version)s

This program visualizes 2d ffts and convolutions 
""" % {"prog": PROGNAME, "version": PROG_VERSION})

def main():
    aw = FFTLab()
    aw.setCaption("%s" % PROGNAME)
    qApp.setMainWidget(aw)
    aw.show()
    sys.exit(qApp.exec_loop())

if __name__ == "__main__": main()

#!/bin/env python
# coding:utf-8: © Arno Pähler,2007-08
# Qt4/PyQt4 version
from ctypes import *
import time

from PyQt4 import QtCore,QtGui,QtOpenGL
from PyQt4.QtCore import Qt
from ogl.gl import *

from cuda.cuda_defs import *
from cuda.cuda_api import *
from cuda.cuda_utils import *

lib = CDLL("./libkernelGL.so")

kernel1 = lib.__device_stub_kernel1
kernel1.restype = None
kernel1.argtypes = [ c_void_p,c_uint,c_uint,c_float ]

kernel2 = lib.__device_stub_kernel2
kernel2.restype = None
kernel2.argtypes = [ c_void_p,c_uint,c_uint,c_float ]

Sf4 = 4*sizeof(c_float)

kernel = kernel2

class CudaGLWidget(QtOpenGL.QGLWidget):
    def __init__(self,parent = None,name = None):
        QtOpenGL.QGLWidget.__init__(self,parent,name)
        self.setWindowTitle("Cuda GL Demo")

        self.device = cuda_CUDA()


        # self.initializeGL gets called automatically
        # and implicitly creates the OpenGL context
        self.reset()
        self.setGeometry(QtCore.QRect(0,0.,self.width,self.height))

        self.t0 = time.time()
        self.frames = 0
        self.startTimer(0)

    def reset(self):
        self.mouse_buttons = -1
        self.last_x = 0.
        self.last_y = 0.
        self.rot_x = 0.
        self.rot_y = 0.
        self.trn_z = -3.
        self.anim = 0.
        self.scale = 1.
        self.play = True

        self.width = 512
        self.height = 512

        self.mesh_w = 256
        self.mesh_h = 256
        self.m_size = self.mesh_w*self.mesh_h

    def runCuda(self):
        vptr = c_void_p()
        cudaGLMapBufferObject(byref(vptr),self.vbo)

        block = dim3(16,16,1)
        grid = dim3(self.mesh_w/block.x,self.mesh_h/block.y,1)
        cudaConfigureCall(grid,block,0,0)
        kernel(vptr,self.mesh_w,self.mesh_h,self.anim)

        cudaGLUnmapBufferObject(self.vbo)

    def initializeGL(self):
        glDisable(GL_DEPTH_TEST)
        self.createVBO()

    def paintGL(self):
        self.runCuda()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.,0.,self.trn_z)
        glRotatef(self.rot_x,1.,0.,0.)
        glRotatef(self.rot_y,0.,1.,0.)
        s = self.scale
        glScalef(s,s,s)

        glBindBuffer(GL_ARRAY_BUFFER,self.vbo)
        glVertexPointer(4,GL_FLOAT,0,0)

        glEnableClientState(GL_VERTEX_ARRAY)
        glColor3f(1.,1.,.75)
        glDrawArrays(GL_POINTS,0,self.m_size)

        glDisableClientState(GL_VERTEX_ARRAY)

        if self.play:
            self.anim += 0.01

    def resizeGL(self,width,height):
        self.width = width
        self.height = height
        w = float(width)/float(height)
        h = 1.

        glViewport(0,0,width,height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(-w,w,-h,h,2.,10.)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.,0.,-3.)

    def createVBO(self):
        self.vbo = vbo = GLuint()
        glGenBuffers(1,byref(vbo))
        glBindBuffer(GL_ARRAY_BUFFER,vbo)

        size = self.m_size*Sf4
        glBufferData(GL_ARRAY_BUFFER,size,0,GL_DYNAMIC_DRAW)
        b_ptr = glMapBuffer(GL_ARRAY_BUFFER,GL_WRITE_ONLY)
        glUnmapBuffer(GL_ARRAY_BUFFER)
        glBindBuffer(GL_ARRAY_BUFFER,0)
        cudaGLRegisterBufferObject(vbo)

    def deleteVBO(self):
        vbo = self.vbo
        glBindBuffer(1,vbo)
        glDeleteBuffers(1,vbo)

        cudaGLUnregisterBufferObject(vbo)
        self.vbo = None

    def keyPressEvent(self,ev):
        key = ev.key()
        if key == Qt.Key_Escape \
        or key == Qt.Key_Q:
            self.deleteVBO()
            cudaThreadExit()
            exit()
        elif key == Qt.Key_R:
            self.reset()
        elif key == Qt.Key_S:
            self.play = not self.play
        elif key == Qt.Key_P:
            image = self.grabFrameBuffer()
            image.save("cuda_GLimg.png","PNG")

    def mousePressEvent(self,ev):
        button = ev.button()
        if button == Qt.LeftButton:
            self.MouseButton = 0
        elif button == Qt.RightButton:
            self.MouseButton = 2
        else:
            self.MouseButton = 1
        self.last_x = ev.x()
        self.last_y = ev.y()

    def mouseReleaseEvent(self,ev):
        self.MouseButton = -1

    def mouseMoveEvent(self,ev):
        x,y = ev.x(),ev.y()
        dx = x-self.last_x
        dy = y-self.last_y
        mouse = self.MouseButton
        if mouse == 0:
            self.rot_x += .2*dy
            self.rot_y += .2*dx
        elif mouse == 2:
            self.trn_z += .01*dy
        self.last_x = x
        self.last_y = y

    def wheelEvent(self,ev):
        delta = ev.delta() # usually +/- 120
        scale = self.scale
        if delta != 0:
            scale += 6./float(delta)
        self.scale = min(2.,max(.2,scale))

    def timerEvent(self,ev):
        t = time.time()
        self.frames +=  1
        if (t-self.t0) >=  5.0:
            seconds = t-self.t0
            fps = self.frames/seconds
            print "%d frames in %3.1f seconds = %6.1f fps" % (
                self.frames,seconds,fps);
            self.frames = 0
            self.t0 = t
        self.updateGL() # needed

##############################################################################
if __name__ == '__main__':
    from sys import argv
    app = QtGui.QApplication(argv)
    widget = CudaGLWidget()
    widget.show()
    app.exec_()

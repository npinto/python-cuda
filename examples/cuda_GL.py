#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
# GLUT version
from ctypes import *

from ogl.gl import *
from OpenGL.GLUT import *

from cuda.cuda_defs import *
from cuda.cuda_api import *

lib = CDLL("./libkernelGL.so")

kernel1 = lib.__device_stub_kernel1
kernel1.restype = None
kernel1.argtypes = [ c_void_p, c_uint, c_uint, c_float ]

kernel2 = lib.__device_stub_kernel2
kernel2.restype = None
kernel2.argtypes = [ c_void_p, c_uint, c_uint, c_float ]

window_width = 512
window_height = 512

mesh_width = 256
mesh_height = 256

anim = 0.0
mouse_buttons = 0
rotate_x,rotate_y,translate_z = 0.,0.,-3.0
global mouse_old_x,mouse_old_y

vbo = GLuint()

kernel = kernel1

def main(argc,argv):
    global vbo

    glutInit(argc,argv)
    glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE)
    glutInitWindowSize(window_width,window_height)
    glutCreateWindow("Cuda GL Demo")

    initGL()

    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)

    vbo = createVBO()
    runCuda(vbo)

    glutMainLoop()

def runCuda(vbo):
    vptr = c_void_p()
    status = cudaGLMapBufferObject(byref(vptr),vbo)

    block = dim3(8,8,1)
    grid = dim3(mesh_width/block.x,mesh_height/block.y,1)
    status = cudaConfigureCall(grid,block,0,0)
    kernel(vptr,mesh_width,mesh_height,anim)

    status = cudaGLUnmapBufferObject(vbo)
    if status != 0:
        exit()

def initGL():
    glClearColor(0.0,0.0,0.0,1.0)
    glDisable(GL_DEPTH_TEST)

    glViewport(0,0,window_width,window_height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    ratio = float(window_width)/float(window_height)
    glFrustum(-1.,1.,-1.,1.,2.,10.)

    return True

def createVBO():
    global vbo
    glGenBuffers(1,byref(vbo))
    glBindBuffer(GL_ARRAY_BUFFER,vbo)

    size = mesh_width*mesh_height*4*sizeof(c_float)
    glBufferData(GL_ARRAY_BUFFER,size,0,GL_DYNAMIC_DRAW)

    glBindBuffer(GL_ARRAY_BUFFER,0)

    status = cudaGLRegisterBufferObject(vbo)
    return vbo

def deleteVBO():
    global vbo
    glBindBuffer(1,vbo)
    glDeleteBuffers(1,vbo)

    status = cudaGLUnregisterBufferObject(vbo)
    vbo = 0

def display():
    global anim,vbo
    runCuda(vbo)

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0.0,0.0,translate_z)
    glRotatef(rotate_x,1.0,0.0,0.0)
    glRotatef(rotate_y,0.0,1.0,0.0)

    glBindBuffer(GL_ARRAY_BUFFER,vbo)
    glVertexPointer(4,GL_FLOAT,0,0)

    glEnableClientState(GL_VERTEX_ARRAY)
    glColor3f(1.0,0.0,0.0)
    glDrawArrays(GL_POINTS,0,mesh_width*mesh_height)
    glDisableClientState(GL_VERTEX_ARRAY)

    glutSwapBuffers()
    glutPostRedisplay()

    anim += 0.01

def keyboard(key,x,y):
    if key == chr(27):
        deleteVBO()
        exit()

def mouse(button,state,x,y):
    global mouse_buttons
    global mouse_old_x,mouse_old_y
    if state == GLUT_DOWN:
        mouse_buttons |= 1<<button
    elif state == GLUT_UP:
        mouse_buttons = 0

    mouse_old_x = x
    mouse_old_y = y
    glutPostRedisplay()

def motion(x,y):
    global mouse_buttons
    global mouse_old_x,mouse_old_y
    global rotate_x,rotate_y,translate_z
    dx = x-mouse_old_x
    dy = y-mouse_old_y

    if mouse_buttons & 1:
        rotate_x += dy*0.2
        rotate_y += dx*0.2
    elif mouse_buttons & 4:
        translate_z += dy*0.01

    mouse_old_x = x
    mouse_old_y = y

if __name__ == "__main__":
    from sys import argv
    argc = len(argv)
    cudaSetDevice(0)
    main(argc,argv)

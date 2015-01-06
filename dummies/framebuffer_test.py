import ctypes
import numpy as np
import cv2
import matplotlib.pyplot as plt
from commons.utils import current_time_ms
from commons.opengl import build_program, use_program, VertexArrayObject, VertexBufferObject, Framebuffer

import OpenGL.GL as gl
import OpenGL.GLUT as glut

rectangle_vp = '''
uniform float time;
uniform vec2 scale_to_square;

attribute vec2 position;
attribute vec4 color;

varying vec4 v_color;

void main()
{
    vec2 pos2d = position;
    float scale = (0.75 - sin(time / 500.0) * 0.25) / sqrt(2.0);
    float alpha = time / 500.0;
    float cosA = cos(alpha);
    float sinA = sin(alpha);
    mat2 rotate = mat2(cosA, sinA, -sinA, cosA);

    pos2d = scale * pos2d;
    pos2d = rotate * pos2d;
    pos2d = scale_to_square * pos2d;
    gl_Position = vec4(pos2d, 0.0, 1.0);
    v_color = color;
}
'''

rectangle_fp = '''
varying vec4 v_color;

void main()
{
    gl_FragColor = v_color;
}
'''

rectangle_vp_small = '''
uniform float time;
uniform vec2 scale_to_square;

attribute vec2 position;
attribute vec4 color;

varying vec4 v_color;

void main()
{
    vec2 pos2d = position;
    float scale = (0.25 - sin(time / 500.0) * 0.20) / sqrt(2.0);
    float alpha = time / 500.0;
    float cosA = cos(alpha);
    float sinA = sin(alpha);
    mat2 rotate = mat2(cosA, sinA, -sinA, cosA);

    pos2d = scale * pos2d;
    pos2d = rotate * pos2d;
    pos2d = scale_to_square * pos2d;
    gl_Position = vec4(pos2d, 0.239, 1.0);
    v_color = color;
}
'''

rectangle_fp_small = '''
varying vec4 v_color;

void main()
{
    gl_FragColor = v_color;
}
'''


def get_color(width, height, debug=False):
    data = np.ndarray((height, width, 4), np.uint8)
    gl.glReadPixels(0, 0, width, height, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, data)
    assert data.min() != data.max()
    if debug:
        cv2.imshow("Test", data[::-1, :])
        cv2.waitKey(1)
    return data


def get_depth(width, height, debug=False):
    data = np.ndarray((height, width), np.float32)
    gl.glReadPixels(0, 0, width, height, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, data)
    assert (data.min() - (0.239 + 1.0) / 2.0) < 1e-6
    if debug:
        if height > 1:
            plt.clf()
            plt.imshow(data, origin='lower', vmin=-1.0, vmax=1.0)
            plt.colorbar()
            plt.show()
        else:
            plt.plot(data[0])
            plt.show()
    return data


if __name__ == '__main__':
    lightness = 0.6
    width, height = 768, 512
    width_2, height_2 = width, height
    test_depth = False
    if test_depth:
        height_2 = 1
    position = np.asarray([[-1, -1], [-1, +1], [+1, -1], [+1, +1]], np.float32)
    position_small = np.asarray([[-0.5, -1], [-0.5, +1], [+1.5, -1], [+1.5, +1]], np.float32)
    color = np.asarray([[lightness, 0, 0, 1], [0, lightness, 0, 1], [0, 0, lightness, 1], [lightness, lightness, 0, 1]],
                       np.float32)
    indices = np.asarray([[0, 1, 2], [1, 2, 3]], np.uint8)
    indices_small = np.asarray([[0, 1, 2], [0, 2, 3]], np.uint8)

    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
    glut.glutCreateWindow('Framebuffer test')
    glut.glutReshapeWindow(width, height)

    gl_program = build_program(rectangle_vp, rectangle_fp)
    gl_program_small = build_program(rectangle_vp_small, rectangle_fp_small)

    vao = VertexArrayObject()

    with vao:
        indices_buffer = VertexBufferObject(gl.GL_ELEMENT_ARRAY_BUFFER)
        with indices_buffer:
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        color_buffer = VertexBufferObject()
        with color_buffer:
            gl.glBufferData(gl.GL_ARRAY_BUFFER, color.nbytes, color, gl.GL_STATIC_DRAW)
            loc = gl.glGetAttribLocation(gl_program, 'color')
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, False, color.strides[0], ctypes.c_void_p(0))

        position_buffer = VertexBufferObject()
        with position_buffer:
            gl.glBufferData(gl.GL_ARRAY_BUFFER, position.nbytes, position, gl.GL_STATIC_DRAW)
            loc = gl.glGetAttribLocation(gl_program, 'position')
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, position.strides[0], ctypes.c_void_p(0))

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, indices_buffer.handle)

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    vao_small = VertexArrayObject()

    with vao_small:
        indices_buffer_small = VertexBufferObject(gl.GL_ELEMENT_ARRAY_BUFFER)
        with indices_buffer_small:
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices_small.nbytes, indices_small, gl.GL_STATIC_DRAW)

        color_buffer_small = VertexBufferObject()
        with color_buffer_small:
            gl.glBufferData(gl.GL_ARRAY_BUFFER, color.nbytes, color, gl.GL_STATIC_DRAW)
            loc = gl.glGetAttribLocation(gl_program, 'color')
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, False, color.strides[0], ctypes.c_void_p(0))

        position_buffer_small = VertexBufferObject()
        with position_buffer_small:
            gl.glBufferData(gl.GL_ARRAY_BUFFER, position_small.nbytes, position_small, gl.GL_STATIC_DRAW)
            loc = gl.glGetAttribLocation(gl_program, 'position')
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, position_small.strides[0], ctypes.c_void_p(0))

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, indices_buffer_small.handle)

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    framebuffer = Framebuffer()
    assert gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE

    color_tex = gl.glGenTextures(1)
    # gl.glActiveTexture(gl.GL_TEXTURE1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, color_tex)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width_2, height_2, 0, gl.GL_RGBA,
                    gl.GL_UNSIGNED_BYTE, None)

    depth_tex = gl.glGenTextures(1)
    # gl.glActiveTexture(gl.GL_TEXTURE2)
    gl.glBindTexture(gl.GL_TEXTURE_2D, depth_tex)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, width_2, height_2, 0, gl.GL_DEPTH_COMPONENT,
                    gl.GL_FLOAT, None)
    # gl.glActiveTexture(gl.GL_TEXTURE0)

    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    with framebuffer:
        # gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D,
                                  depth_tex, 0)
        # gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D,
                                  color_tex, 0)
        # gl.glActiveTexture(gl.GL_TEXTURE0)

    start = current_time_ms()

    def keyboard(key, x, y):
        if key == '\033':
            glut.glutLeaveMainLoop()

    def reshape(new_width, new_height):
        global width, height
        width, height = new_width, new_height

    def update(_):
        with use_program(gl_program_small):
            loc = gl.glGetUniformLocation(gl_program_small, 'time')
            gl.glUniform1f(loc, current_time_ms() - start)
            # scale_width, scale_height = min(width_2, height_2) * 1.0 / width_2, min(width_2, height_2) * 1.0 / height_2
            scale_width, scale_height = 1, 1
            loc = gl.glGetUniformLocation(gl_program_small, 'scale_to_square')
            gl.glUniform2f(loc, scale_width, scale_height)
        with use_program(gl_program):
            loc = gl.glGetUniformLocation(gl_program, 'time')
            gl.glUniform1f(loc, current_time_ms() - start)
            scale_width, scale_height = min(width, height) * 1.0 / width, min(width, height) * 1.0 / height
            loc = gl.glGetUniformLocation(gl_program, 'scale_to_square')
            gl.glUniform2f(loc, scale_width, scale_height)
            glut.glutPostRedisplay()
            glut.glutTimerFunc(0, update, 0)

    last_display_time = start

    def display():
        global last_display_time
        current_time = current_time_ms()
        print 1000 / (current_time - last_display_time), 'FPS\r',
        gl.glViewport(0, 0, width, height)
        last_display_time = current_time
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        with use_program(gl_program), vao:
            gl.glDrawElements(gl.GL_TRIANGLES, indices.size, gl.GL_UNSIGNED_BYTE, None)
        with framebuffer, use_program(gl_program_small), vao_small:
            gl.glViewport(0, 0, width_2, height_2)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            gl.glDrawElements(gl.GL_TRIANGLES, indices.size, gl.GL_UNSIGNED_BYTE, None)
            get_color(width_2, height_2, debug=True)
            get_depth(width_2, height_2, debug=False)
        glut.glutSwapBuffers()

    glut.glutKeyboardFunc(keyboard)
    glut.glutReshapeFunc(reshape)
    glut.glutDisplayFunc(display)
    glut.glutTimerFunc(0, update, 0)
    glut.glutMainLoop()
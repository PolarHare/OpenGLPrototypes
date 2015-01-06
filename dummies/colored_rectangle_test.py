import ctypes
import numpy as np
from commons.utils import current_time_ms
from commons.opengl import build_program, use_program, VertexArrayObject, VertexBufferObject

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

if __name__ == '__main__':
    lightness = 0.6
    width, height = 512, 512
    position = np.asarray([[-1, -1], [-1, +1], [+1, -1], [+1, +1]], np.float32)
    color = np.asarray([[lightness, 0, 0, 1], [0, lightness, 0, 1], [0, 0, lightness, 1], [lightness, lightness, 0, 1]], np.float32)
    indices = np.asarray([[0, 1, 2], [1, 2, 3]], np.uint8)

    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
    glut.glutCreateWindow('Colored rectangle')
    glut.glutReshapeWindow(width, height)

    gl_program = build_program(rectangle_vp, rectangle_fp)

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

    start = current_time_ms()

    def keyboard(key, x, y):
        if key == '\033':
            glut.glutLeaveMainLoop()

    def reshape(new_width, new_height):
        global width, height
        width, height = new_width, new_height
        gl.glViewport(0, 0, width, height)

    def update(_):
        with use_program(gl_program):
            loc = gl.glGetUniformLocation(gl_program, 'time')
            gl.glUniform1f(loc, current_time_ms() - start)
            scale_width, scale_height = min(width, height)*1.0/width, min(width, height)*1.0/height
            loc = gl.glGetUniformLocation(gl_program, 'scale_to_square')
            gl.glUniform2f(loc, scale_width, scale_height)
            glut.glutPostRedisplay()
            glut.glutTimerFunc(0, update, 0)

    last_display_time = start

    def display():
        global last_display_time
        current_time = current_time_ms()
        print 1000 / (current_time - last_display_time), 'FPS\r',
        last_display_time = current_time
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        with use_program(gl_program), vao:
            gl.glDrawElements(gl.GL_TRIANGLES, indices.size, gl.GL_UNSIGNED_BYTE, None)
        glut.glutSwapBuffers()

    glut.glutKeyboardFunc(keyboard)
    glut.glutReshapeFunc(reshape)
    glut.glutDisplayFunc(display)
    glut.glutTimerFunc(0, update, 0)
    glut.glutMainLoop()
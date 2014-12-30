import collections
import ctypes
from contextlib import contextmanager

import numpy as np
import PIL.Image

import OpenGL.GL as gl
from OpenGL.GL.shaders import glGetProgramInfoLog

from glut import GlutWindow

REPEAT_TEXTURE = [(gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT),
                  (gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)]

CLAMP_TEXTURE = [(gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER),
                 (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)]

LINEAR_TEXTURE = [(gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR),
                  (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)]


@contextmanager
def use_gl_program(gl_program):
    gl.glUseProgram(gl_program)
    yield
    gl.glUseProgram(0)


def recognize_gl_type(data):
    return {'float32': gl.GL_FLOAT,
            'float64': gl.GL_DOUBLE,
            'uint32': gl.GL_UNSIGNED_INT
    }[data.dtype.name]

def build_gl_program(vertex_code, fragment_code):
    gl_program = gl.glCreateProgram()

    vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(vertex, vertex_code)
    gl.glShaderSource(fragment, fragment_code)
    gl.glCompileShader(vertex)
    gl.glCompileShader(fragment)

    gl.glAttachShader(gl_program, vertex)
    gl.glAttachShader(gl_program, fragment)
    gl.glLinkProgram(gl_program)

    success = gl.glGetProgramiv(gl_program, gl.GL_LINK_STATUS)
    if success == gl.GL_FALSE:
        log = glGetProgramInfoLog(gl_program)
        print "Linking failed!\nLog:\n'" + log + "'"
        raise

    gl.glDetachShader(gl_program, vertex)
    gl.glDetachShader(gl_program, fragment)
    return gl_program


def build_buffer(gl_program, data, data_usage=gl.GL_STATIC_DRAW, buffer_target=gl.GL_ARRAY_BUFFER):
    with use_gl_program(gl_program):
        gl_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(buffer_target, gl_buffer)
        gl.glBufferData(buffer_target, data.nbytes, data, data_usage)
    gl.glBindBuffer(buffer_target, 0)
    return gl_buffer


def bind_buffer(gl_program, gl_buffer, gl_data):
    ''':type gl_data: GLData'''
    stride = gl_data.data.strides[0]
    offset = ctypes.c_void_p(0)
    loc = gl.glGetAttribLocation(gl_program, gl_data.attribute_name)
    gl.glEnableVertexAttribArray(loc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, gl_buffer)
    gl.glVertexAttribPointer(loc, gl_data.component_size, gl_data.type, False, stride, offset)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

class GLTexture(object):

    next_texture_slot = 1

    def __init__(self, filename, param=CLAMP_TEXTURE+LINEAR_TEXTURE):
        slot = self.next_texture_slot
        self.next_texture_slot += 1

        img = PIL.Image.open(filename)
        img_data = np.array(list(img.getdata()), np.int8)

        self.gl_texture = gl.glGenTextures(1)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        gl.glActiveTexture(gl.GL_TEXTURE0 + slot)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.gl_texture)

        for key, value in param:
            gl.glTexParameterf(gl.GL_TEXTURE_2D, key, value)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.size[0], img.size[1], 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data)

        gl.glActiveTexture(gl.GL_TEXTURE0)


class GLData(object):

    def __init__(self, data, attribute_name, data_is_mutable=False):
        self.data = np.ascontiguousarray(data)
        self.component_size = data.shape[1]
        self.attribute_name = attribute_name
        self.gl_usage = gl.GL_DYNAMIC_DRAW if data_is_mutable else gl.GL_STATIC_DRAW
        self.size = data.size
        self.type = recognize_gl_type(data)


class GLTask(object):

    def __init__(self, vertex_shader, fragment_shader,
                 gl_datas, geometry_indices, geometry_type=gl.GL_TRIANGLE_STRIP):
        ''':type gl_datas: list of GLData'''
        self.gl_program = None
        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader
        if not isinstance(gl_datas, collections.Iterable):
            gl_datas = [gl_datas]
        self.gl_datas = gl_datas
        self.geometry_indices = np.ndarray.astype(geometry_indices, np.uint32)
        self.gl_indices_buffer = None
        self.geometry_type = geometry_type

    def build(self):
        if self.gl_program is not None:
            raise
        self.gl_program = build_gl_program(self.vertex_shader, self.fragment_shader)
        self.gl_indices_buffer = build_buffer(self.gl_program, self.geometry_indices,
                                              gl.GL_STATIC_DRAW, gl.GL_ELEMENT_ARRAY_BUFFER)

        for gl_data in self.gl_datas:
            gl_buffer = build_buffer(self.gl_program, gl_data.data, gl_data.gl_usage)
            bind_buffer(self.gl_program, gl_buffer, gl_data)

    def bind_uniform(self, attribute_name, value, gl_uniform=gl.glUniform1f):
        ''':type gl_task: GLTask'''
        if self.gl_program is None:
            raise
        with use_gl_program(self.gl_program):
            loc = gl.glGetUniformLocation(self.gl_program, attribute_name)
            gl_uniform(loc, *value)

    def bind_texture(self, attribute_name, texture):
        ''':type texture: GLTexture'''
        self.bind_uniform(attribute_name, [texture.gl_texture], gl.glUniform1i)


class GLWindow(GlutWindow):

    def __init__(self, gl_tasks, limit_fps=100, show_fps=True,
                 window_name='OpenGL Window', size=np.asarray([768, 512])):
        ''':type gl_tasks: list of GLTask'''
        super(GLWindow, self).__init__(limit_fps, show_fps, window_name, size)
        for gl_task in gl_tasks:
            gl_task.build()
        self.gl_tasks = gl_tasks

    def display(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        for gl_task in self.gl_tasks:
            with use_gl_program(gl_task.gl_program):
                # gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, gl_task.gl_indices_buffer)
                indices = gl_task.geometry_indices
                gl.glDrawElements(gl_task.geometry_type, indices.size, recognize_gl_type(indices), indices)
                # gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        super(GLWindow, self).display()


if __name__ == '__main__':
    fill_flag = GLTask('''
        in vec4 position;

        void main()
        {
            gl_Position = position;
        }
        ''', '''
        void main()
        {
            gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);  // BLUE_COLOR
        }
        ''',
        [GLData(np.asarray([[-0.9, -0.9], [-0.9, +0.9], [+0.9, -0.9], [+0.9, +0.9]]), 'position')],
        np.asarray([[0, 1, 2], [0, 1, 3]]), geometry_type=gl.GL_TRIANGLES)
    win = GLWindow([fill_flag])
    win.run()
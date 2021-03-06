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

CLAMP_TO_EDGE_TEXTURE = [(gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE),
                         (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)]


LINEAR_TEXTURE = [(gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR),
                  (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)]

NEAREST_TEXTURE = [(gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST),
                  (gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)]


@contextmanager
def use_program(program):
    gl.glUseProgram(program)
    yield
    gl.glUseProgram(0)


def recognize_gl_type(data):
    return {'float32': gl.GL_FLOAT,
            'float64': gl.GL_DOUBLE,
            'uint32': gl.GL_UNSIGNED_INT
    }[data.dtype.name]


def build_program(vertex_code, fragment_code):
    program = gl.glCreateProgram()

    vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vertex, vertex_code)
    gl.glCompileShader(vertex)
    gl.glAttachShader(program, vertex)

    fragment = None
    if fragment_code is not None:
        fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fragment, fragment_code)
        gl.glCompileShader(fragment)
        gl.glAttachShader(program, fragment)

    gl.glLinkProgram(program)

    success = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if success == gl.GL_FALSE:
        log = glGetProgramInfoLog(program)
        print "Linking failed!\nLog:\n'" + log + "'"
        raise

    gl.glDetachShader(program, vertex)
    if fragment is not None:
        gl.glDetachShader(program, fragment)
    return program


def build_buffer(program, data, data_usage=gl.GL_STATIC_DRAW, buffer_target=gl.GL_ARRAY_BUFFER):
    # with use_program(program):
    buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(buffer_target, buffer)
    gl.glBufferData(buffer_target, data.nbytes, data, data_usage)
    gl.glBindBuffer(buffer_target, 0)
    return buffer


def load_buffer(program, buffer, data):
    """:type data: GLData"""
    stride = data.data.strides[0]
    offset = ctypes.c_void_p(0)
    loc = gl.glGetAttribLocation(program, data.attribute_name)
    gl.glEnableVertexAttribArray(loc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
    gl.glVertexAttribPointer(loc, data.component_size, data.type, False, stride, offset)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)


class Bindable(object):
    def __enter__(self):
        self.bind()

    def __exit__(self, *args):
        self.unbind()

    def bind(self):
        raise NotImplementedError("Abstract method call")

    def unbind(self):
        raise NotImplementedError("Abstract method call")


class Framebuffer(Bindable):

    def __init__(self):
        self.handle = gl.glGenFramebuffers(1)

    def bind(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.handle)

    def unbind(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)


class VertexBufferObject(Bindable):

    def __init__(self, target=gl.GL_ARRAY_BUFFER):
        self.target = target
        self.handle = gl.glGenBuffers(1)

    def bind(self):
        gl.glBindBuffer(self.target, self.handle)

    def unbind(self):
        gl.glBindBuffer(self.target, 0)


class VertexArrayObject(Bindable):

    def __init__(self):
        self.handle = gl.glGenVertexArrays(1)

    def bind(self):
        gl.glBindVertexArray(self.handle)

    def unbind(self):
        gl.glBindVertexArray(0)


class Texture(Bindable):

    target = None
    next_texture_slot = 1

    def __init__(self):
        self.slot = Texture.next_texture_slot
        Texture.next_texture_slot += 1
        self.handle = gl.glGenTextures(1)
        with self:
            gl.glBindTexture(self.target, self.handle)

    def get_next_slot(self):
        raise NotImplementedError("Abstract method call")

    def bind(self):
        gl.glActiveTexture(gl.GL_TEXTURE0 + self.slot)

    def unbind(self):
        gl.glActiveTexture(gl.GL_TEXTURE0)

    def release(self):
        gl.glDeleteTextures(np.uint32([self.handle]))

    def set_params(self, args):
        with self:
            for key, value in args:
                if isinstance(value, int):
                    gl.glTexParameteri(self.target, key, value)
                else:
                    gl.glTexParameterf(self.target, key, value)


class Texture1D(Texture):
    target = gl.GL_TEXTURE_1D


class Texture2D(Texture):
    target = gl.GL_TEXTURE_2D


class Texture1DArray(Texture):
    target = gl.GL_TEXTURE_1D_ARRAY


class Texture2DArray(Texture):
    target = gl.GL_TEXTURE_2D_ARRAY


def create_image_texture(filename, param=CLAMP_TEXTURE+LINEAR_TEXTURE):

    img = PIL.Image.open(filename)
    type = gl.GL_RGBA if img.format == 'PNG' else gl.GL_RGB
    channels_count = 4 if type == gl.GL_RGBA else 3
    img_data = np.array(list(img.getdata()), np.int8)
    img_data = img_data.reshape((img.size[1], img.size[0], channels_count))
    img_data = img_data[::-1, :, :]

    texture = Texture2D()
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

    with texture:
        for key, value in param:
            gl.glTexParameterf(gl.GL_TEXTURE_2D, key, value)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, type, img.size[0], img.size[1], 0, type, gl.GL_UNSIGNED_BYTE,
                        np.ascontiguousarray(img_data))
    return texture


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
                 datas, geometry_indices, geometry_type=gl.GL_TRIANGLE_STRIP):
        ''':type datas: list of GLData'''
        self.program = None
        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader
        if not isinstance(datas, collections.Iterable):
            datas = [datas]
        self.datas = datas
        self.geometry_indices = np.ndarray.astype(geometry_indices, np.uint32)
        self.indices_buffer = None
        self.geometry_type = geometry_type

    def build(self):
        if self.program is not None:
            raise
        self.program = build_program(self.vertex_shader, self.fragment_shader)

        self.vao = VertexArrayObject()
        with self.vao:
            self.indices_buffer = build_buffer(self.program, self.geometry_indices,
                                                  gl.GL_STATIC_DRAW, gl.GL_ELEMENT_ARRAY_BUFFER)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_buffer)
            for data in self.datas:
                buffer = build_buffer(self.program, data.data, data.gl_usage)
                load_buffer(self.program, buffer, data)

    @contextmanager
    def draw_context(self):
        yield

    def bind_uniform(self, attribute_name, value, gl_uniform=gl.glUniform1f):
        if self.program is None:
            raise
        with use_program(self.program):
            loc = gl.glGetUniformLocation(self.program, attribute_name)
            gl_uniform(loc, *value)

    def bind_texture(self, attribute_name, texture):
        """:type texture: Texture"""
        self.bind_uniform(attribute_name, [texture.slot], gl.glUniform1i)


class GLWindow(GlutWindow):

    def __init__(self, tasks, limit_fps=100, show_fps=True,
                 window_name='OpenGL Window', size=np.asarray([768, 512])):
        """:type tasks: list of GLTask"""
        super(GLWindow, self).__init__(limit_fps, show_fps, window_name, size)
        for task in tasks:
            task.build()
        self.tasks = tasks

    def display(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        for task in self.tasks:
            with use_program(task.program), task.draw_context(), task.vao:
                gl.glDrawElements(task.geometry_type, task.geometry_indices.size,
                                      recognize_gl_type(task.geometry_indices), None)
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
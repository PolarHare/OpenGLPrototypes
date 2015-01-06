from contextlib import contextmanager
import math
import numpy as np

import OpenGL.GL as gl

import commons.opengl as opengl
from commons import matrix

shadowcaster_vp = '''
uniform mat3 frustumMatrix;

attribute vec2 position;

varying vec2 frustum_position;

void main()
{
    vec3 xyw = frustumMatrix * vec3(position, 1.0);
    xyw /= xyw.z;
    frustum_position = xyw.xy;
    // gl_Position = vec4(xyw.x, 0.0, xyw.y, 1.0);
    gl_Position = vec4(position.x, 0.0, 0.0, 0.1);
}
'''

shadowcaster_fp = '''
varying vec2 frustum_position;

void main()
{
    gl_FragColor = vec4(0.5, 0.5, 0.5, 0.5);
    gl_FragDepth = 0.1;
}
'''


class DirectionalLight(object):

    def __init__(self, position, course, left, right, near, far):
        self.position = position
        self.course = course  # course is in degrees. If course = 0, than light is oriented as y-axis
        self.left = left
        self.right = right
        self.near = near
        self.far = far

    @classmethod
    def create_symmetric(cls, position, course, angle_of_view, near, far):
        radian_angle_of_view = angle_of_view * math.pi / 180
        right = math.tan(radian_angle_of_view / 2) * far
        left = -right
        return cls(position, course, left, right, near, far)

    def create_frustum_matrix(self):
        rotation = matrix.rotation_matrix_2d(-self.course * math.pi / 180)
        translation = matrix.translation_matrix_2d(*(-self.position))
        return matrix.frustum_matrix_2d(self.left, self.right, self.near, self.far).dot(rotation).dot(translation)


class GLDirectionalLightTask(opengl.GLTask):

    def __init__(self, light, gl_datas, geometry_indices, shadow_resolution=512):
        """:type light: DirectionalLight"""
        super(GLDirectionalLightTask, self).__init__(shadowcaster_vp, shadowcaster_fp, gl_datas, geometry_indices, gl.GL_POINTS)
        self.light = light
        self.light_was_changed = True
        self.shadow_resolution = shadow_resolution

        self.depth_tex = None
        self.color_tex = None
        self.framebuffer = None

    def build(self):
        super(GLDirectionalLightTask, self).build()
        self.depth_tex = opengl.Texture2D()
        self.color_tex = opengl.Texture2D()
        with self.depth_tex:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, self.shadow_resolution, 1, 0, gl.GL_DEPTH_COMPONENT,
                            gl.GL_FLOAT, None)
        with self.color_tex:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.shadow_resolution, 1, 0, gl.GL_RGBA,
                            gl.GL_UNSIGNED_BYTE, None)

        self.framebuffer = opengl.Framebuffer()
        with self.framebuffer:
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D,
                                      self.depth_tex.handle, 0)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D,
                                      self.color_tex.handle, 0)
        self.update_uniforms()

    @contextmanager
    def draw_context(self):
        with self.framebuffer:
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D,
                                      self.depth_tex.handle, 0)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D,
                                      self.color_tex.handle, 0)
            old_viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)

            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glClearDepth(0.9239)
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT | gl.GL_COLOR_BUFFER_BIT)
            gl.glViewport(0, 0, self.shadow_resolution, 1)
            # gl.glDrawBuffer(gl.GL_NONE)
            assert gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE

            print self.get_depth_array().max(), self.get_depth_array().min()
            print self.get_color_array().max(), self.get_color_array().min()

            yield

            print self.get_depth_array().max(), self.get_depth_array().min()
            print self.get_color_array().max(), self.get_color_array().min()
            # gl.glClearDepth(0.7)
            gl.glViewport(*old_viewport)
        print self.get_depth_array().max(), self.get_depth_array().min(), self.get_color_array().max(), self.get_color_array().min()

    def get_color_array(self):
        data = np.ndarray((self.shadow_resolution, 4), np.uint8)
        with self.framebuffer:
            gl.glReadPixels(0, 0, self.shadow_resolution, 1, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)

        data2 = np.ndarray((self.shadow_resolution, 4), np.uint8)
        with self.color_tex:
            gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data2)
        # if not np.all(data == data2):
        #     raise

        return data

    def get_depth_array(self):
        data = np.ndarray((self.shadow_resolution), np.float32)
        with self.framebuffer:
            gl.glReadPixels(0, 0, self.shadow_resolution, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, data)

        data2 = np.ndarray((self.shadow_resolution), np.float32)
        with self.depth_tex:
            gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, data2)
        # if not np.all(data == data2):
        #     raise

        return data

    def update_uniforms(self):
        if self.light_was_changed:
            frustum_matrix = self.light.create_frustum_matrix()
            self.bind_uniform('frustumMatrix', [1, False, np.ascontiguousarray(frustum_matrix.T)], gl.glUniformMatrix3fv)
            self.light_was_changed = False

    def touch_light(self):
        self.light_was_changed = True
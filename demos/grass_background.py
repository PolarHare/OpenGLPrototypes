import numpy as np

import commons.opengl as opengl
from commons.opengl import GLWindow
from commons.opengl import GLTask
from commons.opengl import GLData
import commons.matrix as matrix

import OpenGL.GL as gl

rectangle_vp = '''
#line 13
uniform mat3 world_to_camera;

attribute vec3 pos;

varying vec2 world_pos;

void main()
{
    world_pos = pos.xy;
    vec3 tmp = world_to_camera * pos;
    gl_Position = vec4(tmp.xyz, 1.0);
}
'''

rectangle_fp = '''
#line 29
uniform sampler2D grass_tex;

varying vec2 world_pos;

void main()
{
    gl_FragColor = texture2D(grass_tex, world_pos);
}
'''


class GrassWindow(GLWindow):

    def __init__(self, texture_fn="../data/grass.png", size=np.asarray([768, 512])):
        range = 10.0
        position = np.asarray([[-range, -range, 1.0], [-range, +range, 1.0], [+range, -range, 1], [+range, +range, 1]], np.float32)
        gl_datas = [GLData(position, 'pos')]

        self.grass_task = GLTask(rectangle_vp, rectangle_fp, gl_datas, np.asarray([[0, 1, 2], [1, 2, 3]]), geometry_type=gl.GL_TRIANGLES)
        super(GrassWindow, self).__init__([self.grass_task], window_name='Grass', size=size)
        self.grass_task.bind_texture('grass_tex',
                                     opengl.create_image_texture(texture_fn, opengl.CLAMP_TO_EDGE_TEXTURE + opengl.LINEAR_TEXTURE))
        self.camera_pos = np.asarray([0.0, 0.0])
        self.update_world()

    def update_world(self):
        size = self.window_size
        meters_in_screen = np.asarray([3.0, 3.0 * size[1] / size[0]])
        camera_rect = np.asarray([self.camera_pos - meters_in_screen / 2, self.camera_pos + meters_in_screen / 2])
        world_to_camera = matrix.rect_to_rect_matrix(camera_rect, np.asarray([[-1.0, -1.0], [1.0, 1.0]]))
        self.grass_task.bind_uniform("world_to_camera",
                                     [1, False, np.ascontiguousarray(world_to_camera.T)],
                                     gl.glUniformMatrix3fv)

    def reshape(self, width, height):
        super(GrassWindow, self).reshape(width, height)
        self.update_world()

    def keyboard(self, key, x, y):
        super(GrassWindow, self).keyboard(key, x, y)
        if key == 'a':
            self.camera_pos[0] -= 0.1
        if key == 'd':
            self.camera_pos[0] += 0.1
        if key == 'w':
            self.camera_pos[1] += 0.1
        if key == 's':
            self.camera_pos[1] -= 0.1
        self.update_world()


if __name__ == '__main__':
    win = GrassWindow()
    win.run()
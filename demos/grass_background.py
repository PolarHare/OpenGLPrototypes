import numpy as np

import commons.opengl as opengl
from commons.opengl import GLWindow
from commons.opengl import GLTask
from commons.opengl import GLData
import commons.matrix as matrix

import OpenGL.GL as gl

textured_rectangle_vp = '''
#line 13
uniform mat3 world_to_camera;

attribute vec3 position;

varying vec2 world_pos;

void main()
{
    world_pos = position.xy;
    vec3 tmp = world_to_camera * position;
    gl_Position = vec4(tmp.xy, 0.0, 1.0);
}
'''

textured_rectangle_fp = '''
#line 29
uniform sampler2D tex;

varying vec2 world_pos;

void main()
{
    gl_FragColor = texture(tex, world_pos);
}
'''


rectangle_vp = '''
uniform mat3 world_to_camera;

attribute vec2 position;
attribute vec4 color;

varying vec4 v_color;

void main()
{
    vec3 tmp = world_to_camera * vec3(position, 1.0);
    gl_Position = vec4(tmp.xy, 0.0, 1.0);
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


class GrassWindow(GLWindow):

    def __init__(self, grass_fn="../data/grass.png", kitten_fn="../data/kitten.jpg", size=np.asarray([768, 512])):
        grass_range = 10.0
        grass_position = np.asarray([[-grass_range, -grass_range, 1.0], [-grass_range, +grass_range, 1.0],
                                     [+grass_range, +grass_range, 1], [+grass_range, -grass_range, 1]], np.float32)
        grass_data = [GLData(grass_position, 'position')]

        kitten_range = 0.9
        kitten_position = np.asarray([[0, 0, 1], [0, kitten_range, 1],
                                     [kitten_range, kitten_range, 1], [kitten_range, 0, 1]], np.float32)
        kitten_data = [GLData(kitten_position, 'position')]

        box_lightness = 0.6
        box_position = np.asarray([[-0.1, 0.7], [0.1, 1.0], [0.7, 0.9], [0.2, 0.6]], np.float32)
        box_color = np.asarray([(box_lightness, 0, 0, 0.1), (0, box_lightness, 0, 0.7),
                            (0, 0, box_lightness, 0.7), (box_lightness, box_lightness, 0, 0.7)], np.float32)
        box_data = [GLData(box_position, 'position'), GLData(box_color, 'color')]

        self.grass_task = GLTask(textured_rectangle_vp, textured_rectangle_fp, grass_data, np.asarray([[0, 1, 2], [0, 2, 3]]), geometry_type=gl.GL_TRIANGLES)
        self.kitten_task = GLTask(textured_rectangle_vp, textured_rectangle_fp, kitten_data, np.asarray([[0, 1, 2], [0, 2, 3]]), geometry_type=gl.GL_TRIANGLES)
        self.box_task = GLTask(rectangle_vp, rectangle_fp, box_data, np.asarray([[0, 1, 2], [0, 2, 3]]), geometry_type=gl.GL_TRIANGLES)

        super(GrassWindow, self).__init__([self.grass_task, self.kitten_task, self.box_task], window_name='Grass', size=size)
        self.grass_task.bind_texture('tex',
                                     opengl.create_image_texture(grass_fn, opengl.REPEAT_TEXTURE + opengl.LINEAR_TEXTURE))
        self.kitten_task.bind_texture('tex',
                                      opengl.create_image_texture(kitten_fn, opengl.CLAMP_TEXTURE + opengl.LINEAR_TEXTURE))
        gl.glDepthFunc(gl.GL_LEQUAL)

        self.camera_pos = np.asarray([0.0, 0.0])
        self.move = np.asarray([0.0, 0.0])
        self.meters_in_width = 3.0
        self.update_world()

    def update_world(self):
        size = self.window_size

        meters_in_screen = np.asarray([self.meters_in_width, self.meters_in_width * size[1] / size[0]])
        camera_rect = np.asarray([self.camera_pos - meters_in_screen / 2, self.camera_pos + meters_in_screen / 2])
        world_to_camera = matrix.rect_to_rect_matrix(camera_rect, np.asarray([[-1.0, -1.0], [1.0, 1.0]]))
        self.grass_task.bind_uniform("world_to_camera",
                                     [1, False, np.ascontiguousarray(world_to_camera.T)],
                                     gl.glUniformMatrix3fv)
        self.kitten_task.bind_uniform("world_to_camera",
                                     [1, False, np.ascontiguousarray(world_to_camera.T)],
                                     gl.glUniformMatrix3fv)
        self.box_task.bind_uniform("world_to_camera",
                                     [1, False, np.ascontiguousarray(world_to_camera.T)],
                                     gl.glUniformMatrix3fv)

    def reshape(self, width, height):
        super(GrassWindow, self).reshape(width, height)
        self.update_world()

    def update(self, passed_time):
        if not np.all(self.move == 0.0):
            speed = self.meters_in_width * 0.002 * passed_time
            self.camera_pos += self.move * speed
            self.move = np.asarray([0.0, 0.0])
            self.update_world()
        super(GrassWindow, self).update(passed_time)

    def keyboard(self, key, x, y):
        super(GrassWindow, self).keyboard(key, x, y)
        if key == 'a':
            self.move[0] = -1
        if key == 'd':
            self.move[0] = 1
        if key == 'w':
            self.move[1] = 1
        if key == 's':
            self.move[1] = -1
        if key == '-':
            self.meters_in_width *= 1.1
            self.update_world()
        if key in {'+', '='}:
            self.meters_in_width /= 1.1
            self.update_world()

if __name__ == '__main__':
    win = GrassWindow()
    win.run()
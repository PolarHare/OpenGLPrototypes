import numpy as np

from commons.utils import current_time_ms
from commons.opengl import GLWindow
from commons.opengl import GLTask
from commons.opengl import GLData

import OpenGL.GL as gl

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


class RectangleWindow(GLWindow):

    def __init__(self, size=np.asarray([768, 512])):
        lightness = 0.6
        color = np.asarray([(lightness, 0, 0, 1), (0, lightness, 0, 1), (0, 0, lightness, 1), (lightness, lightness, 0, 1)], np.float32)
        position = np.asarray([(-1, -1), (-1, +1), (+1, -1), (+1, +1)], np.float32)
        gl_datas = [GLData(color, 'color'), GLData(position, 'position')]

        self.rectangle_task = GLTask(rectangle_vp, rectangle_fp, gl_datas, np.asarray([[0, 1, 2], [1, 2, 3]]), geometry_type=gl.GL_TRIANGLES)
        super(RectangleWindow, self).__init__(gl_tasks=[self.rectangle_task], window_name='Colored rectangle', size=size)

    def run(self):
        self.start_time = current_time_ms()
        super(RectangleWindow, self).run()

    def update(self, _):
        self.rectangle_task.bind_uniform('time', [current_time_ms() - self.start_time])
        width, height = self.window_size
        if width < height:
            self.rectangle_task.bind_uniform('scale_to_square', [1, width*1.0/float(height)], gl.glUniform2f)
        else:
            self.rectangle_task.bind_uniform('scale_to_square', [height*1.0/float(width), 1], gl.glUniform2f)
        super(RectangleWindow, self).update(_)


if __name__ == '__main__':
    win = RectangleWindow()
    win.run()
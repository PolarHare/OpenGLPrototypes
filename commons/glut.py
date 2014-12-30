import numpy as np

import OpenGL.GL as gl
import OpenGL.GLUT as glut

from commons import utils

glut.glutInit()

class GlutWindow(object):

    display_size = np.asarray([glut.glutGet(glut.GLUT_SCREEN_WIDTH), glut.glutGet(glut.GLUT_SCREEN_HEIGHT)])
    display_busy_up_to_height = 0
    display_busy_up_to_corner = np.asarray([0, 0])

    @staticmethod
    def allocate_window_position(window_size):
        # adding window to last row
        next_pos = np.asarray([GlutWindow.display_busy_up_to_corner[0], GlutWindow.display_busy_up_to_height])
        if np.any((next_pos + window_size) >= GlutWindow.display_size):
            # moving to next row
            next_pos = np.asarray([0, GlutWindow.display_busy_up_to_corner[1]])
            GlutWindow.display_busy_up_to_height = GlutWindow.display_busy_up_to_corner[1]
        if np.any((next_pos + window_size) >= GlutWindow.display_size):
            # all screen is busy - starting new layer of windows
            next_pos = np.asarray([0, 0])
            GlutWindow.display_busy_up_to_height = 0
        GlutWindow.display_busy_up_to_corner = next_pos + window_size
        return next_pos

    def __init__(self, limit_fps=100, show_fps=True, window_name='GlutWindow', size=np.asarray([768, 512])):
        self.limit_fps = limit_fps
        self.show_fps = show_fps
        self.window_name = window_name
        self.window_size = size

        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
        glut.glutInitWindowPosition(*GlutWindow.allocate_window_position(self.window_size))
        self.window_handle = glut.glutCreateWindow(self.window_name)
        glut.glutReshapeWindow(*self.window_size)
        glut.glutReshapeFunc(self.reshape)
        glut.glutDisplayFunc(self.display)
        glut.glutKeyboardFunc(self.keyboard)

    def update(self, _):
        if self.show_fps:
            current_time = utils.current_time_ms()
            passed_time = current_time - self.previous_update_time
            fps = 1000 // passed_time
            if current_time > self.previous_fps_print + 100:
                print fps, 'FPS\r',
                self.previous_fps_print = current_time
            self.previous_update_time = current_time
        glut.glutPostRedisplay()
        glut.glutTimerFunc(self.interval_ms, self.update, 0)

    def run(self):
        self.interval_ms = 0 if self.limit_fps <= 0 else 1000 / self.limit_fps
        if self.show_fps:
            self.previous_update_time = utils.current_time_ms()
            self.previous_fps_print = -1
        glut.glutTimerFunc(self.interval_ms, self.update, 0)
        glut.glutMainLoop()

    def display(self):
        glut.glutSwapBuffers()

    def reshape(self, width, height):
        self.window_size = np.asarray([width, height])
        gl.glViewport(0, 0, width, height)

    def keyboard(self, key, x, y):
        if key == '\033':
            glut.glutLeaveMainLoop()


if __name__ == '__main__':
    win = GlutWindow()
    win.run()
import ctypes
import numpy as np
from commons import matrix
from commons.utils import current_time_ms, timer
from commons.opengl import build_program, use_program, VertexArrayObject, VertexBufferObject, create_image_texture, \
    CLAMP_TEXTURE, LINEAR_TEXTURE, REPEAT_TEXTURE

import OpenGL.GL as gl
import OpenGL.GLUT as glut

textured_rectangle_vp = '''
#version 150
#line 13
uniform mat3 world_to_camera;

in vec2 texture_coordinate;
in vec2 position;

out VertexData {
    vec2 texture_coordinate;
} VertexOut;

void main()
{
    vec3 camera_position = world_to_camera * vec3(position, 1.0);
    gl_Position = vec4(camera_position.xy, 0.0, camera_position.z);
    VertexOut.texture_coordinate = texture_coordinate;
}
'''

textured_rectangle_fp = '''
#version 150
#line 33
uniform sampler2D tex;

in VertexData {
    vec2 texture_coordinate;
} VertexIn;

void main()
{
    gl_FragColor = texture(tex, VertexIn.texture_coordinate);
}
'''

if __name__ == '__main__':
    width, height = 768, 512
    rect_range = 1000
    texture_width = 1.5

    camera_pos = np.asarray([0.0, 0.0])
    meters_in_width = 4.0

    position = np.asarray([[-rect_range, -rect_range], [-rect_range, +rect_range], [+rect_range, -rect_range], [+rect_range, +rect_range]], np.float32)
    texture_coordinate = position / texture_width
    indices = np.asarray([[0, 1, 2], [1, 2, 3]], np.uint8)

    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
    glut.glutCreateWindow('Textured rectangle')
    glut.glutReshapeWindow(width, height)

    gl_program = build_program(textured_rectangle_vp, textured_rectangle_fp)

    vao = VertexArrayObject()

    with vao:
        indices_buffer = VertexBufferObject(gl.GL_ELEMENT_ARRAY_BUFFER)
        with indices_buffer:
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        position_buffer = VertexBufferObject()
        with position_buffer:
            gl.glBufferData(gl.GL_ARRAY_BUFFER, position.nbytes, position, gl.GL_STATIC_DRAW)
            loc = gl.glGetAttribLocation(gl_program, 'position')
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, position.strides[0], ctypes.c_void_p(0))

        tex_coord_buffer = VertexBufferObject()
        with tex_coord_buffer:
            gl.glBufferData(gl.GL_ARRAY_BUFFER, texture_coordinate.nbytes, texture_coordinate, gl.GL_STATIC_DRAW)
            loc = gl.glGetAttribLocation(gl_program, 'texture_coordinate')
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, texture_coordinate.strides[0], ctypes.c_void_p(0))

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, indices_buffer.handle)

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    log = gl.glGetAttribLocation(gl_program, 'tex')
    with timer('texture loading'):
        grass_tex = create_image_texture("../data/grass.png", REPEAT_TEXTURE + LINEAR_TEXTURE)

    with use_program(gl_program), grass_tex:
        gl.glUniform1i(loc, grass_tex.slot)

    def update_world():
        meters_in_screen = np.asarray([meters_in_width, meters_in_width * height / width])
        camera_rect = np.asarray([camera_pos - meters_in_screen / 2, camera_pos + meters_in_screen / 2])
        world_to_camera = matrix.rect_to_rect_matrix(camera_rect, np.asarray([[-1.0, -1.0], [1.0, 1.0]]))
        with use_program(gl_program):
            loc = gl.glGetUniformLocation(gl_program, 'world_to_camera')
            gl.glUniformMatrix3fv(loc, 1, False, world_to_camera.T)

    update_world()

    def keyboard(key, x, y):
        global camera_pos
        speed = 0.1
        if key == 'w':
            camera_pos[1] += speed
        elif key == 's':
            camera_pos[1] -= speed
        elif key == 'd':
            camera_pos[0] += speed
        elif key == 'a':
            camera_pos[0] -= speed
        if key in {'w', 's', 'd', 'a'}:
            update_world()
        if key == '\033':
            glut.glutLeaveMainLoop()

    def reshape(new_width, new_height):
        global width, height
        width, height = new_width, new_height
        gl.glViewport(0, 0, width, height)

    def update(_):
        with use_program(gl_program):
            glut.glutPostRedisplay()
            glut.glutTimerFunc(0, update, 0)

    last_display_time = current_time_ms()

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
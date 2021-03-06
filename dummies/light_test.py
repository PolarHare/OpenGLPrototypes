import copy
import ctypes
from math import atan2, pi

import numpy as np
import matplotlib.pyplot as plt

from commons import matrix
from commons.matrix import rect_to_rect_matrix
from commons.utils import current_time_ms, timer
from commons.opengl import build_program, use_program, VertexArrayObject, VertexBufferObject, create_image_texture, \
    CLAMP_TEXTURE, LINEAR_TEXTURE, REPEAT_TEXTURE, NEAREST_TEXTURE, Framebuffer, Texture1DArray
from entities.light import DirectionalLight

import OpenGL.GL as gl
import OpenGL.GLUT as glut

LIGHTS_MAX = 32

textured_rectangle_vp = '''
#version 150
#line 20
uniform mat3 world_to_camera;

in vec2 texture_coordinate;
in vec2 position;

out VertexData {
    vec2 texture_coordinate;
    vec2 position;
} VertexOut;

void main()
{
    vec3 camera_position = world_to_camera * vec3(position, 1.0);
    VertexOut.texture_coordinate = texture_coordinate;
    VertexOut.position = position;
    gl_Position = vec4(camera_position.xy, 0.0, camera_position.z);
}
'''

textured_rectangle_fp = '''
#version 150
#line 42
uniform sampler2D background_tex;

uniform int lights_count;

uniform sampler1DArray light_depth_tex;

uniform mat3 world_to_light_depth_tex[LIGHTS_MAX];
uniform vec2 light_pos[LIGHTS_MAX];
uniform float light_range[LIGHTS_MAX];

in VertexData {
    vec2 texture_coordinate;
    vec2 position;
} VertexIn;

void main()
{
    gl_FragColor = texture(background_tex, VertexIn.texture_coordinate);
    float min_light = 0.2;
    float max_light = min_light;
    for (int i = 0; i < lights_count; i++) {
        vec3 light_tex_pos = world_to_light_depth_tex[i] * vec3(VertexIn.position, 1.0);
        float x = light_tex_pos.x;
        float y = light_tex_pos.y;
        float z = light_tex_pos.z;
        if (x > 0 && x < 1.0*z && y > 0 && y < 1.0*z) {
            float dist_from_light = length(VertexIn.position - light_pos[i]);
            float light_depth = texture(light_depth_tex, vec2(x/z, i)).r;
            if (y/z <= light_depth && dist_from_light*dist_from_light <= light_range[i] * light_range[i]) {
                // float a = (min_light - 1.0) / (light_range[i] * light_range[i]);
                // float parabolic_shadow = a*dist_from_light*dist_from_light + 1;
                float linear_shadow = 1.0 + dist_from_light * (min_light - 1.0) / light_range[i];
                max_light = max(linear_shadow, max_light);
            }
        }
    }
    gl_FragColor *= max_light;
}
'''.replace('LIGHTS_MAX', str(LIGHTS_MAX))

shadowcaster_vp = '''
#version 150
#line 85
uniform mat3 world_to_light_camera;

in vec2 position;

void main()
{
    vec3 camera_position = world_to_light_camera * vec3(position, 1.0);
    gl_Position = vec4(camera_position.x, 0.0, camera_position.y, camera_position.z);
}
'''


def get_depth(width, height, debug=False):
    data = np.ndarray((height, width), np.float32)
    gl.glReadPixels(0, 0, width, height, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, data)
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


def get_depth_tex(tex, width, height, debug=False):
    data = np.ndarray((height, width), np.float32)
    with tex:
        gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, data)
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
    width, height = 768, 512
    shadow_resolution = 2048
    rect_range = 1000
    texture_width = 1.5

    camera_pos = np.asarray([0.0, 0.0])
    meters_in_width = 15.0
    light_range = 5.0

    window_to_world = None

    position = np.asarray([[-rect_range, -rect_range], [-rect_range, +rect_range], [+rect_range, -rect_range],
                           [+rect_range, +rect_range]], np.float32)
    texture_coordinate = position / texture_width
    indices = np.asarray([[0, 1, 2], [1, 2, 3]], np.uint8)

    box_width, box_height = 0.1, 0.1
    box_split_x, box_split_y = 0.5, 0.5
    box_offset_x, box_offset_y = -1.0, -1.0
    box_first = np.array([[box_offset_x, box_offset_y], [box_offset_x, box_offset_y + box_height],
                          [box_offset_x + box_width, box_offset_y + box_height],
                          [box_offset_x + box_width, box_offset_y]])
    box_first_indices = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    box_row_step = np.array([box_width + box_split_x, 0])
    box_col_step = np.array([0, box_height + box_split_y])
    box_rows, box_columns = 5, 5
    box_position = np.ascontiguousarray(np.asarray([[box_first + box_row_step * col + box_col_step * row
                                                     for col in range(box_columns)]
                                                    for row in range(box_rows)], np.float32).ravel())
    box_indices = np.ascontiguousarray(np.asarray([[box_first_indices + box_first.shape[0] * (box_columns * row + col)
                                                    for col in range(box_columns)]
                                                   for row in range(box_rows)], np.uint8).ravel())
    box_position_stride = 8  # box_position.strides[2]

    light = DirectionalLight.create_symmetric(np.asarray([0.0, 0.0]), 0, 120, 0.01, light_range)
    lights_count = 1
    lights = [light]

    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
    glut.glutCreateWindow('Light test')
    glut.glutReshapeWindow(width, height)

    gl_program = build_program(textured_rectangle_vp, textured_rectangle_fp)
    gl_program_shadow = build_program(shadowcaster_vp, None)

    vao = VertexArrayObject()
    vao_shadow = VertexArrayObject()

    with vao:
        indices_buffer = VertexBufferObject(gl.GL_ELEMENT_ARRAY_BUFFER)
        with indices_buffer:
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, indices_buffer.handle)

        position_buffer = VertexBufferObject()
        with position_buffer:
            gl.glBufferData(gl.GL_ARRAY_BUFFER, position.nbytes, position, gl.GL_STATIC_DRAW)
            background_loc = gl.glGetAttribLocation(gl_program, 'position')
            gl.glEnableVertexAttribArray(background_loc)
            gl.glVertexAttribPointer(background_loc, 2, gl.GL_FLOAT, False, position.strides[0], ctypes.c_void_p(0))

        tex_coord_buffer = VertexBufferObject()
        with tex_coord_buffer:
            gl.glBufferData(gl.GL_ARRAY_BUFFER, texture_coordinate.nbytes, texture_coordinate, gl.GL_STATIC_DRAW)
            background_loc = gl.glGetAttribLocation(gl_program, 'texture_coordinate')
            gl.glEnableVertexAttribArray(background_loc)
            gl.glVertexAttribPointer(background_loc, 2, gl.GL_FLOAT, False, texture_coordinate.strides[0],
                                     ctypes.c_void_p(0))

    with vao_shadow:
        indices_buffer_shadow = VertexBufferObject(gl.GL_ELEMENT_ARRAY_BUFFER)
        with indices_buffer_shadow:
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, box_indices.nbytes, box_indices, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, indices_buffer_shadow.handle)

        position_buffer_shadow = VertexBufferObject()
        with position_buffer_shadow:
            gl.glBufferData(gl.GL_ARRAY_BUFFER, box_position.nbytes, box_position, gl.GL_STATIC_DRAW)
            background_loc = gl.glGetAttribLocation(gl_program_shadow, 'position')
            gl.glEnableVertexAttribArray(background_loc)
            gl.glVertexAttribPointer(background_loc, 2, gl.GL_FLOAT, False, box_position_stride, ctypes.c_void_p(0))

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    def update_world():
        global window_to_world
        meters_in_screen = np.asarray([meters_in_width, meters_in_width * height / width])
        camera_rect = np.asarray([camera_pos - meters_in_screen / 2, camera_pos + meters_in_screen / 2])
        window_to_world = matrix.rect_to_rect_matrix([[0, 0], [width, height]], camera_rect).dot(
            matrix.invert_y_in_range(0, height))
        world_to_camera = matrix.rect_to_rect_matrix(camera_rect, [[-1.0, -1.0], [1.0, 1.0]])
        with use_program(gl_program):
            loc = gl.glGetUniformLocation(gl_program, 'world_to_camera')
            gl.glUniformMatrix3fv(loc, 1, False, world_to_camera.T)

    def update_light(layer_index):
        cur_light = lights[layer_index]
        frustum_matrix = cur_light.create_frustum_matrix()
        with use_program(gl_program):
            loc = gl.glGetUniformLocation(gl_program, 'world_to_light_depth_tex[{}]'.format(layer_index))
            matrix = rect_to_rect_matrix([[-1, -1], [1, 1]], [[0, 0], [1, 1]]).dot(frustum_matrix)
            gl.glUniformMatrix3fv(loc, 1, False, matrix.T)
            light_pos_loc = gl.glGetUniformLocation(gl_program, 'light_pos[{}]'.format(layer_index))
            gl.glUniform2f(light_pos_loc, cur_light.position[0], cur_light.position[1])
            light_range2_loc = gl.glGetUniformLocation(gl_program, 'light_range[{}]'.format(layer_index))
            gl.glUniform1f(light_range2_loc, cur_light.far)

            lights_count_loc = gl.glGetUniformLocation(gl_program, 'lights_count')
            gl.glUniform1i(lights_count_loc, lights_count)
        with use_program(gl_program_shadow):
            with shadow_framebuffer:
                gl.glFramebufferTextureLayer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, depth_tex.handle, 0,
                                             layer_index)
            loc = gl.glGetUniformLocation(gl_program_shadow, 'world_to_light_camera')
            matrix = frustum_matrix
            gl.glUniformMatrix3fv(loc, 1, False, matrix.T)

    with timer('texture loading'):
        grass_tex = create_image_texture("../data/kitten.jpg", REPEAT_TEXTURE + LINEAR_TEXTURE)

    shadow_framebuffer = Framebuffer()
    assert gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE

    depth_tex = Texture1DArray()
    depth_tex.set_params(CLAMP_TEXTURE + NEAREST_TEXTURE)
    with depth_tex:
        gl.glTexImage2D(depth_tex.target, 0, gl.GL_DEPTH_COMPONENT32F, shadow_resolution, LIGHTS_MAX, 0, gl.GL_DEPTH_COMPONENT,
                        gl.GL_FLOAT, None)

    background_loc = gl.glGetUniformLocation(gl_program, 'background_tex')
    light_depth_loc = gl.glGetUniformLocation(gl_program, 'light_depth_tex')
    with use_program(gl_program):
        gl.glUniform1i(background_loc, grass_tex.slot)
        gl.glUniform1i(light_depth_loc, depth_tex.slot)

    def render_light_depth():
        with shadow_framebuffer:
            gl.glViewport(0, 0, shadow_resolution, 1)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            gl.glDrawBuffer(gl.GL_NONE)
            with use_program(gl_program_shadow), vao_shadow:
                gl.glDrawElements(gl.GL_LINES, box_indices.size, gl.GL_UNSIGNED_BYTE, None)
                # get_depth(shadow_resolution, 1, debug=True)

    update_world()
    update_light(lights_count - 1)
    with timer('rendering light'):
        render_light_depth()

    mouse_down_click = None

    def mouse(button, state, x, y):
        GLUT_WHEEL_UP, GLUT_WHEEL_DOWN = 3, 4
        global light, lights_count, mouse_down_click
        click_pos = window_to_world.dot(np.array([x, y, 1.0]))
        click_pos = click_pos[:2] / click_pos[2]
        if button == glut.GLUT_LEFT_BUTTON:
            if state == glut.GLUT_DOWN:
                mouse_down_click = click_pos
                light.position = mouse_down_click
            elif state == glut.GLUT_UP:
                mouse_down_click = None
                if len(lights) < LIGHTS_MAX:
                    light = copy.deepcopy(light)
                    lights_count += 1
                    lights.append(light)
        elif button in {GLUT_WHEEL_UP, GLUT_WHEEL_DOWN}:
            direction = 1 if button == GLUT_WHEEL_UP else -1
            light.course += direction * 3
            update_light(lights_count - 1)
            render_light_depth()

    def mouse_moved(x, y):
        global light
        click_pos = window_to_world.dot(np.array([x, y, 1.0]))
        click_pos = click_pos[:2] / click_pos[2]
        if mouse_down_click is not None:
            delta_x, delta_y = click_pos - mouse_down_click
            if abs(delta_x) > 0.1 or abs(delta_y) > 0.1:
                light.course = -90 + atan2(delta_y, delta_x) * 180 / pi
            update_light(lights_count - 1)
            render_light_depth()
        else:
            light.position = click_pos
            update_light(lights_count - 1)
            render_light_depth()

    max_fps = 0

    def keyboard(key, x, y):
        global camera_pos, meters_in_width, max_fps
        speed = meters_in_width / 23.9
        if key == 'w':
            camera_pos[1] += speed
        elif key == 's':
            camera_pos[1] -= speed
        elif key == 'd':
            camera_pos[0] += speed
        elif key == 'a':
            camera_pos[0] -= speed
        elif key == '\033':
            glut.glutLeaveMainLoop()
        elif key == '-':
            meters_in_width *= 1.1
        elif key in {'+', '='}:
            meters_in_width /= 1.1
        elif key == ' ':
            max_fps = 0
        if key in {'w', 's', 'd', 'a', '-', '+'}:
            update_world()

    def reshape(new_width, new_height):
        global width, height
        width, height = new_width, new_height
        gl.glViewport(0, 0, width, height)
        update_world()

    def update(_):
        with use_program(gl_program):
            glut.glutPostRedisplay()
            glut.glutTimerFunc(0, update, 0)

    last_display_time = current_time_ms()

    def display():
        global last_display_time, max_fps
        current_time = current_time_ms()
        fps = 1000 / (current_time - last_display_time)
        max_fps = max(fps, max_fps)
        print '{} FPS  (max fps: {})\r'.format(fps, max_fps),
        last_display_time = current_time

        gl.glViewport(0, 0, width, height)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        with use_program(gl_program), vao:
            gl.glDrawElements(gl.GL_TRIANGLES, indices.size, gl.GL_UNSIGNED_BYTE, None)

        glut.glutSwapBuffers()

    glut.glutMouseFunc(mouse)
    glut.glutMotionFunc(mouse_moved)
    glut.glutPassiveMotionFunc(mouse_moved)
    glut.glutKeyboardFunc(keyboard)
    glut.glutReshapeFunc(reshape)
    glut.glutDisplayFunc(display)
    glut.glutTimerFunc(0, update, 0)
    glut.glutMainLoop()
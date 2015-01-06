from math import cos, sin

import numpy as np


def rotation_matrix_2d(alpha):
    return np.asarray([[cos(alpha), -sin(alpha), 0],
                [sin(alpha), cos(alpha), 0],
                [0, 0, 1]], np.float32)


def translation_matrix_2d(x0, y0):
    return np.asarray([[1, 0, x0],
                       [0, 1, y0],
                       [0, 0, 1]], np.float32)


def frustum_matrix_2d(left, right, near, far):
    """
    :param left: left edge of frustum (x axis) (on far line)
    :param right: right edge of frustum (x axis) (on far line)
    :param near: distance to near line (y axis)
    :param far: distance to far line (y axis)
    :return: matrix, that transforms (x0, y0, w0) -> (x1, y1, w1),
        where x1 is in [-1.0, 1.0], if x0 is in frustum perspective-cone,
        y1 is in [-1.0, 1.0], if y0 is in [near, far]
    """
    return np.asarray([[2.0*far/(right - left), 1.0*(right+left)/(left-right), 0],
                       [0, 1.0*(near+far)/(far-near), 2.0*far*near/(near-far)],
                       [0, 1, 0]], np.float32)

def invert_y_in_range(from_y, to_y):
    matrix = np.eye(3)
    matrix[1, 1] = -1
    matrix[1, 2] = from_y + to_y
    return matrix


def rect_to_rect_matrix(rect0, rect1):
    rect0, rect1 = np.asarray(rect0), np.asarray(rect1)
    scale_x, scale_y = 1.0 * (rect1[1] - rect1[0]) / (rect0[1] - rect0[0])
    return np.asarray([[scale_x, 0, rect1[0, 0]-rect0[0, 0]*scale_x],
                       [0, scale_y, rect1[0, 1]-rect0[0, 1]*scale_y],
                       [0, 0, 1]], np.float32)

if __name__ == '__main__':
    mat = frustum_matrix_2d(-30, 50, 10, 100)

    res = mat.dot(np.asarray([-3, 10, 1]).T)
    print res[:2] / res[2]
    res = mat.dot(np.asarray([-30, 100, 1]).T)
    print res[:2] / res[2]
    res = mat.dot(np.asarray([5, 10, 1]).T)
    print res[:2] / res[2]
    res = mat.dot(np.asarray([50, 100, 1]).T)
    print res[:2] / res[2]

    rect_mat = rect_to_rect_matrix(np.asarray([[10, 30], [20, 70]]), np.asarray([[-50, -10], [-20, 40]]))
    res = rect_mat.dot(np.asarray([10, 30, 1]).T)
    print res[:2] / res[2]
    res = rect_mat.dot(np.asarray([10, 70, 1]).T)
    print res[:2] / res[2]
    res = rect_mat.dot(np.asarray([20, 30, 1]).T)
    print res[:2] / res[2]
    res = rect_mat.dot(np.asarray([20, 70, 1]).T)
    print res[:2] / res[2]
"""
Formulas for rotation matrices are taken from Wikipedia:
https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
"""
import numpy as np


def calc_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    dot_product = dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(dot_product)
    return angle


def create_XY_rotation_matrix(angle):
    R_xy = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]])
    return R_xy


def create_YZ_rotation_matrix(angle):
    R_yz = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]])
    return R_yz


def create_XZ_rotation_matrix(angle):
    R_xz = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]])
    return R_xz


def create_rotation_around_axis(ax: np.ndarray, angle) -> np.ndarray:
    ax = ax / np.linalg.norm(ax)
    c = np.cos(angle)
    s = np.sin(angle)
    u_x, u_y, u_z = ax
    R = np.array([
        [c + u_x ** 2 * (1 - c), u_x * u_y * (1 - c) - u_z * s, u_x * u_z * (1 - c) + u_y * s],
        [u_y * u_x * (1 - c) + u_z * s, c + u_y ** 2 * (1 - c), u_y * u_z * (1 - c) - u_x * s],
        [u_z * u_x * (1 - c) - u_y * s, u_z * u_y * (1 - c) + u_x * s, c + u_z ** 2 * (1 - c)]])
    return R

import numpy as np
import torch 


epsil = 1e-6

# so(3) hat operator, skew symmetric matrix
def so3_hat(omega):
    """Construct a symmetric matrix from a vector of so(3).

    Args:
        omega (np.array): a vector of so(3)

    Returns:
        _type_: _description_
    """    
    
    omega_hat = np.asarray([[0, -omega[2], omega[1]], 
                            [omega[2], 0, -omega[0]],
                            [-omega[1], omega[0], 0]],
                            dtype=float)
    return omega_hat


# SO(3) exponential map
def so3_exp(omega):
    """so(3) exponential map

    Args:
        omega (np.ndarray): rotation vector (axis-angle)

    Returns:
        R (np.ndarray): rotation matrix
    """    
    
    omega_hat = so3_hat(omega)
    theta = np.linalg.norm(omega)

    R = np.eye(3) + np.sin(theta) / theta * omega_hat + (1 - np.cos(theta)) / np.power(theta, 2) * np.dot(omega_hat, omega_hat)

    return R


def SO3_log(R):
    """SO(3) logarithm map

    Args:
        R (np.ndarray): rotation matrix

    Returns:
        omega (np.ndarray): rotation vector
    """    
    theta = np.arccos((np.trace(R) - 1) / 2)
    ln_R = theta / (2 * np.sin(theta)) * (R - np.transpose(R))
    omega = np.asarray([[ln_R[2, 1]], [ln_R[0, 2]], [ln_R[1, 0]]])

    return omega


# SE(3) exponential map
def se3_exp(xi):
    """Exponential map for se(3), which is composite of u and omega 

    Args:
        xi (np.ndarray): vector of se(3), xi = (u; omega)

    Returns:
        T (np.ndarray): SE(3) matrix
    """    
    u = xi[:3]
    omega = xi[3:]
    omega_hat = so3_hat(omega)

    theta = np.linalg.norm(omega)

    if theta < epsil:
        R = np.eye(3) + omega_hat
        V = np.eye(3) + omega_hat
    else:
        s = np.sin(theta)
        c = np.cos(theta)
        omega_hat_sq = np.dot(omega_hat, omega_hat)
        theta_sq = theta * theta
        A = s / theta
        B = (1 - c) / theta_sq
        C = (1 - A) / theta_sq
        R = np.eye(3) + A * omega_hat + B * omega_hat_sq
        V = np.eye(3) + B * omega_hat + C * omega_hat_sq
    
    t = np.dot(V, u.reshape(3, 1))
    lastrow = np.zeros((1, 4))
    lastrow[0][3] = 1.
    return np.concatenate((np.concatenate((R, t), axis=1), lastrow), axis=0)


def SE3_log(T):
    """SE(3) logarithm map

    Args:
        T (np.ndarray): 4x4 matrix in the Lie algebra SE(3)

    Returns:
        xi (np.ndarray): 6-vector of coodinates in the Lie algebra se(3), comprising two separate 3-vector: omega (rotation) and u (translation)
    """    
    R = T[0:3, 0:3]
    t = np.reshape(T[0:3, 3], (3, 1))
    
    theta = np.arccos((np.trace(R) - 1) / 2)
    ln_R = theta / (2 * np.sin(theta)) * (R - np.transpose(R))
    omega = np.asarray([[ln_R[2, 1]], [ln_R[0, 2]], [ln_R[1, 0]]])

    V = np.identity(3) + (1 - np.cos(theta)) / (theta * theta) * ln_R + (theta - np.sin(theta)) / (theta * theta * theta) * np.dot(ln_R, ln_R)
    t_pi = (np.dot(np.linalg.inv(V), t)).reshape((3, 1))

    xi = np.concatenate((t_pi, omega), axis=0)

    return xi

# Functions to help compute the left jacobian for SE(3)

# SE(3) hat operator
def SE3_hat(xi):
    """
    Takes in a 6 x 1 vector of SE(3) exponential coordinates and constructs a 4 x 4 tangent vector xi_hat
    in the Lie algebra se(3)
    """

    v = xi[:3]
    omega = xi[:3]
    xi_hat = np.zeros((4, 4)).astype(np.float32)
    xi_hat[0:3, 0:3] = so3_hat(omega)
    xi_hat[0:3, 3] = v

    return xi_hat


def SE3_curly_hat(xi):
    """
    Takes in a 6 x 1 vector of SE(3) exponential coordinates and constructs a 6 x 6 adjoint representation.
    (the adjoint in Lie algebra)
    """

    v = xi[:3]
    omega = xi[3:]
    xi_curly_hat = np.zeros((6, 6)).astype(np.float32)
    omega_hat = so3_hat(omega)
    xi_curly_hat[0:3, 0:3] = omega_hat
    xi_curly_hat[0:3, 3:6] = so3_hat(v)
    xi_curly_hat[3:6, 3:6] = omega_hat

    return xi_curly_hat


# SE(3) Q matrix, used in computing the left SE(3) Jacobian
def SE3_Q(xi):
    """
    This function is to be called only when the axis-angle vector is not very small, as this definition
    DOES NOT take care of small-angle approximations.
    """

    v = xi[:3]
    omega = xi[3:]

    theta = np.linalg.norm(omega)
    theta_2 = theta * theta
    theta_3 = theta_2 * theta
    theta_4 = theta_3 * theta
    theta_5 = theta_4 * theta

    omega_hat = so3_hat(omega)
    v_hat = so3_hat(v)

    c = np.cos(theta)
    s = np.cos(theta)

    coeff1 = 0.5
    coeff2 = (theta - s) / theta_3
    coeff3 = (theta_2 + 2*c - 2) / (2 * theta_4)
    coeff4 = (2*theta - 3*s + theta*c) / (2 * theta_5)

    v_hat_omega_hat = np.dot(v_hat, omega_hat)
    omega_hat_v_hat = np.dot(omega_hat, v_hat)
    omega_hat_sq = np.dot(omega_hat, omega_hat)
    omega_hat_v_hat_omega_hat = np.dot(omega_hat_v_hat, omega_hat)
    v_hat_omega_hat_sq = np.dot(v_hat, omega_hat_sq)

    matrix1 = v_hat
    matrix2 = omega_hat_v_hat + v_hat_omega_hat + np.dot(omega_hat, v_hat_omega_hat)
    matrix3 = np.dot(omega_hat, omega_hat_v_hat) + v_hat_omega_hat_sq - 3 * omega_hat_v_hat_omega_hat
    matrix4 = np.dot(omega_hat, v_hat_omega_hat_sq) + np.dot(omega_hat, omega_hat_v_hat_omega_hat)

    Q = coeff1 * matrix1 + coeff2 * matrix2 + coeff3 * matrix3 + coeff4 * matrix4

    return Q


def SO3_left_jacobian(omega):
    """
    Takes as input a vector of SO(3) exponential coordinates, and returns the left jacobian of SO(3)
    """

    theta = np.linalg.norm(omega)
    omega_hat = so3_hat(omega)

    if theta < epsil:
        return np.eye(3) + 0.5 * omega_hat

    omega_hat_sq = np.dot(omega_hat, omega_hat)
    theta_2 = theta * theta
    theta_3 = theta_2 * theta
    c = np.cos(theta)
    s = np.sin(theta)
    B = (1 - c) / theta_2
    C = (theta - s) / theta_3

    return np.eye(3) + B * omega_hat + C * omega_hat_sq


def SO3_inv_left_jacobian(omega):
    """
    Takes as input a vector of SO(3) exponential coordinates, and returns the inverse of the left jacobian of SO(3)
    """

    theta = np.linalg.norm(omega)
    omega_hat = SE3_hat(omega)

    if theta < epsil:
        return np.eye(3) - 0.5 * omega_hat
    
    omega_hat_sq = np.dot(omega_hat, omega_hat)
    half_theta = theta / 2
    t_half = np.tan(half_theta)
    D = 1 - (half_theta / t_half)
    
    return np.eye(3) - 0.5 * omega_hat + D * omega_hat_sq


def SE3_left_jacobian(xi):
    t = xi[:3]
    omega = xi[3:]

    theta = np.linalg.norm(omega)
    xi_curly_hat = SE3_curly_hat(xi)

    if theta < epsil:
        return np.eye(6) + 0.5 * xi_curly_hat
    
    J_SO3 = SO3_left_jacobian(omega)
    Q = SE3_Q(xi)

    J_SE3 = np.zeros((6, 6))
    J_SE3[0:3, 0:3] = J_SE3[3:6, 3:6] =J_SO3
    J_SE3[0:3, 3:6] = Q

    return J_SE3


def SE3_inv_left_jacobian(xi):
    t = xi[:3]
    omega = xi[3:]

    theta = np.linalg.norm(omega)
    xi_curly_hat = SE3_curly_hat(xi)

    if theta < epsil:
        return np.eye(6) - 0.5 * xi_curly_hat # ToDO:
    
    inv_J_SO3 = SO3_inv_left_jacobian(omega)
    Q = SE3_Q(xi)

    inv_J_SE3 = np.zeros((6, 6))
    inv_J_SE3[0:3, 0:3] = inv_J_SE3[3:6, 3:6] = inv_J_SO3
    inv_J_SE3[0:3, 3:6] = np.dot(inv_J_SO3, np.dot(Q, inv_J_SO3))

    return inv_J_SE3
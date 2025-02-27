import rowan
import numpy as np

def from_euler(alpha, beta, gamma, convention="zyx", axis_type="intrinsic"):
    r"""Convert Euler angles to quaternions.

    For generality, the rotations are computed by composing a sequence of
    quaternions corresponding to axis-angle rotations. While more efficient
    implementations are possible, this method was chosen to prioritize
    flexibility since it works for essentially arbitrary Euler angles as
    long as intrinsic and extrinsic rotations are not intermixed.

    Args:
        alpha ((...) :class:`numpy.ndarray`):
            Array of :math:`\alpha` values in radians.
        beta ((...) :class:`numpy.ndarray`):
            Array of :math:`\beta` values in radians.
        gamma ((...) :class:`numpy.ndarray`):
            Array of :math:`\gamma` values in radians.
        convention (str):
            One of the 12 valid conventions xzx, xyx, yxy, yzy, zyz, zxz, xzy, xyz, yxz,
            yzx, zyx, zxy.
        axis_type (str):
            Whether to use extrinsic or intrinsic rotations.

    Returns:
        (..., 4) :class:`numpy.ndarray`: Quaternions corresponding to the input angles.

    Example::

        >>> rowan.from_euler(0.3, 0.5, 0.7)
        array([0.91262714, 0.29377717, 0.27944389, 0.05213241])
    """
    angles = np.broadcast_arrays(alpha, beta, gamma)

    convention = convention.lower()

    if len(convention) > 3 or (set(convention) - set("xyz")):
        raise ValueError(
            "All acceptable conventions must be 3 \
character strings composed only of x, y, and z",
        )

    basis_axes = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }
    # Temporary method to ensure shapes conform
    for ax, vec in basis_axes.items():
        basis_axes[ax] = np.broadcast_to(vec, angles[0].shape + (vec.shape[-1],))

    # Split by convention, the easiest
    rotations = []
    if axis_type == "extrinsic":
        # Loop over the axes and add each rotation
        for i, char in enumerate(convention):
            ax = basis_axes[char]
            rotations.append(from_axis_angle(ax, angles[i]))
    elif axis_type == "intrinsic":
        for i, char in enumerate(convention):
            ax = basis_axes[char]
            print(char, ax, angles[i])
            rotations.append(rowan.from_axis_angle(ax, angles[i]))
            # Rotate the bases as well
            for key, value in basis_axes.items():
                print("rotate ", key, value)
                basis_axes[key] = rowan.rotate(rotations[-1], value)
    else:
        raise ValueError("Only valid axis_types are intrinsic and extrinsic")
    
    print(rotations)

    # Compose the total rotation
    final_rotation = np.broadcast_to(np.array([1, 0, 0, 0]), rotations[0].shape)
    print(final_rotation)
    for q in rotations:
        final_rotation = rowan.multiply(q, final_rotation)

    return final_rotation

# from_euler(0.1, 0.2, 0.3)

# static inline struct vec quat2rpy(struct quat q) {
# 	// from https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
# 	struct vec v;
# 	v.x = atan2f(2.0f * (q.w * q.x + q.y * q.z), 1 - 2 * (fsqr(q.x) + fsqr(q.y))); // roll
# 	v.y = asinf(2.0f * (q.w * q.y - q.x * q.z)); // pitch
# 	v.z = atan2f(2.0f * (q.w * q.z + q.x * q.y), 1 - 2 * (fsqr(q.y) + fsqr(q.z))); // yaw
# 	return v;
# }

def quat2rpy(q):
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    return np.array([
        np.arctan2(2.0 * (qw * qx + qy * qz), 1 - 2 * (qx*qx + qy*qy)),
        np.arcsin(2.0 * (qw * qy - qx * qz)),
        np.arctan2(2.0 * (qw * qz + qx * qy), 1 - 2 * (qy*qy + qz*qz)),
    ])

import rowan
q = rowan.random.random_sample()
v = quat2rpy(q)
print(v)
p = rowan.from_euler(*v, "xyz", "extrinsic")
print(q, p)
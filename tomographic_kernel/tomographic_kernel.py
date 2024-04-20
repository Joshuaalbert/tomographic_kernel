from logging import getLogger
from typing import NamedTuple, Tuple, Optional, List, Union

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import vmap, tree_map
from jax.lax import scan

logger = getLogger(__name__)

TEC_CONV = -8.4479745e6  # Hz/mTECU


def broadcast_leading_dim(*arrays):
    leading_dim = None
    for array in arrays:
        if len(array.shape) == 2:
            leading_dim = array.shape[0]
            break
    if leading_dim is None:
        return arrays
    _arrays = []
    for array in arrays:
        if len(array.shape) == 2:
            _arrays.append(array)
            continue
        _arrays.append(jnp.reshape(array, (leading_dim, array.shape[-1])))
    return tuple(_arrays)


def scan_vmap(f):
    """
    Applies a function `f` over a pytree, over the first dimension of each leaf.
    Similar to vmap, but sequentially executes, rather that broadcasting.
    """

    def run(*args):
        def body(state, X):
            return state, f(*X)

        _, results = scan(body, (), args)
        return results

    return run


class GeodesicTuple(NamedTuple):
    x: jnp.ndarray
    k: jnp.ndarray
    t: jnp.ndarray
    ref_x: jnp.ndarray


class TomographicKernel:
    def cov_func(self, X1: GeodesicTuple, X2: Optional[GeodesicTuple] = None) -> jnp.ndarray:
        """
        Compute covariance matrix

        Args:
            X1: GeodesicTuple
            X2: GeodesicTuple or None

        Returns:
            matrix of covariance func
        """
        raise NotImplementedError()

    def mean_func(self, X1: GeodesicTuple) -> jnp.ndarray:
        """
        Compute the mean.

        Args:
            X1:

        Returns:
            array of means for geodesics
        """
        raise NotImplementedError()


class MultiLayerTomographicKernel(TomographicKernel):
    def __init__(self, x0: jnp.ndarray, earth_centre: jnp.ndarray,
                 bottom: Union[List[jnp.ndarray], jnp.ndarray],
                 width: Union[List[jnp.ndarray], jnp.ndarray],
                 fed_kernel: Union[tfp.math.psd_kernels.PositiveSemidefiniteKernel, List[
                     tfp.math.psd_kernels.PositiveSemidefiniteKernel]],
                 fed_mu: Union[List[jnp.ndarray], jnp.ndarray],
                 wind_velocity: Optional[Union[List[jnp.ndarray], jnp.ndarray]] = None,
                 S_marg: int = 25, compute_tec: bool = False):
        """
        Tomographic model with curved thick layer above the Earth.

        Args:
            x0: [3] jnp.ndarray The array center. We assume a spherical Earth and that height is referenced radially
                from the shell that x0 sits on.
            earth_centre: Earth's centre
            fed_kernel: StationaryKernel, covariance function of the free-electron density.
            fed_mu: variation scaling in mTECU/km, or 10^10 electron/m^3
            wind_velocity: optional frozen flow velocity (tangent above x0 and bottom of layer)
            S_marg: int, the resolution of quadrature
            compute_tec: bool, whether to compute TEC or DTEC.
        """
        self.S_marg = S_marg
        self.x0 = x0
        self.earth_centre = earth_centre
        self.fed_kernel = fed_kernel
        self.compute_tec = compute_tec
        self.bottom, self.width, self.wind_velocity, self.fed_kernel, self.fed_mu = self.broadcast_lists(
            bottom, width, wind_velocity, fed_kernel, fed_mu
        )

        self.cov_func = self.build_cov_func()
        self.mean_func = self.build_mean_func()

    def broadcast_lists(self, *objs):
        objs = list(map(lambda obj: obj if isinstance(obj, list) else [obj], objs))
        max_len = max(map(len, objs))
        objs = tuple(map(lambda obj: obj * max_len if len(obj) != max_len else obj, objs))
        return objs

    def compute_integration_limits_flat(self, x: jnp.ndarray, k: jnp.ndarray, bottom: jnp.ndarray,
                                        width: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the integration limits of the flat layer ionosphere.

        Args:
            x: [3] or [N, 3]
            k: [3] or [N, 3]
            bottom: scalar height of bottom of layer
            width: scalar width of layer

        Returns:
            s_min, s_max with shapes:
                - scalars if x and k are [3]
                - arrays of [N] if x or k is [N,3]
        """
        if (len(x.shape) == 2) or (len(k.shape) == 2):
            x, k = jnp.broadcast_arrays(x, k)
            return vmap(lambda x, k: self.compute_integration_limits_flat(x, k, bottom, width))(x, k)
        smin = (bottom - (x[2] - self.x0[2])) / k[2]
        smax = (bottom + width - (x[2] - self.x0[2])) / k[2]
        return smin, smax

    def compute_integration_limits(self, x: jnp.ndarray, k: jnp.ndarray, bottom: jnp.ndarray,
                                   width: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the integration limits of the curved layer ionosphere.

        Args:
            x: [3] or [N, 3]
            k: [3] or [N, 3]
            bottom: scalar height of bottom of layer
            width: scalar width of width of layer

        Returns:
            s_min, s_max with shapes:
                - scalars if x and k are [3]
                - arrays of [N] if x or k is [N,3]
        """
        if (len(x.shape) == 2) or (len(k.shape) == 2):
            x, k = jnp.broadcast_arrays(x, k)
            return vmap(lambda x, k: self.compute_integration_limits(x, k, bottom, width))(x, k)

        dx = x - self.earth_centre
        dxk = dx @ k
        dx2 = dx @ dx
        bottom_radius2 = jnp.sum(jnp.square(self.x0 + jnp.asarray([0., 0., bottom]) - self.earth_centre))
        top_radius2 = jnp.sum(jnp.square(self.x0 + jnp.asarray([0., 0., bottom + width]) - self.earth_centre))
        smin = -dxk + jnp.sqrt(dxk ** 2 + (bottom_radius2 - dx2))
        smax = -dxk + jnp.sqrt(dxk ** 2 + (top_radius2 - dx2))
        return smin, smax

    def build_cov_func(self):
        """
        Construct a callable that returns the tomographic kernel function.

        Returns:
            callable(x1:[N,3],k1:[N,3],x2:[M,3],k2:[M,3]) -> [N, M]
        """

        def ray_integral(f):
            """
            Integrates a function of a scalar variable over [0,1].

            Args:
                f: callable of scalar to integrate

            Returns:
                integral over [0,1]
            """
            t = jnp.linspace(0., 1., self.S_marg + 1)
            return jnp.sum(vmap(f)(t), axis=0) * (1. / self.S_marg)

        def build_geodesic(x, k, t, bottom, width, wind_velocity):
            """
            Convert observation coordinates to a geodesic callable.

            Args:
                x: observe coordinates
                k: observed looking direction
                t: time of observer

            Returns:
                callable from [0,1] to a point in space, and length of segment intersection with shell
            """
            smin, smax = self.compute_integration_limits(x, k, bottom, width)

            def g(epsilon):
                y = x + k * (smin + (smax - smin) * epsilon)
                return frozen_flow_transform(t, y, x0=self.x0, bottom=bottom,
                                             earth_centre=self.earth_centre,
                                             wind_velocity=wind_velocity)

            return g, (smax - smin)

        def integrate_integrand(X1: GeodesicTuple, X2: GeodesicTuple) -> jnp.ndarray:
            """
            Computes the integral of tomographic kernel for a single pair of coordinates

            Args:
                X1: single geodesic coordinate
                X2: single geodesic coordinate

            Returns:
                scalar
            """

            def create_integrand_single_layer(bottom, width, wind_velocity, fed_kernel):
                g1, ds1 = build_geodesic(X1.x, X1.k, X1.t, bottom=bottom, width=width, wind_velocity=wind_velocity)
                g2, ds2 = build_geodesic(X2.x, X2.k, X2.t, bottom=bottom, width=width, wind_velocity=wind_velocity)
                if not self.compute_tec:
                    g1_ref, ds1_ref = build_geodesic(X1.ref_x, X1.k, X1.t, bottom=bottom, width=width,
                                                     wind_velocity=wind_velocity)
                    g2_ref, ds2_ref = build_geodesic(X2.ref_x, X2.k, X2.t, bottom=bottom, width=width,
                                                     wind_velocity=wind_velocity)

                # print(f"Mid-point separation: {jnp.linalg.norm(g1(0.5) - g2(0.5))}")

                def f(epsilon_1, epsilon_2):
                    results = (ds1 * ds2) * fed_kernel.matrix(g1(epsilon_1)[None], g2(epsilon_2)[None])
                    if not self.compute_tec:
                        results += (ds1_ref * ds2_ref) * fed_kernel.matrix(g1_ref(epsilon_1)[None],
                                                                           g2_ref(epsilon_2)[None])
                        results -= (ds1 * ds2_ref) * fed_kernel.matrix(g1(epsilon_1)[None],
                                                                       g2_ref(epsilon_2)[None])
                        results -= (ds1_ref * ds2) * fed_kernel.matrix(g1_ref(epsilon_1)[None],
                                                                       g2(epsilon_2)[None])
                    return results[0, 0]

                return f

            def complete_integrand(epsilon_1, epsilon_2):
                output = []
                for bottom, width, wind_velocity, fed_kernel in zip(self.bottom, self.width, self.wind_velocity,
                                                                    self.fed_kernel):
                    f = create_integrand_single_layer(bottom, width, wind_velocity, fed_kernel)
                    output.append(f(epsilon_1, epsilon_2))
                return sum(output[1:], output[0])

            return ray_integral(
                lambda epsilon_2: ray_integral(lambda epsilon_1: complete_integrand(epsilon_1, epsilon_2)))

        def Kxy(X1: GeodesicTuple, X2: GeodesicTuple) -> jnp.ndarray:
            """
            Computes the covariance function for TEC or differential TEC.

            Coordinates are in ENU frame situated at the array reference antenna.

            Args:
                X1: GeodesicTuple, batch N
                X2: GeodesicTuple, batch M

            Returns:
                matrix of [N, M]
            """
            if X1.x.shape[0] == 1:
                K = vmap(lambda X2: integrate_integrand(tree_map(lambda x: x[0], X1), X2))(X2)
                return K[None]
            else:
                return scan_vmap(lambda X1: vmap(lambda X2: integrate_integrand(X1, X2))(X2))(X1)

        def cov_func(X1: GeodesicTuple, X2: Optional[GeodesicTuple] = None):
            """

            Args:
                X1: GeodesicTuple
                X2: GeodesicTuple or None

            Returns:
                matrix of covariance func
            """
            X1 = GeodesicTuple(*broadcast_leading_dim(*X1))
            if X2 is None:
                X2 = X1
            else:
                X2 = GeodesicTuple(*broadcast_leading_dim(*X2))
            return Kxy(X1, X2)

        return cov_func

    def build_mean_func(self):
        """
        Computes the intersection with ionosphere, and multiplies by constant FED mean.
        This depends on the geometry of the ionosphere.

        Args:
            bottom: bottom of ionosphere in km
            width: width of ionosphere in km
            fed_mu: variation scaling in mTECU/km, or 10^10 electron/m^3
            wind_velocity: in East-North-Up frame

        Returns:
            callable(x:[N,3],k:[N,3],t:[N]) -> [N]
        """

        def geodesic_intersection(x, k, t, bottom, width):
            smin, smax = self.compute_integration_limits(x, k, bottom, width)
            return (smax - smin)

        def single_layer_intersection(X1: GeodesicTuple, bottom, width, fed_mu):
            ds1 = geodesic_intersection(X1.x, X1.k, X1.t, bottom=bottom, width=width)
            intersection = ds1
            if not self.compute_tec:
                ds1_ref = geodesic_intersection(X1.ref_x, X1.k, X1.t, bottom=bottom, width=width)
                intersection - ds1_ref
            return fed_mu * intersection

        def complete_intersection(X1: GeodesicTuple):
            output = []
            for bottom, width, fed_mu in zip(self.bottom, self.width, self.fed_mu):
                output.append(single_layer_intersection(X1, bottom=bottom, width=width, fed_mu=fed_mu))
            return sum(output[1:], output[0])

        def mean_func(X1: GeodesicTuple):
            """
            Compute the mean of the line of sign integral.

            Args:
                X1: geodesic

            Returns:
                mean along the geodesic
            """
            X1 = GeodesicTuple(*broadcast_leading_dim(*X1))
            return vmap(complete_intersection)(X1)

        return mean_func


def rotate_vector(y, rotation_axis, angle):
    u_cross_x = jnp.cross(rotation_axis, y)
    rotated_y = rotation_axis * (rotation_axis @ y) \
                + jnp.cos(angle) * jnp.cross(u_cross_x, rotation_axis) \
                + jnp.sin(angle) * u_cross_x
    return rotated_y


def frozen_flow_transform(t, y, x0, bottom, earth_centre, wind_velocity=None) -> jnp.ndarray:
    """
    Computes the frozen flow transform on the coordinates.

    Args:
        t: time in seconds
        y: position in km, origin at centre of Earth
        x0: position of reference point.
        bottom: bottom of ionosphere in km
        earth_centre: coordinate of earth centre in same frame as y
        wind_velocity: layer velocity in km/s

    Returns:
        Coordinates inverse rotating the coordinates to take into account the flow of ionosphere around
        surface of Earth.
    """
    # rotate around Earth's core
    if t is None:
        return y
    if wind_velocity is None:
        return y

    rotation_axis = jnp.cross(jnp.asarray([0., 0., 1.]), wind_velocity)
    rotation_axis /= jnp.maximum(jnp.linalg.norm(rotation_axis), 1e-6)

    radius = jnp.linalg.norm(x0 + jnp.asarray([0., 0., bottom]) - earth_centre)
    # v(r) = theta_dot * r
    # km/s / km
    theta_dot = jnp.linalg.norm(wind_velocity) / radius
    # We negate to undo the motion of the wind
    angle = -theta_dot * t
    # logger.info(f'radius {radius} km velocity {wind_velocity} km/s theta_dot {theta_dot} rad / sec, angle {angle*180/np.pi} deg')
    # Rotation
    rotated_y = rotate_vector(y - earth_centre, rotation_axis, angle) + earth_centre
    # logger.info(f"{t} {jnp.linalg.norm(rotated_y-y)}")
    # assert jnp.linalg.norm.(y-rotated_y) < 1e-6
    return rotated_y


def mean_square_difference(X1: GeodesicTuple, X2: GeodesicTuple, kernel: MultiLayerTomographicKernel):
    """
    Compute the mean square difference of the underlying between two points, i.e.

        E[(DTEC(X1) - DTEC(X2))^2] if the kernel is a DTEC kernel.

    or

        E[(TEC(X1) - TEC(X2))^2] if the kernel is a TEC kernel.

    Args:
        X1: the first point (GeodesicTuple) of shape [M]
        X2: the second point (GeodesicTuple) of shape [N]
        kernel: the kernel

    Returns:
        the mean square difference of the underlying between X1 and X2 [M, N]
    """
    var_1 = jnp.diag(kernel.cov_func(X1, X1))  # [M]
    var_2 = jnp.diag(kernel.cov_func(X2, X2))  # [N]
    mu_1 = kernel.mean_func(X1)  # [M]
    mu_2 = kernel.mean_func(X2)  # [N]
    cov_12 = kernel.cov_func(X1, X2)  # [M, N]
    return (
            var_1[:, None] + mu_1[:, None] ** 2
            - 2. * (cov_12 + mu_1[:, None] * mu_2[None, :])
            + var_2[None, :] + mu_2[None, :] ** 2
    )

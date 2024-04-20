from typing import Tuple

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax.numpy as jnp
import pylab as plt
from jax import vmap

from tomographic_kernel.frames import ENU
from tomographic_kernel.models.cannonical_models import build_ionosphere_tomographic_kernel, SPECIFICATION
from tomographic_kernel.tomographic_kernel import GeodesicTuple, mean_square_difference


def test_direction_scale():
    obstime = at.Time("2019-03-19T19:58:14.9", format='isot')
    array_centre = ac.ITRS(x=6700 * au.m, y=0 * au.m, z=0 * au.m, obstime=obstime)
    enu_frame = ENU(obstime=obstime, location=array_centre.earth_location)
    direction = ac.SkyCoord(east=0. * au.m, north=0. * au.m, up=1 * au.m, frame=enu_frame)
    northern_hemisphere = direction.transform_to(ac.ICRS()).dec.deg > 0
    earth_centre = ac.ITRS(x=0 * au.m, y=0 * au.m, z=0 * au.m, obstime=obstime).transform_to(enu_frame)

    plt.figure(figsize=(10, 10))

    specification = 'light_dawn'

    for b in [1., 2., 5., 10., 20.]:
        x0 = ac.SkyCoord(east=0. * au.m, north=0. * au.m, up=0 * au.m, frame=enu_frame)
        x1 = ac.SkyCoord(east=0. * au.m, north=b * 1000. * au.m, up=0 * au.m, frame=enu_frame)

        t2, msd_dtec = compute_dtec_time_variance(
            direction=direction,
            earth_centre=earth_centre,
            northern_hemisphere=northern_hemisphere,
            x0=x0,
            x1=x1,
            specification=specification
        )
        plt.plot(t2, msd_dtec.flatten(), label=fr'$\mathbf{{b}}=$ {b}km N')

        x0 = ac.SkyCoord(east=0. * au.m, north=0. * au.m, up=0 * au.m, frame=enu_frame)
        x1 = ac.SkyCoord(east=b * 1000. * au.m, north=0. * au.m, up=0 * au.m, frame=enu_frame)

        t2, msd_dtec = compute_dtec_time_variance(
            direction=direction,
            earth_centre=earth_centre,
            northern_hemisphere=northern_hemisphere,
            x0=x0,
            x1=x1,
            specification=specification
        )
        plt.plot(t2, msd_dtec.flatten(), label=fr'$\mathbf{{b}}=$ {b}km E')

    plt.title(
        r"Expectation of $\tau^2(\mathbf{b}, k, k') \triangleq (\Delta \mathrm{TEC}(\mathbf{b}, k) - \Delta \mathrm{TEC}(\mathbf{b}, k'))^2$")

    plt.xlabel(r"$\Delta \mathrm{RA}$ [deg]")
    plt.ylabel(r"$\mathbb{E}[\tau^2(\mathbf{b}, k, k')]$ [mTECU sq.]")
    # threshold
    tau_threshold = 20. * 700 * 0.002
    plt.axhline(tau_threshold ** 2, color='r', linestyle='--', label='20 deg phase error @ 700MHz')
    # put legend on right side
    plt.legend(loc='center right')
    plt.show()


def single_rotation(phi, theta, k):
    # Rotation matrix for y-axis
    Ry = jnp.array([
        [jnp.cos(theta), 0, jnp.sin(theta)],
        [0, 1, 0],
        [-jnp.sin(theta), 0, jnp.cos(theta)]
    ])

    # Rotation matrix for z-axis
    Rz = jnp.array([
        [jnp.cos(phi), -jnp.sin(phi), 0],
        [jnp.sin(phi), jnp.cos(phi), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = jnp.dot(Rz, Ry)

    # Applying rotation to vector k
    return jnp.dot(R, k)

def compute_dtec_time_variance(direction: ac.SkyCoord, earth_centre: ac.SkyCoord, northern_hemisphere: bool,
                               x0: ac.SkyCoord, x1: ac.SkyCoord, specification: SPECIFICATION) -> Tuple[
    jnp.ndarray, jnp.ndarray]:
    """
    Compute the dynamic timescale of the ionosphere between two points in time.

    E[(DTEC(x1, x0, t) - DTEC(x1, x0, t'))^2]

    Args:
        direction: the direction
        earth_centre: the centre of Earth
        northern_hemisphere: whether in northern hemisphere
        x0: reference antenna
        x1: other antenna
        specification: the specification of the ionosphere

    Returns:
        the time array and E[(DTEC(x1, x0, t) - DTEC(x1, x0, t'))^2]
    """
    x0_enu_xyz = x0.cartesian.xyz.to(au.km).value
    x1_enu_xyz = x1.cartesian.xyz.to(au.km).value
    k0_enu_xyz = direction.cartesian.xyz.value
    earth_centre_enu_xyz = earth_centre.cartesian.xyz.to(au.km).value
    kernel = build_ionosphere_tomographic_kernel(x0=x0_enu_xyz,
                                                 earth_centre=earth_centre_enu_xyz,
                                                 specification=specification,
                                                 S_marg=25,
                                                 compute_tec=False,
                                                 northern_hemisphere=northern_hemisphere)
    N = 30
    t1 = jnp.zeros((1,))

    # Vectorizing the single_rotation function
    batched_rotation = vmap(single_rotation, in_axes=(0, 0, None), out_axes=0)

    theta = jnp.linspace(0., 5*jnp.pi/180., N)
    phi = jnp.zeros_like(theta)

    k2_enu_xyz = batched_rotation(phi, theta, k0_enu_xyz)
    # ensure arcosine of dot product is similar to theta
    assert jnp.allclose(theta, jnp.arccos(jnp.sum(k2_enu_xyz * k0_enu_xyz, axis=-1)), atol=1e-5)

    X1 = GeodesicTuple(x=x1_enu_xyz[None, :],
                       k=k0_enu_xyz[None, :],
                       t=t1[:, None],
                       ref_x=x0_enu_xyz[None, :])
    X2 = GeodesicTuple(x=jnp.tile(x1_enu_xyz[None, :], (N, 1)),
                       k=k2_enu_xyz[None, :],
                       t=jnp.tile(t1[:, None], (N, 1)),
                       ref_x=jnp.tile(x0_enu_xyz[None, :], (N, 1)))
    msd_dtec = mean_square_difference(X1, X2, kernel)
    return theta * 180 / jnp.pi, msd_dtec

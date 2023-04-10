import logging
from typing import TypeVar

import astropy.coordinates as ac
import astropy.units as au
import numpy as np
from h5parm import DataPack
from jax import numpy as jnp, tree_map, pmap
from jax._src.lax.control_flow import scan

from tomographic_kernel.frames import ENU

logger = logging.getLogger(__name__)
dist_type = au.km
angle_type = au.rad


def voronoi_finite_polygons_2d(vor, radius):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    # if radius is None:
    #     radius = np.max(np.linalg.norm(points - np.mean(points, axis=0),axis=1))
    #     # radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def get_coordinates(datapack: DataPack, ref_ant=0, ref_dir=0):
    tmp_selection = datapack._selection
    dummy_soltab = datapack.soltabs[0].replace('000', '')
    datapack.select(ant=ref_ant, dir=ref_dir)
    axes = datapack.__getattr__("axes_{}".format(dummy_soltab))
    _, _ref_ant = datapack.get_antennas(axes['ant'])
    _, _ref_dir = datapack.get_directions(axes['dir'])
    datapack.select(**tmp_selection)
    axes = datapack.__getattr__("axes_{}".format(dummy_soltab))
    _, _antennas = datapack.get_antennas(axes['ant'])
    _, _directions = datapack.get_directions(axes['dir'])
    _, times = datapack.get_times(axes['time'])
    Nt = len(times)

    X_out = []
    ref_ant_out = []
    ref_dir_out = []
    for t in range(Nt):
        obstime = times[t]
        ref_location = ac.ITRS(x=_ref_ant.x, y=_ref_ant.y, z=_ref_ant.z)
        ref_ant = ac.ITRS(x=_ref_ant.x, y=_ref_ant.y, z=_ref_ant.z, obstime=obstime)
        ref_dir = ac.ICRS(ra=_ref_dir.ra, dec=_ref_dir.dec)
        enu = ENU(location=ref_location, obstime=obstime)
        ref_ant = ref_ant.transform_to(enu).cartesian.xyz.to(dist_type).value.T
        ref_dir = ref_dir.transform_to(enu).cartesian.xyz.value.T
        antennas = ac.ITRS(x=_antennas.x, y=_antennas.y, z=_antennas.z, obstime=obstime)
        antennas = antennas.transform_to(enu).cartesian.xyz.to(dist_type).value.T
        directions = ac.ICRS(ra=_directions.ra, dec=_directions.dec)
        directions = directions.transform_to(enu).cartesian.xyz.value.T
        X_out.append(make_coord_array(directions, antennas, flat=False))
        ref_ant_out.append(ref_ant)
        ref_dir_out.append(ref_dir)
    # Nt, Nd, Na, 6
    X = np.stack(X_out, axis=0)
    ref_ant = np.concatenate(ref_ant_out, axis=0)
    ref_dir = np.concatenate(ref_dir_out, axis=0)
    return X, ref_ant, ref_dir


def great_circle_sep(ra1, dec1, ra2, dec2):
    """
    Seperation on S1
    Args:
        ra1: radians
        dec1: radians
        ra2: radians
        dec2: radians

    Returns: radians
    """
    dra = jnp.abs(ra1 - ra2)
    # ddec = np.abs(dec1-dec2)
    num2 = (jnp.cos(dec2) * jnp.sin(dra)) ** 2 + (
            jnp.cos(dec1) * jnp.sin(dec2) - jnp.sin(dec1) * jnp.cos(dec2) * jnp.cos(dra)) ** 2
    den = jnp.sin(dec1) * jnp.sin(dec2) + jnp.cos(dec1) * jnp.cos(dec2) * jnp.cos(dra)
    return jnp.arctan2(jnp.sqrt(num2), den)


def make_coord_array(*X, flat=True, coord_map=None):
    """
    Create the design matrix from a list of coordinates
    :param X: list of length p of float, array [Ni, D]
        Ni can be different for each coordinate array, but D must be the same.
    :param flat: bool
        Whether to return a flattened representation
    :param coord_map: callable(coordinates), optional
            If not None then get mapped over the coordinates
    :return: float, array [N0,...,Np, D] if flat=False else [N0*...*Np, D]
        The coordinate design matrix
    """

    if coord_map is not None:
        X = [coord_map(x) for x in X]

    def add_dims(x, where, sizes):
        shape = []
        tiles = []
        for i in range(len(sizes)):
            if i not in where:
                shape.append(1)
                tiles.append(sizes[i])
            else:
                shape.append(-1)
                tiles.append(1)
        return jnp.tile(jnp.reshape(x, shape), tiles)

    N = [x.shape[0] for x in X]
    X_ = []

    for i, x in enumerate(X):
        for dim in range(x.shape[1]):
            X_.append(add_dims(x[:, dim], [i], N))
    X = jnp.stack(X_, axis=-1)
    if not flat:
        return X
    return jnp.reshape(X, (-1, X.shape[-1]))


def polyfit(x, y, deg):
    """
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting polynomial
    Returns
    -------
    p : ndarray, shape (deg + 1,) or (deg + 1, K)
        Polynomial coefficients, highest power first.  If `y` was 2-D, the
        coefficients for `k`-th data set are in ``p[:,k]``.
    """
    order = int(deg) + 1
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")
    rcond = len(x) * jnp.finfo(x.dtype).eps
    lhs = jnp.stack([x ** (deg - i) for i in range(order)], axis=1)
    rhs = y
    scale = jnp.sqrt(jnp.sum(lhs * lhs, axis=0))
    lhs /= scale
    c, resids, rank, s = jnp.linalg.lstsq(lhs, rhs, rcond)
    c = (c.T / scale).T  # broadcast scale coefficients
    return c


def poly_smooth(x, y, deg=5, weights=None):
    """
    Smooth y(x) with a `deg` degree polynomial in x
    Args:
        x: [N]
        y: [N]
        deg: int

    Returns: smoothed y [N]
    """
    if weights is None:
        coeffs = polyfit(x, y, deg=deg)
    else:
        coeffs = weighted_polyfit(x, y, deg=deg, weights=weights)
    return sum([p * x ** (deg - i) for i, p in enumerate(coeffs)])


def weighted_polyfit(x, y, deg, weights):
    order = int(deg) + 1
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")
    rcond = len(x) * jnp.finfo(x.dtype).eps
    X = jnp.stack([x ** (deg - i) for i in range(order)], axis=1)
    scale = jnp.sqrt(jnp.sum(X * X, axis=0))
    X /= scale
    c, resids, rank, s = jnp.linalg.lstsq((X.T * weights) @ X, (X.T * weights) @ y, rcond)
    c = (c.T / scale).T  # broadcast scale coefficients
    return c


def test_weighted_polyfit():
    def f(x):
        return 1. + x + x ** 2 + x ** 3

    x = jnp.linspace(0., 1., 3)
    assert jnp.allclose(polyfit(x, f(x), 4), weighted_polyfit(x, f(x), 4, jnp.ones_like(x)), atol=1e-3)


def wrap(phi):
    return (phi + jnp.pi) % (2 * jnp.pi) - jnp.pi


F = TypeVar('F')


def chunked_pmap(f: F, chunksize, *, batch_size=None) -> F:
    def _f(*args, batch_size=batch_size, **kwargs):
        def queue(*args, **kwargs):
            """
            Distributes the computation in queues which are computed with scan.
            Args:
                *args:
            """

            def body(state, X):
                (args, kwargs) = X
                return state, f(*args, **kwargs)

            _, result = scan(body, (), (args, kwargs))
            return result

        if chunksize > 1:
            if batch_size is None:
                batch_size = args[0].shape[0] if len(args) > 0 else None
            assert batch_size is not None, "Couldn't get batch_size, please provide explicitly"
            remainder = batch_size % chunksize
            extra = (chunksize - remainder) % chunksize
            args = tree_map(lambda arg: _pad_extra(arg, chunksize), args)
            kwargs = tree_map(lambda arg: _pad_extra(arg, chunksize), kwargs)
            result = pmap(queue)(*args, **kwargs)
            result = tree_map(lambda arg: jnp.reshape(arg, (-1,) + arg.shape[2:]), result)
            if extra > 0:
                result = tree_map(lambda x: x[:-extra], result)
        else:
            result = queue(*args, **kwargs)
        return result

    _f.__doc__ = f.__doc__
    _f.__annotations__ = f.__annotations__
    return _f


def _pad_extra(arg, chunksize):
    N = arg.shape[0]
    remainder = N % chunksize
    if (remainder != 0) and (N > chunksize):
        # only pad if not a zero remainder
        extra = (chunksize - remainder) % chunksize
        arg = jnp.concatenate([arg] + [arg[0:1]] * extra, axis=0)
        N = N + extra
    else:
        extra = 0
    T = N // chunksize
    arg = jnp.reshape(arg, (chunksize, N // chunksize) + arg.shape[1:])
    return arg

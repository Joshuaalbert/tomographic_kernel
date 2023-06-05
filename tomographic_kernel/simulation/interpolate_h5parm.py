import logging

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
from h5parm import DataPack
from scipy.spatial import KDTree

from tomographic_kernel.tomographic_kernel import TEC_CONV
from tomographic_kernel.utils import great_circle_sep, make_coord_array, wrap

logger = logging.getLogger(__name__)


def haversine_distance(lon1, lat1, lon2, lat2):
    return np.abs(np.arctan2(np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1),
                             np.sin(lon2 - lon1) * np.cos(lat2)))


def calculate_midpoint(lon1, lat1, lon2, lat2):
    dLon = lon2 - lon1
    Bx = np.cos(lat2) * np.cos(dLon)
    By = np.cos(lat2) * np.sin(dLon)
    lon_mid = np.arctan2(np.sin(lon1) + np.sin(lon2), np.sqrt((np.cos(lat1) + Bx) ** 2 + By ** 2))

    # Find the midpoint latitude
    lat_mid = np.arctan2(np.sin(lat1) + np.sin(lat2), np.sqrt((np.cos(lat1) + Bx) ** 2 + By ** 2))
    return np.asarray([lon_mid, lat_mid])


def compute_mean_coordinates(coords_rad):
    # Convert coordinates to radians
    # Convert to unit vectors
    x = np.cos(coords_rad[:, 0]) * np.cos(coords_rad[:, 1])
    y = np.sin(coords_rad[:, 0]) * np.cos(coords_rad[:, 1])
    z = np.sin(coords_rad[:, 1])

    # Compute the mean of unit vectors
    mean_vector = np.asarray([np.mean(x), np.mean(y), np.mean(z)])

    # Convert back to longitude and latitude in radians
    mean_lon = np.arctan2(mean_vector[1], mean_vector[0])
    mean_lat = np.arcsin(mean_vector[2])

    # Convert mean coordinates back to degrees
    mean_coord = np.asarray([mean_lon, mean_lat])

    dist = list(map(lambda coord: haversine_distance(*mean_coord, *coord), coords_rad))
    idx1 = np.argmax(dist)

    dist = list(map(lambda coord: haversine_distance(*coords_rad[idx1], *coord), coords_rad))
    idx2 = np.argmax(dist)

    return calculate_midpoint(*coords_rad[idx1], *coords_rad[idx2])


def test_compute_mean_coordinates():
    # Example coordinates
    coordinates = np.array([[0, 0],
                            [0, np.pi / 2]])

    # Compute the mean coordinates
    mean_coords = compute_mean_coordinates(coordinates)

    assert np.allclose(mean_coords, np.asarray([0., np.pi / 4]), atol=1e-4)


def map_spherical_to_plane(directions):
    lon_mean, lat_mean = compute_mean_coordinates(directions)
    dx = lon_mean - directions[:, 0]
    lon_prime = np.cos(directions[:, 1]) * np.sin(dx)
    lat_prime = lat_mean - directions[:, 1]
    return np.stack([lon_prime, lat_prime], axis=1)


def interpolation(dtec_in, antennas_in: ac.ITRS, directions_in: ac.ICRS, times_in: at.Time,
                  antennas_out: ac.ITRS, directions_out: ac.ICRS, times_out: at.Time,
                  k=3):
    """
    Interpolate dtec_in onto out coordinates using linear interpolation.

    Args:
        dtec_in: [Nd, Na, Nt]
        antennas_in: [Na]
        directions_in: [Nd]
        times_in: [Nt]
        antennas_out: [Na']
        directions_out: [Nd']
        times_out: [Nt']

    Returns:
        dtec_out [Nd', Na', Nt']
    """
    logger.info("Interpolating...")
    antennas_in = antennas_in.cartesian.xyz.to(au.km).value.T
    antennas_out = antennas_out.cartesian.xyz.to(au.km).value.T
    directions_in = np.stack([directions_in.ra.rad, directions_in.dec.rad], axis=1)
    directions_out = np.stack([directions_out.ra.rad, directions_out.dec.rad], axis=1)
    times_in = times_in.mjd * 86400.
    times_in -= times_in[0]
    times_out = times_out.mjd * 86400.
    times_out -= times_out[0]

    from scipy.spatial.distance import cdist
    antennas_dist = cdist(antennas_in, antennas_out, metric='euclidean')
    directions_dist = cdist(directions_in, directions_out, metric=lambda k1, k2: great_circle_sep(*k1, *k2))
    times_dist = cdist(times_in, times_out, metric='euclidean')
    # antennas_first
    dtec_out = np.zeros((directions_in.shape[0], antennas_out.shape[0], times_in.shape[0]))
    closest_ants = np.argsort(antennas_dist, axis=0)
    for i, x_out in enumerate(antennas_out):
        dists = antennas_dist[closest_ants[:k, i], i]
        print(dists)
        kernel = np.exp(-0.5 * (dists / 1.) ** 2)
        kernel /= kernel.sum()
        dtec_out[:, i, :] = np.sum(dtec_in[:, closest_ants[:k, i], :] * kernel[None, :, None], axis=1)

    # directions_next
    dtec_in = dtec_out
    dtec_out = np.zeros((directions_out.shape[0], antennas_out.shape[0], times_in.shape[0]))
    closest_dirs = np.argsort(directions_dist, axis=0)
    for i, x_out in enumerate(directions_out):
        dists = directions_dist[closest_dirs[:k, i], i]
        print(dists)
        kernel = np.exp(-0.5 * (dists / 1.) ** 2)
        kernel /= kernel.sum()
        dtec_out[i, :, :] = np.sum(dtec_in[closest_dirs[:k, i], :, :] * kernel[:, None, None], axis=0)

    # times last
    dtec_in = dtec_out
    dtec_out = np.zeros((directions_out.shape[0], antennas_out.shape[0], times_out.shape[0]))
    closest_times = np.argsort(times_dist, axis=0)
    for i, x_out in enumerate(times_out):
        dists = directions_dist[closest_times[:k, i], i]
        print(dists)
        kernel = np.exp(-0.5 * (dists / 1.) ** 2)
        kernel /= kernel.sum()
        dtec_out[:, :, i] = np.sum(dtec_in[:, :, closest_times[:k, i]] * kernel[None, None, :], axis=2)

    return dtec_out


def make_coords(antennas: ac.ITRS, directions: ac.ICRS, times: at.Time):
    logger.info("Interpolating...")
    antennas = antennas.cartesian.xyz.to(au.km).value.T
    directions = np.stack([directions.ra.rad, directions.dec.rad], axis=1)
    directions = map_spherical_to_plane(directions=directions)
    times = times.mjd * 86400.
    times -= times[0]
    X = make_coord_array(directions, antennas, times[:, None])
    return X


def interpolate_h5parm(input_h5parm: str, output_h5parm: str, k: int = 3):
    """
    Interpolates a given h5parm onto another one, using simple interpolation which will fail to give good results
    when spacing is too sparse. Works by interpolating dtec.

    Args:
        input_h5parm: input h5parm with tec
        output_h5parm: output h5parm with space for tec, phase
    """
    with DataPack(input_h5parm, readonly=True) as dp:
        assert dp.axes_order == ['pol', 'dir', 'ant', 'freq', 'time']
        dp.current_solset = 'sol000'
        dp.select(pol=slice(0, 1, 1))
        tec_grid, axes = dp.tec
        tec_grid = tec_grid[0]  # remove pol
        tec_grid_flat = tec_grid.flatten()
        _, directions_grid = dp.get_directions(axes['dir'])
        _, antennas_grid = dp.get_antennas(axes['ant'])
        _, times_grid = dp.get_times(axes['time'])

    with DataPack(output_h5parm, readonly=True) as dp:
        assert dp.axes_order == ['pol', 'dir', 'ant', 'freq', 'time']
        dp.current_solset = 'sol000'
        dp.select(pol=slice(0, 1, 1))
        axes = dp.axes_phase
        _, freqs = dp.get_freqs(axes['freq'])
        _, directions = dp.get_directions(axes['dir'])
        _, antennas = dp.get_antennas(axes['ant'])
        _, times = dp.get_times(axes['time'])

    logger.info("Normalising coordinates")
    X_grid = make_coords(directions=directions_grid, antennas=antennas_grid, times=times_grid)
    X_grid -= np.mean(X_grid, axis=0, keepdims=True)
    X_grid /= np.std(X_grid, axis=0, keepdims=True) + 1e-3

    X = make_coords(directions=directions, antennas=antennas, times=times)
    X -= np.mean(X, axis=0, keepdims=True)
    X /= np.std(X, axis=0, keepdims=True) + 1e-3

    logger.info("Interpolating")
    # Step 2: Build the tree and find closest points
    tree = KDTree(X_grid)
    dist, ind = tree.query(X, k=k)  # Change k to find more than 1 nearest point
    if k == 1:  # nearest
        tec_flat = tec_grid_flat[ind]
    else:  # inverse distance weighted
        weights = 1.0 / dist  # [N, K]
        tec_flat = np.sum(weights * tec_grid_flat[ind], axis=1) / np.sum(weights, axis=1)
    tec = tec_flat.reshape((1, len(directions), len(antennas), len(times)))
    logger.info("Storing results.")
    with DataPack(output_h5parm, readonly=False) as dp:
        dp.current_solset = 'sol000'
        dp.select(pol=slice(0, 1, 1))
        dp.tec = tec
        dp.phase = np.asarray(wrap(tec[..., None, :] * (TEC_CONV / freqs.to('Hz').value[:, None])))

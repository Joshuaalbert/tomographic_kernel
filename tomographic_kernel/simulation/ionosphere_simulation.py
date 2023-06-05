import logging
from timeit import default_timer
from typing import Tuple

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import pylab as plt
from h5parm import DataPack
from h5parm.datapack import load_array_file, save_array_file
from h5parm.utils import directions_from_sky_model, create_empty_datapack
from jax import numpy as jnp, tree_map, jit, random, devices
from tqdm import tqdm

from tomographic_kernel.frames import ENU
from tomographic_kernel.models.cannonical_models import build_ionosphere_tomographic_kernel, SPECIFICATION
from tomographic_kernel.plotting import plot_vornoi_map
from tomographic_kernel.simulation.interpolate_h5parm import interpolate_h5parm
from tomographic_kernel.tomographic_kernel import GeodesicTuple, TEC_CONV
from tomographic_kernel.utils import make_coord_array, chunked_pmap, wrap

logger = logging.getLogger(__name__)

ARRAYS = {
    'lofar': DataPack.lofar_array_hba,
    'dsa2000W': './dsa2000.W.cfg',
    'dsa2000W10': './dsa2000.W.10.cfg',
    'dsa2000W_200m_grid': './dsa2000.W.200m_grid.cfg',
    'dsa2000W_300m_grid': './dsa2000.W.300m_grid.cfg',
    'dsa2000W_400m_grid': './dsa2000.W.400m_grid.cfg',
    'dsa2000W_500m_grid': './dsa2000.W.500m_grid.cfg',
    'dsa2000W_600m_grid': './dsa2000.W.600m_grid.cfg',
    'dsa2000W_700m_grid': './dsa2000.W.700m_grid.cfg',
    'dsa2000W_800m_grid': './dsa2000.W.800m_grid.cfg',
    'dsa2000W_900m_grid': './dsa2000.W.900m_grid.cfg',
    'dsa2000W_1000m_grid': './dsa2000.W.1000m_grid.cfg',
    'dsa2000W_1500m_grid': './dsa2000.W.1500m_grid.cfg',
    'dsa2000W_2000m_grid': './dsa2000.W.2000m_grid.cfg',
    'two_ovro_antennas': './two_ovro_antennas.cfg'
}


def get_num_directions(avg_spacing, field_of_view_diameter, min_n=1) -> int:
    """
    Get the number of directions that will space the field of view by the given spacing.

    Args:
        avg_spacing: average spacing in arc min
        field_of_view_diameter: field of view in degrees

    Returns:
        int, the number of directions to sample inside the S^2
    """
    V = 2. * np.pi * (field_of_view_diameter / 2.) ** 2
    pp = 0.5
    n = -V * np.log(1. - pp) / (avg_spacing / 60.) ** 2 / np.pi / 2.
    n = max(int(n), min_n)
    return n


def h5parm_to_np(h5parm: str) -> Tuple[np.ndarray, at.Time, ac.ITRS]:
    """
    Open an H5Parm and extract the phase, times, and antennas.

    Args:
        h5parm: 5parm filename

    Returns:
        phase (in radians), times (at.Time), and antennas (ac.ITRS)
    """
    with DataPack(h5parm, readonly=True) as dp:
        print("Axes order:", dp.axes_order)
        dp.current_solset = 'sol000'
        dp.select()
        phase, axes = dp.phase
        phase = phase[0]  # remove pol axis
        antenna_labels, antennas = dp.get_antennas(axes['ant'])
        timestamps, times = dp.get_times(axes['time'])

    return phase, times, antennas


def visualisation(h5parm, ant=None, time=None):
    """
    Makes a visualisation of the simulation results.

    Args:
        h5parm: h5parm file
        ant: which antenna to select (None means all)
        time: which time to select (None means all)

    Returns:

    """
    with DataPack(h5parm, readonly=True) as dp:
        dp.current_solset = 'sol000'
        dp.select(ant=ant, time=time, dir=0)
        print("Axes order:", dp.axes_order)
        dtec, axes = dp.tec
        phase, axes = dp.phase
        dtec = dtec[0]  # remove pol axis
        patch_names, directions = dp.get_directions(axes['dir'])
        antenna_labels, antennas = dp.get_antennas(axes['ant'])
        timestamps, times = dp.get_times(axes['time'])

    frame = ENU(obstime=times[0], location=antennas[0].earth_location)
    directions = directions.transform_to(frame)
    t = times.mjd * 86400.
    t -= t[0]
    dt = np.diff(t).mean()
    # x = antennas.cartesian.xyz.to(au.km).value.T  # [1:, :]
    x = ac.ITRS(*antennas.cartesian.xyz, obstime=times[0]).transform_to(frame).cartesian.xyz.to(au.km).value.T
    # x[1,:] = x[0,:]
    # x[1,0] += 0.3
    k = directions.cartesian.xyz.value.T
    logger.info(f"Directions: {directions}")
    logger.info(f"Antennas: {x} {antenna_labels}")
    logger.info(f"Times: {t}")
    Na = x.shape[0]
    logger.info(f"Number of antenna to plot: {Na}")
    Nd = k.shape[0]
    Nt = t.shape[0]

    fig, axs = plt.subplots(1, Nt, sharex=True, sharey=True,
                            figsize=(4 * Nt, 4),
                            squeeze=False)

    for i in range(Nt):
        ax = axs[0][i]
        ax = plot_vornoi_map(x[:, 0:2], dtec[0, :, i], ax=ax, colorbar=True)
        ax.set_xlabel(r"$x_{\rm east}$")
        if i == 0:
            ax.set_ylabel(r"$x_{\rm north}$")
        ax.set_title(f"TEC screen for dir 0 @ t={int(t[i])} sec")

    plt.savefig("simulated_dtec.pdf")
    plt.close('all')


def compute_representation(specification: SPECIFICATION, S_marg: int, compute_tec: bool,
                           antennas: ac.ITRS, directions: ac.ICRS, times: at.Time, ref_ant: ac.ITRS, ref_time: at.Time):
    # Plot Antenna Layout in East North Up frame
    ref_frame = ENU(obstime=ref_time, location=ref_ant.earth_location)

    _antennas = ac.ITRS(*antennas.cartesian.xyz, obstime=ref_time).transform_to(ref_frame)
    plt.scatter(_antennas.east, _antennas.north, marker='+')
    plt.xlabel(f"East (m)")
    plt.ylabel(f"North (m)")
    plt.savefig("antenna_locations.pdf")
    plt.close('all')
    _antennas = _antennas.cartesian.xyz.to(au.km).value.T
    max_baseline = np.max(np.linalg.norm(_antennas[:, None, :] - _antennas[None, :, :], axis=-1))
    logger.info(f"Maximum antenna baseline: {max_baseline} km")

    x0 = ac.ITRS(*antennas[0].cartesian.xyz, obstime=ref_time).transform_to(ref_frame).cartesian.xyz.to(au.km).value
    earth_centre_x = ac.ITRS(x=0 * au.m, y=0 * au.m, z=0. * au.m, obstime=ref_time).transform_to(
        ref_frame).cartesian.xyz.to(au.km).value

    northern_hemisphere = ref_ant.earth_location.geodetic.lat.value > 0

    tomo_kernel = build_ionosphere_tomographic_kernel(x0=x0, earth_centre=earth_centre_x,
                                                      specification=specification,
                                                      S_marg=S_marg,
                                                      compute_tec=compute_tec,
                                                      northern_hemisphere=northern_hemisphere)

    t = times.mjd * 86400.
    t -= t[0]

    X1 = dict(x=[], k=[], t=[], ref_x=[])

    logger.info("Computing coordinates in frame ...")
    for i, time in tqdm(enumerate(times)):
        frame = ENU(obstime=time, location=ref_ant.earth_location)

        x = ac.ITRS(*antennas.cartesian.xyz, obstime=time).transform_to(frame).cartesian.xyz.to(
            au.km).value.T
        ref_ant_x = ac.ITRS(*ref_ant.cartesian.xyz, obstime=time).transform_to(frame).cartesian.xyz.to(
            au.km).value

        k = directions.transform_to(frame).cartesian.xyz.value.T

        X = make_coord_array(t[i:i + 1, None], x, k, ref_ant_x[None, :], flat=True)

        X1['t'].append(X[:, 0:1])
        X1['x'].append(X[:, 1:4])
        X1['k'].append(X[:, 4:7])
        X1['ref_x'].append(X[:, 7:8])
    # Stacking in time, gives shape of data (Nt, Na, Nd)
    X1 = GeodesicTuple(**dict((key, jnp.concatenate(value, axis=0)) for key, value in X1.items()))
    logger.info(f"Total number of coordinates: {X1.x.shape[0]}")

    def compute_covariance_row(X1: GeodesicTuple, X2: GeodesicTuple):
        K = tomo_kernel.cov_func(X1, X2)

        return K[0, :]

    covariance_row = lambda X: compute_covariance_row(tree_map(lambda x: x.reshape((1, -1)), X), X1)

    mean = jit(lambda X1: tomo_kernel.mean_func(X1))(X1)
    mean.block_until_ready()

    is_nan = jnp.any(jnp.isnan(mean))
    if is_nan:
        logger.info(f"Nans appears in mean:")
        logger.info(f"{np.where(np.isnan(mean))}")

    t0 = default_timer()
    compute_covariance_row_parallel = chunked_pmap(covariance_row, chunksize=len(devices()), batch_size=X1.x.shape[0])
    # compute_covariance_row_parallel = vmap(covariance_row)
    # with disable_jit():
    cov = compute_covariance_row_parallel(X1)
    cov.block_until_ready()
    logger.info(f"Computation of the tomographic covariance took {default_timer() - t0} seconds.")

    is_nan = jnp.any(jnp.isnan(cov))
    if is_nan:
        logger.info(f"Nans appears in covariance matrix:")
        logger.info(f"{np.where(np.isnan(cov))}")

    return mean, cov


def simulate_from_representation(mean: jnp.ndarray, cov: jnp.ndarray,
                                 Nt: int, Na: int, Nd: int):
    @jit
    def cholesky_simulate(key, jitter):
        Z = random.normal(key, (cov.shape[0], 1), dtype=cov.dtype)
        L = jnp.linalg.cholesky(cov + jitter * jnp.eye(cov.shape[0]))
        dtec = (L @ Z + mean[:, None])[:, 0].reshape((Nt, Na, Nd)).transpose((2, 1, 0))
        is_nans = jnp.any(jnp.isnan(L))
        return is_nans, dtec

    @jit
    def svd_simulate(key):
        Z = random.normal(key, (cov.shape[0], 1), dtype=cov.dtype)
        max_eig, min_eig, L = msqrt(cov)
        dtec = (L @ Z + mean[:, None])[:, 0].reshape((Nt, Na, Nd)).transpose((2, 1, 0))
        is_nans = jnp.any(jnp.isnan(L))
        return max_eig, min_eig, is_nans, dtec

    t0 = default_timer()
    jitter = 1e-3
    logger.info(f"Computing Cholesky with jitter: {jitter}")
    logger.info(f"Jitter: {jitter} adds equivalent of {jnp.sqrt(jitter)} mTECU white noise to simulated DTEC.")
    is_nans, dtec = cholesky_simulate(random.PRNGKey(42), jitter)
    is_nans.block_until_ready()
    logger.info(f"Cholesky-based simulation took {default_timer() - t0} seconds.")
    if is_nans:
        jitter = 0.5 ** 2
        logger.info("Numerically instable. Using numpy cholesky.")
        t0 = default_timer()
        logger.info(f"Computing Cholesky with jitter: {jitter}")
        logger.info(f"Jitter: {jitter} adds equivalent of {jnp.sqrt(jitter)} mTECU white noise to simulated DTEC.")
        is_nans, dtec = cholesky_simulate(random.PRNGKey(42), jitter)
        logger.info(f"Cholesky-based simulation took {default_timer() - t0} seconds.")
    if is_nans:
        t0 = default_timer()
        logger.info("Numerically instable. Using SVD.")
        max_eig, min_eig, is_nans, dtec = svd_simulate(random.PRNGKey(42))
        is_nans.block_until_ready()
        logger.info(f"SVD-based simulation took {default_timer() - t0} seconds.")
        logger.info(f"Condition: {max_eig / min_eig}, minimum/maximum singular values {min_eig}, {max_eig}")
        if is_nans:
            raise ValueError("Covariance matrix is too numerically unstable.")
    return dtec


def grid_coordinates(coords: np.ndarray, dx: float):
    min_bounds = np.min(coords, axis=0)
    max_bounds = np.max(coords, axis=0)
    bins = [np.arange(min_bound - dx, max_bound + 2 * dx, dx) for min_bound, max_bound in
            zip(min_bounds, max_bounds)]
    bin_centres = list(map(lambda edges: 0.5 * (edges[1:] + edges[:-1]), bins))
    mgrid = np.meshgrid(*bin_centres, indexing='ij')
    grid_coords = np.stack([c.flatten() for c in mgrid], axis=1)  # M, D
    f, _ = np.histogramdd(coords, bins=bins)
    mask = f.flatten() > 0
    keep_coords = grid_coords[mask]
    return keep_coords


def test_grid_coordiates():
    coords = np.random.uniform(low=0, high=[1, 1], size=(10, 2))
    coords_grid = grid_coordinates(coords, dx=0.15)
    dx = np.min(np.max(np.abs(coords_grid[:, None, :] - coords[None, :, :]), axis=-1), axis=-1)
    assert np.all(dx <= 0.15)
    #
    # import pylab as plt
    # plt.scatter(coords[:, 0], coords[:, 1])
    # plt.scatter(coords_grid[:, 0], coords_grid[:, 1])
    # plt.show()


class Simulation(object):
    def __init__(self,
                 specification: SPECIFICATION,
                 S_marg: int = 25):
        """
        Simulation of DTEC.

        Args:
            specification: the ionosphere specification
            S_marg: the quadrature resolution
            compute_tec: if true that only compute TEC not diferential TEC
        """
        self.specification = specification
        self.S_marg = S_marg

    def run(self, output_h5parm: str, avg_direction_spacing: float, field_of_view_diameter: float,
            duration: float,
            time_resolution: float, grid_res_m: float,
            start_time, array_name, phase_tracking: ac.ICRS, min_freq: float = 700., max_freq: float = 2000.,
            Nf: int = 2, sky_model: str | None = None,
            Nd: int | None = None,
            clobber: bool = True):
        """
        Launch the simulation.

        Args:
            Nd: number of directions, if not given uses sky model
            sky_model: BBS skymodel, must be given if Nd is None
            Nf: number of frequencies
            max_freq: minimum frequency
            min_freq: maximum frequency
            output_h5parm: str, name of output H5Parm
            avg_direction_spacing: average spacing between directions in arcmin
            field_of_view_diameter: width of field of view in degrees
            duration: duration in seconds of observation
            time_resolution: resolution in seconds of simulation of ionosphere
            start_time: time in modified Julian days, (mjd)
            array_name: array name to simulate
            phase_tracking: `astropy.coordinates.ICRS` of phase tracking centre
            clobber: if true overwrite existing simulation (Default True!)
        """
        directions = None
        if sky_model is not None:
            directions = directions_from_sky_model(sky_model)
            Nd = len(directions)

        if Nd is None:
            Nd = get_num_directions(avg_direction_spacing, field_of_view_diameter)
        Nt = max(1, int(duration / time_resolution) + 1)
        # which is incorrect near poles. Stay away from poles.
        _ = create_empty_datapack(Nd, Nf, Nt,
                                  pols=None,
                                  field_of_view_diameter=field_of_view_diameter,
                                  start_time=start_time,
                                  time_resolution=time_resolution,
                                  min_freq=min_freq,
                                  max_freq=max_freq,
                                  array_file=ARRAYS[array_name],
                                  phase_tracking=(phase_tracking.ra.deg, phase_tracking.dec.deg),
                                  save_name=output_h5parm,
                                  directions=directions,
                                  clobber=clobber)
        # Grid the array
        _, antennas = load_array_file(ARRAYS[array_name])
        antennas_grid = grid_coordinates(coords=antennas, dx=grid_res_m)  # 1.5km grid
        antennas_grid = ac.ITRS(x=antennas_grid[:, 0] * au.m, y=antennas_grid[:, 1] * au.m,
                                z=antennas_grid[:, 2] * au.m)
        gridded_array_name = f"{ARRAYS[array_name]}.gridded"
        antennas_grid_labels = list(map(lambda i: f"ant{i}", range(len(antennas_grid))))
        save_array_file(gridded_array_name, antennas=antennas_grid, labels=antennas_grid_labels)

        dp_grid = create_empty_datapack(Nd, Nf, Nt,
                                        pols=None,
                                        field_of_view_diameter=field_of_view_diameter,
                                        start_time=start_time,
                                        time_resolution=time_resolution,
                                        min_freq=min_freq,
                                        max_freq=max_freq,
                                        array_file=gridded_array_name,
                                        phase_tracking=(phase_tracking.ra.deg, phase_tracking.dec.deg),
                                        save_name=f"{output_h5parm}.gridded",
                                        directions=directions,
                                        clobber=clobber)
        # Run simulation on gridded array
        with dp_grid:
            dp_grid.current_solset = 'sol000'
            dp_grid.select(pol=slice(0, 1, 1))
            axes = dp_grid.axes_phase
            patch_names, directions = dp_grid.get_directions(axes['dir'])
            antenna_labels, antennas = dp_grid.get_antennas(axes['ant'])
            timestamps, times = dp_grid.get_times(axes['time'])
            _, freqs = dp_grid.get_freqs(axes['freq'])
            ref_ant = antennas[0]
            ref_time = times[0]

        with open(f"{output_h5parm}.sky_model", "w") as f:
            f.write(
                f"# (Name,Type,Ra,Dec,I, ReferenceFrequency='{str(0.5 * (min_freq + max_freq))}e6', SpectralIndex='[0.0]') = format\n")
            for patch_name, direction in zip(patch_names, directions):
                # 3C196, POINT, 08:13:36.062300, +48.13.02.24900, 153.0, , [-0.56, -0.05212]
                f.write(f"{patch_name.decode()}, POINT, "
                        f"{direction.ra.to_string(unit=au.hour, alwayssign=False, sep=':', pad=True, precision=6)}, "
                        f"{direction.dec.to_string(unit=au.degree, alwayssign=True, sep='.', pad=True, precision=5)}, "
                        f"1.0, , \n")

        Na = len(antennas)
        Nd = len(directions)
        Nt = len(times)

        logger.info(f"Number of directions: {Nd}")
        logger.info(f"Number of antennas: {Na}")
        logger.info(f"Number of times: {Nt}")
        logger.info(f"Reference Ant: {ref_ant}")
        logger.info(f"Reference Time: {ref_time.isot}")

        # plot Ra/Dec
        plt.scatter(directions.ra.deg, directions.dec.deg)
        plt.xlabel(f"RA(J2000) (dg)")
        plt.ylabel(f"DEC(J2000) (deg)")
        plt.savefig("directions.pdf")
        plt.close('all')

        mean, cov = compute_representation(
            specification=self.specification,
            S_marg=self.S_marg,
            compute_tec=True,
            antennas=antennas,
            directions=directions,
            times=times,
            ref_ant=ref_ant,
            ref_time=ref_time
        )

        logger.info("Saving cov, and mean")
        np.save("dtec_covariance.npy", cov)
        np.save("dtec_mean.npy", mean)

        plt.plot(mean)
        plt.savefig("dtec_mean.pdf")
        plt.close('all')

        plt.imshow(cov, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.savefig("dtec_covariance.pdf")
        plt.close('all')

        plt.hist(np.sqrt(np.diag(cov)), bins='auto')
        plt.savefig("dtec_variance_hist.pdf")
        plt.close('all')

        dtec = simulate_from_representation(
            mean=mean,
            cov=cov,
            Nt=Nt,
            Na=Na,
            Nd=Nd
        )

        # Reference against reference antenna
        dtec -= dtec[:, 0:1, :]  # Nd, Na, Nt

        logger.info(f"Saving result to {dp_grid.filename}")
        with dp_grid:
            dp_grid.current_solset = 'sol000'
            dp_grid.select(pol=slice(0, 1, 1))
            dp_grid.tec = np.asarray(dtec[None])
            phase = wrap(dtec[..., None, :] * (TEC_CONV / freqs.to('Hz').value[:, None]))
            dp_grid.phase = np.asarray(phase[None])

        # Interpolate to output h5parm
        interpolate_h5parm(input_h5parm=dp_grid.filename, output_h5parm=output_h5parm,
                           k=7)
        visualisation(output_h5parm)


def msqrt(A):
    """
    Computes the matrix square-root using SVD, which is robust to poorly conditioned covariance matrices.
    Computes, M such that M @ M.T = A

    Args:
    A: [N,N] Square matrix to take square root of.

    Returns: [N,N] matrix.
    """
    U, s, Vh = jnp.linalg.svd(A)
    L = U * jnp.sqrt(s)
    max_eig = jnp.max(s)
    min_eig = jnp.min(s)
    return max_eig, min_eig, L

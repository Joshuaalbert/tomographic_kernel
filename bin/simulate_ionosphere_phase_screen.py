from jax.config import config

config.update("jax_enable_x64", True)

import argparse

from tomographic_kernel.simulation.ionosphere_simulation import ARRAYS, Simulation

import logging
import sys

import os

import astropy.coordinates as ac
import astropy.time as at

logger = logging.getLogger(__name__)


def main(output_h5parm, ncpu, phase_tracking,
         array_name, start_time, time_resolution, duration,
         field_of_view_diameter, avg_direction_spacing,
         S_marg,
         specification, min_freq, max_freq, Nf, sky_model, Nd):
    """
    Run the simulator.
    """
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ncpu}"

    sim = Simulation(specification=specification, S_marg=S_marg, compute_tec=False)
    sim.run(output_h5parm=output_h5parm, avg_direction_spacing=avg_direction_spacing,
            field_of_view_diameter=field_of_view_diameter, duration=duration, time_resolution=time_resolution,
            start_time=start_time, array_name=array_name, phase_tracking=phase_tracking,
            min_freq=min_freq, max_freq=max_freq, Nf=Nf,
            sky_model=sky_model, Nd=Nd)


def debug_main():
    phase_tracking = ac.SkyCoord("00h00m0.0s", "+37d07m47.400s", frame='icrs')
    main(output_h5parm='dsa2000W_2000m_datapack.h5',
         ncpu=6,
         phase_tracking=phase_tracking,
         array_name='dsa2000W',
         start_time=at.Time('2019-03-19T19:58:14.9', format='isot'),
         time_resolution=15.,
         duration=45.,
         field_of_view_diameter=4.,
         avg_direction_spacing=10000.,
         S_marg=15,
         specification='simple',
         min_freq=700.,
         max_freq=2000.,
         Nf=2,
         sky_model=None,
         Nd=None)


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.register("type", "phase_tracking", lambda v: ac.SkyCoord(*v.split(" "), frame='icrs'))
    parser.register("type", "start_time", lambda v: at.Time(v, format='isot'))
    parser.add_argument('--output_h5parm', help='H5Parm file to file to place the simulated differential TEC, ".h5"',
                        default=None, type=str, required=True)
    parser.add_argument('--phase_tracking',
                        help='Phase tracking center in ICRS frame in format "00h00m0.0s +37d07m47.400s".',
                        default=None, type="phase_tracking", required=True)
    parser.add_argument('--array_name', help=f'Name of array, options are {sorted(list(ARRAYS.keys()))}.',
                        default='dsa2000W_2000m_grid', type=str, required=True)
    parser.add_argument('--start_time', help=f'Start time in isot format "2019-03-19T19:58:14.9".',
                        default=None, type='start_time', required=True)
    parser.add_argument('--time_resolution', help=f'Temporal resolution in seconds.',
                        default=30., type=float, required=False)
    parser.add_argument('--duration', help=f'Temporal resolution in seconds.',
                        default=0., type=float, required=False)
    parser.add_argument('--field_of_view_diameter', help=f'Diameter of field of view in degrees.',
                        default=4., type=float, required=False)
    parser.add_argument('--avg_direction_spacing', help=f'Average spacing between directions in arcmin.',
                        default=32., type=float, required=False)
    parser.add_argument('--ncpu', help='Number of CPUs to use to compute covariance matrix.',
                        default=None, type=int, required=False)
    parser.add_argument('--S_marg', help='Resolution of simulation',
                        default=25, type=int, required=False)
    parser.add_argument('--specification', help=f'Ionosphere spec: simple, dawn, dusk, dawn_challenge, dusk_challenge',
                        default='dawn', type=str, required=False)
    parser.add_argument('--min_freq', help=f'Min frequency in MHz',
                        default=700., type=float, required=False)
    parser.add_argument('--max_freq', help=f'Max frequency in MHz',
                        default=2000., type=float, required=False)
    parser.add_argument('--Nf', help=f'Number of channels',
                        default=1, type=int, required=False)
    parser.add_argument('--sky_model', help=f'Sky model to pull directions from.',
                        default=None, type=str, required=False)
    parser.add_argument('--Nd',
                        help=f'Optional number directions, in which case we wont compute from avg_direction_spacing.',
                        default=None, type=int, required=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulates DTEC over an observation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if len(sys.argv) == 1:
        debug_main()
        exit(0)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    logger.info("Running with:")
    for option, value in vars(flags).items():
        logger.info("\t{} -> {}".format(option, value))
    main(**vars(flags))

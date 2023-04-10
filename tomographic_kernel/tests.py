from jax.config import config

from tomographic_kernel.frames import ENU
from tomographic_kernel.plotting import plot_vornoi_map

config.update("jax_enable_x64", True)

from h5parm import DataPack
import astropy.units as au
import astropy.coordinates as ac
import pylab as plt


def test_plot_data():
    dp = DataPack('/home/albert/data/gains_screen/data/L342938_DDS5_full_merged.h5', readonly=True)
    with dp:
        select = dict(pol=slice(0, 1, 1), ant=[0, 50], time=slice(0, 100, 1))
        dp.current_solset = 'sol000'
        dp.select(**select)
        tec_mean, axes = dp.tec
        tec_mean = tec_mean[0, ...]
        const_mean, axes = dp.const
        const_mean = const_mean[0, ...]
        tec_std, axes = dp.weights_tec
        tec_std = tec_std[0, ...]
        patch_names, directions = dp.get_directions(axes['dir'])
        antenna_labels, antennas = dp.get_antennas(axes['ant'])
        timestamps, times = dp.get_times(axes['time'])
    antennas = ac.ITRS(*antennas.cartesian.xyz, obstime=times[0])
    ref_ant = antennas[0]

    for i, time in enumerate(times):
        frame = ENU(obstime=time, location=ref_ant.earth_location)
        antennas = antennas.transform_to(frame)
        directions = directions.transform_to(frame)
        x = antennas.cartesian.xyz.to(au.km).value.T
        k = directions.cartesian.xyz.value.T

        dtec = tec_mean[:, 1, i]
        ax = plot_vornoi_map(k[:, 0:2], dtec)
        ax.set_title(time)
        plt.show()

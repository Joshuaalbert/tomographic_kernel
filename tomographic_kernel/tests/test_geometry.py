from jax import numpy as jnp

from tomographic_kernel.tomographic_kernel import rotate_vector, frozen_flow_transform


def test_rotate_vector():
    x = jnp.asarray([1., 0., 0.])
    y = jnp.asarray([0., 1., 0.])
    z = jnp.asarray([0., 0., 1.])
    angle = jnp.pi / 2.
    assert jnp.allclose(rotate_vector(x, z, angle), y, atol=1e-6)
    assert jnp.allclose(rotate_vector(x, y, angle), -z, atol=1e-6)
    assert jnp.allclose(rotate_vector(z, y, angle), x, atol=1e-6)
    assert jnp.allclose(rotate_vector(y, x, angle), z, atol=1e-6)
    assert not jnp.any(jnp.isnan(rotate_vector(y, jnp.asarray([0, 0, 0]), angle)))


def test_frozen_flow_transform():
    earth_centre = jnp.asarray([0., 0., 0.])
    t = 0.
    y = jnp.asarray([0., 0., 6400.])
    x0 = y
    bottom = 300.
    wind_velocity = jnp.asarray([-0.240, 0.030, 0.])
    assert jnp.allclose(y,
                        frozen_flow_transform(t, y, x0, bottom, earth_centre=earth_centre, wind_velocity=wind_velocity))

    t = 60
    y = jnp.asarray([0., 0., 6400.])
    x0 = jnp.asarray([0., 100., 6400.])
    bottom = 300.
    wind_velocity = jnp.asarray([-0.240, 0.030, 0.])
    print(frozen_flow_transform(t, y, x0, bottom, earth_centre=earth_centre, wind_velocity=wind_velocity))

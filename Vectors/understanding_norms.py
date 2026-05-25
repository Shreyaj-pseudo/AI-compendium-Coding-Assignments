import jax.numpy as jnp

v = jnp.array([3.0, 50.0, 1.0,2,4,1,2])

l1 = jnp.sum(jnp.abs(v))
l2 = jnp.sqrt(jnp.sum(v ** 2))

print(f"L1: {l1}, L2: {l2:.2f}")
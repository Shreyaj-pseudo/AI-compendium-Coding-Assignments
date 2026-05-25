import jax.numpy as jnp

u = jnp.array([1.0, 1.0, 3.0])
v = jnp.array([3.0, 10.0, 1.0])

euclidean = jnp.sqrt(jnp.sum((u - v) ** 2))
manhattan = jnp.sum(jnp.abs(u - v))

print(f"Euclidean: {euclidean:.2f}, Manhattan: {manhattan}")
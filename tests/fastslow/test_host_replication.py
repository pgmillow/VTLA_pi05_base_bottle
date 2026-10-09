import jax
import numpy as np
import pytest

from openpi.training import sharding


def test_launcher_transfer_guard_is_explicitly_applied(monkeypatch):
    previous = jax.config.jax_transfer_guard_device_to_device
    monkeypatch.setenv('JAX_TRANSFER_GUARD_DEVICE_TO_DEVICE', 'disallow')
    try:
        sharding.make_mesh(1)
        assert jax.config.jax_transfer_guard_device_to_device == 'disallow'
    finally:
        jax.config.update('jax_transfer_guard_device_to_device', previous)


@pytest.mark.parametrize('typed_key', [True, False])
def test_control_values_uploaded_from_cpu_to_each_device(monkeypatch, typed_key):
    if len(jax.devices()) < 2:
        pytest.skip('Run with XLA_FLAGS=--xla_force_host_platform_device_count=4')
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('replicas',))
    target = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    source = jax.random.split(jax.random.key(43))[0] if typed_key else np.asarray(7, np.int32)
    expected = np.asarray(jax.random.key_data(source)) if typed_key else source
    original_put = jax.device_put
    uploads = []

    def require_host_input(value, device, *args, **kwargs):
        assert isinstance(value, np.ndarray), 'Device-to-device copy reintroduced'
        uploads.append(device)
        return original_put(value, device, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(jax, 'device_put', require_host_input)
        result = sharding.replicate_via_host(source, target)
    assert set(uploads) == target.addressable_devices
    assert len(uploads) == len(target.addressable_devices)
    assert result.sharding.is_equivalent_to(target, result.ndim)
    for shard in result.addressable_shards:
        actual = jax.random.key_data(shard.data) if typed_key else shard.data
        np.testing.assert_array_equal(np.asarray(actual), expected)
    # Already replicated inputs are reused; no repeated host transfers each step.
    assert sharding.replicate_via_host(result, target) is result
    if typed_key:
        folded = jax.random.fold_in(result, 2)
        expected_fold = jax.random.key_data(jax.random.fold_in(source, 2))
        for shard in folded.addressable_shards:
            np.testing.assert_array_equal(np.asarray(jax.random.key_data(shard.data)), expected_fold)

import torch as th
from functorch import combine_state_for_ensemble, vmap


def test_output_size():
    num_models = 5
    batch_size = 64
    in_features, out_features = 3, 3
    models = [th.nn.Linear(in_features, out_features) for i in range(num_models)]
    data = th.randn(batch_size, in_features)

    fmodel, params, buffers = combine_state_for_ensemble(models)
    output = vmap(fmodel, (0, 0, None))(params, buffers, data)

    assert output.shape == (num_models, batch_size, out_features)

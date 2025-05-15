import numpy as np
import tensorflow as tf
from baseline_MAST import get_mast_pretraining_model

# Define a variety of input shapes to test
input_shapes = [
    (128, 128),
    (64, 80),
    (100, 100),
    (256, 64),
]

for shape in input_shapes:
    print(f"Testing input shape: {shape}")
    model = get_mast_pretraining_model(input_shape=shape, patch_size=(16, 16), masking_ratio=0.5)
    dummy_input = np.random.randn(1, shape[0], shape[1], 1).astype(np.float32)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape[1:3] == shape, f"Output shape {output.shape[1:3]} does not match input shape {shape}"  # Only check H, W
print("All output shapes match input shapes!")

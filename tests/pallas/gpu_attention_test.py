# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
from jax._src import config
from jax._src import test_util as jtu
from jax.experimental.pallas.ops.gpu import decode_attention
import jax.numpy as jnp
import numpy as np


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

# pylint: disable=no-value-for-parameter


config.parse_flags_with_absl()


@jtu.with_config(jax_traceback_filtering="off")
class PallasBaseTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest("Must only run on GPUs, or CPUs")
    if jtu.test_device_matches(["cpu"]) and not self.INTERPRET:
      self.skipTest("On CPU, the test works only in interpret mode")
    if jax.config.x64_enabled:
      self.skipTest("The test works only in 32-bit")
    if (jtu.test_device_matches(["cuda"]) and
        not jtu.is_cuda_compute_capability_at_least("8.0")):
      self.skipTest("Only works on GPU with capability >= sm80")
    if sys.platform == "win32" and not self.INTERPRET:
      self.skipTest("Only works on non-Windows platforms")

    super().setUp()


class DecodeAttentionTest(PallasBaseTest):
  INTERPRET = False

  @parameterized.named_parameters(*[
      (
          f"{batch_size=}_{seq_len=}_{num_heads=}_{head_dim=}_{kwargs=}",
          batch_size,
          seq_len,
          num_heads,
          head_dim,
          kwargs,
      )
      for (
          batch_size,
          seq_len,
          num_heads,
          head_dim,
          kwargs,
      ) in [
          (1, 1024, 1, 64, {}),
          (2, 1024, 2, 64, {}),
          (1, 1024, 8, 64, {}),
      ]
  ])
  @jax.numpy_dtype_promotion("standard")
  def test_mqa(
      self,
      batch_size,
      seq_len,
      num_heads,
      head_dim,
      kwargs,
  ):
    del kwargs

    k1, k2, k3 = random.split(random.key(0), 3)
    q = random.normal(k1, (batch_size, num_heads, head_dim), dtype=jnp.float16)
    k = random.normal(k2, (batch_size, seq_len, head_dim), dtype=jnp.float16)
    v = random.normal(k3, (batch_size, seq_len, head_dim), dtype=jnp.float16)

    o = decode_attention.mqa(q, k, v, interpret=self.INTERPRET)
    o_ref = decode_attention.mqa_reference(q, k, v)
    np.testing.assert_allclose(o, o_ref, atol=0.05)

  @parameterized.named_parameters(*[
      (
          f"{batch_size=}_{seq_len=}_{num_q_heads=}_{num_kv_heads=}_{head_dim=}_{kwargs=}",
          batch_size,
          seq_len,
          num_q_heads,
          num_kv_heads,
          head_dim,
          kwargs,
      )
      for (
          batch_size,
          seq_len,
          num_q_heads,
          num_kv_heads,
          head_dim,
          kwargs,
      ) in [
          (1, 1024, 16, 4, 64, {}),
          (1, 1024, 16, 16, 64, {}),
          (1, 1024, 32, 32, 64, {}),
      ]
  ])
  @jax.numpy_dtype_promotion("standard")
  def test_gqa(
      self,
      batch_size,
      seq_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      kwargs,
  ):
    del kwargs

    k1, k2, k3 = random.split(random.key(0), 3)
    q = random.normal(
        k1, (batch_size, num_q_heads, head_dim), dtype=jnp.float16
    )
    k = random.normal(
        k2, (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16
    )
    v = random.normal(
        k3, (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16
    )

    o = decode_attention.gqa(q, k, v, interpret=self.INTERPRET)
    o_ref = decode_attention.gqa_reference(q, k, v)
    np.testing.assert_allclose(o, o_ref, atol=0.05)

class DecodeAttentionInterpreterTest(DecodeAttentionTest):
  INTERPRET = True


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())

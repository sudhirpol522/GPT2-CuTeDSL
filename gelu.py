import torch
import cutlass.cute as cute
import cutlass
from cutlass.cute.runtime import from_dlpack
from cuda.bindings.driver import CUstream
import cutlass.torch as cutlass_torch



@cute.kernel
def gelu_coalsed(
  X:cute.Tensor,
  Y:cute.Tensor,
  N:cute.Int32):
  tidx,_,_=cute.arch.thread_idx()
  bidx,_,_=cute.arch.block_idx()

  idx=bidx*256+tidx
  if idx<N:
    x = X[idx]

    u = 0.7978845608028654 * (
        x + 0.044715 * x * x * x
    )

    t = cute.math.tanh(u)

    Y[idx] = 0.5 * x * (1.0 + t)


@cute.jit
def launch_v0(X:cute.Tensor, Y:cute.Tensor,stream:CUstream):
  N=X.shape[0]
  gelu_coalsed(X,Y,N).launch(grid=[cute.ceil_div(N,256),1,1], block=[256,1,1],stream=stream)

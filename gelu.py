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


## 
* **Small workloads (decoding):** Keep elements per thread low (e.g., 1) and use smaller CTAs to maximize SM coverage—or eliminate launch overhead entirely via kernel fusion and CUDA Graphs.
* **Large workloads (prefill):** Use larger CTAs, thread coarsening (multiple elements per thread), and contiguous vectorized memory accesses to reduce scheduling overhead and maximize ILP and memory bandwidth.
* **General rule:** Dynamically tune threads and elements per thread based on input size ($N$) to balance GPU-wide parallelism (enough CTAs to fill all SMs) against per-thread efficiency (ILP and register pressure).

                                                                                                                                                                                            
                                                                                                                                                                                            @cute.kernel
def gelu_extra_elements_per_thread(X:cute.Tensor, Y:cute.Tensor,N:cute.Tensor, EPT:cute.Constexpr):
  tidx,_,_=cute.arch.thread_idx()
  bidx,_,_=cute.arch.block_idx()
  bdim,_,_=cute.arch.block_dim()

  idx=bidx*bdim*EPT+tidx
  for j in cutlass.range_constexpr(EPT):

    idx = base + j * bdim

    if idx < X.shape[0]:
        x = X[idx]
        x = cutlass.Float32(x)

        u = 0.7978845608028654 * (
            x + 0.044715 * x * x * x
        )

        t = cute.math.tanh(u)

        t=0.5 * x * (1.0 + t)

        Y[idx] = y.to(Y.element_type)

@cute.jit
def launch_v1(X:cute.Tensor, Y:cute.Tensor,stream:CUstream):
  blocks=cute.ceil_div(X.shape[0],THREADS*EPT)
  gelu_extra_elements_per_thread(X,Y,blocks,EPT).launch(grid=[blocks,1,1], block=[THREADS,1,1],stream=stream)

##########

@cute.jit
def gelu_math(x):
    x = x.to(cutlass.Float32)

    u = 0.7978845608028654 * (
        x + 0.044715 * x * x * x
    )

    t = cute.math.tanh(u)

    return 0.5 * x * (1.0 + t)


@cute.kernel
def gelu_vectorized(X:cute.Tensor, Y:cute.Tensor,N:cute.Int32, VEC:cutlass.Constexpr):
  tidx,_,_=cute.arch.thread_idx()
  bidx,_,_=cute.arch.block_idx()
  bdim,_,_=cute.arch.block_dim()

  group=bidx*bdim+tidx
  base=group*VEC

  if base+VEC<=N:
    x_view=cute.local_tile(X,(VEC,),(group,))
    y_view=cute.local_tile(Y,(VEC,),(group,))

    x = x_view.load()
    y_view.store(gelu_math(x).to(Y.element_type))

  else:
    for j in cutlass.range_constexpr(VEC):
      idx = base + j
      if idx < N:
          Y[idx] = gelu_math(
              X[idx]
          ).to(Y.element_type)



@cute.jit
def gelu(X: cute.Tensor, Y: cute.Tensor, N: cutlass.Int32):
    VEC = 128 // X.element_type.width          
    threads = 256
    groups = (N + VEC - 1) // VEC             
    blocks = (groups + threads - 1) // threads
    gelu_vectorized(X, Y, N, VEC).launch(grid=[blocks, 1, 1], block=[threads, 1, 1])


import torch
x_torch=torch.ones(1,100)
y_torch=torch.zeros(1,100)
x=from_dlpack(x_torch,assumed_align=16)
y=from_dlpack(y_torch,assumed_align=16)


X = from_dlpack(x_torch, assumed_align=16)
Y = from_dlpack(y_torch, assumed_align=16)

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

N = 100
x = torch.randn(N, device="cuda", dtype=torch.float16)
y = torch.empty_like(x)

X = from_dlpack(x, assumed_align=16)   # alignment lets the compiler use 128-bit loads
Y = from_dlpack(y, assumed_align=16)

gelu_c = cute.compile(gelu, X, Y, N)   # compile once, reuse
gelu_c(X, Y, N)
torch.cuda.synchronize()

ref = torch.nn.functional.gelu(x.float(), approximate="tanh").to(x.dtype)
torch.testing.assert_close(y, ref, atol=1e-2, rtol=1e-2)
print("max abs err:", (y.float() - ref.float()).abs().max().item())

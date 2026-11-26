import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch

from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
from cuda.bindings.driver import CUstream

@cute.kernel
def layernorm_v1(
    input_flat_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
    hidden_size: cutlass.Constexpr,
    gamma: cute.Tensor,
    beta: cute.Tensor,
    WARPS_PER_BLOCK: cutlass.Constexpr,
    EPSILON: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    WARP_SIZE = 32
    ELEMENTS_PER_THREAD = hidden_size // WARP_SIZE

    warp_id = tidx // WARP_SIZE
    lane_id = tidx % WARP_SIZE

    row_idx = bidx * WARPS_PER_BLOCK + warp_id
    num_rows = input_flat_tensor.shape[0] // hidden_size

    if row_idx < num_rows:

        row_offset = row_idx * hidden_size


        thread_sum = cutlass.Float32(0.0)

        for i in cutlass.range_constexpr(ELEMENTS_PER_THREAD):
            col_offset = lane_id + i * WARP_SIZE

            x = cutlass.Float32(
                input_flat_tensor[row_offset + col_offset]
            )

            thread_sum = thread_sum + x

        mean = cute.arch.warp_reduction_sum(thread_sum) / hidden_size

      

        thread_sq_diff_sum = cutlass.Float32(0.0)

        for i in cutlass.range_constexpr(ELEMENTS_PER_THREAD):
            col_offset = lane_id + i * WARP_SIZE

            x = cutlass.Float32(
                input_flat_tensor[row_offset + col_offset]
            )

            diff = x - mean

            thread_sq_diff_sum = (
                thread_sq_diff_sum + diff * diff
            )

        variance = (cute.arch.warp_reduction_sum(thread_sq_diff_sum)/ hidden_size)

        inv_std = cute.math.rsqrt(variance + EPSILON)

 
        for i in cutlass.range_constexpr(ELEMENTS_PER_THREAD):
            col_offset = lane_id + i * WARP_SIZE

            x = cutlass.Float32(
                input_flat_tensor[row_offset + col_offset]
            )

            gamma_val = cutlass.Float32(gamma[col_offset])
            beta_val = cutlass.Float32(beta[col_offset])

            normalized_val = (x - mean) * inv_std

            result = normalized_val * gamma_val + beta_val

            output_tensor[row_offset + col_offset] = (
                result.to(output_tensor.element_type)
            )



@cute.jit
def launch_ln_v1(
    input_tensor: cute.Tensor,
    gamma: cute.Tensor,
    beta: cute.Tensor,
    output_tensor: cute.Tensor,
    stream: CUstream,
    HIDDEN_SIZE: cutlass.Constexpr,
    WARPS_PER_BLOCK: cutlass.Constexpr,
    EPSILON: cutlass.Constexpr,
):
    WARP_SIZE = 32

    num_rows = input_tensor.shape[0] // HIDDEN_SIZE

    num_blocks = cute.ceil_div(
        num_rows, WARPS_PER_BLOCK
    )

    threads_per_block = WARPS_PER_BLOCK * WARP_SIZE

    layernorm_v1(
        input_tensor,
        output_tensor,
        HIDDEN_SIZE,
        gamma,
        beta,
        WARPS_PER_BLOCK,
        EPSILON,
    ).launch(
        grid=[num_blocks, 1, 1],
        block=[threads_per_block, 1, 1],
        stream=stream,
    )

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
###################

## the main takeaway from this kernel is that first create a logical view in the global memory then with that iterator create the "register tensor with the layout you are interested in" then pull the data in that register using the .load()
## method and then do the calculation in this we are able to skip the autovec copy, we have used the cutedsl based warp reduction then we have used the reduce method of "TensorSSA" directly which is inbuilt from this it seems cutedsl needs the data in the correct
## format rest of the computation is being handled by it
@cute.kernel
def layernorm_v2(
    input_flat_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
    hidden_size: cutlass.Constexpr,
    gamma: cute.Tensor,
    beta: cute.Tensor,
    WARPS_PER_BLOCK: cutlass.Constexpr,
    VECTOR_SIZE: cutlass.Constexpr,
    EPSILON: cutlass.Constexpr,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()

    WARP_SIZE = 32
    ELEMENTS_PER_THREAD = hidden_size // WARP_SIZE
    # this one is important its about how many vectors or values you have per thread across each dimension basically it is giving us 3 for 768 dimensional hidden dimension
    VECTORS_PER_THREAD = ELEMENTS_PER_THREAD // VECTOR_SIZE

    lane_id = thread_idx % WARP_SIZE
    warp_id = thread_idx // WARP_SIZE

    # Each warp processes one LayerNorm row.
    row_idx = block_idx * WARPS_PER_BLOCK + warp_id

    num_rows = input_flat_tensor.shape[0] // hidden_size

    if row_idx < num_rows:
        input_row = cute.local_tile(input_flat_tensor, (hidden_size,), (row_idx,))
        output_row = cute.local_tile(output_tensor, (hidden_size,), (row_idx,))

        warp_layout = cute.make_layout(
            (VECTOR_SIZE, WARP_SIZE, VECTORS_PER_THREAD),
            stride=(
                1,
                VECTOR_SIZE,
                WARP_SIZE * VECTOR_SIZE,
            ),
        )

        input_warp_tile = cute.make_tensor(
            input_row.iterator,
            warp_layout,
        )

        output_warp_tile = cute.make_tensor(
            output_row.iterator,
            warp_layout,
        )

        gamma_warp_tile = cute.make_tensor(
            gamma.iterator,
            warp_layout,
        )

        beta_warp_tile = cute.make_tensor(
            beta.iterator,
            warp_layout,
        )

        input_thread_tile = input_warp_tile[None, lane_id, None]
        output_thread_tile = output_warp_tile[None, lane_id, None]
        gamma_thread_tile = gamma_warp_tile[None, lane_id, None]
        beta_thread_tile = beta_warp_tile[None, lane_id, None]

        input_values = input_thread_tile.load().to(cutlass.Float32)

        thread_sum = input_values.reduce(
            cute.ReductionOp.ADD,
            cutlass.Float32(0.0),
            reduction_profile=0,
        )

        row_sum = cute.arch.warp_reduction_sum(thread_sum)

        mean = row_sum / hidden_size

        centered_values = input_values - mean

        squared_diff_values = centered_values * centered_values

        thread_sq_diff_sum = squared_diff_values.reduce(
            cute.ReductionOp.ADD,
            cutlass.Float32(0.0),
            reduction_profile=0,
        )

        row_sq_diff_sum = cute.arch.warp_reduction_sum(thread_sq_diff_sum)

        variance = row_sq_diff_sum / hidden_size

        inv_std = cute.math.rsqrt(variance + EPSILON)

        gamma_values = gamma_thread_tile.load().to(cutlass.Float32)
        beta_values = beta_thread_tile.load().to(cutlass.Float32)

        normalized_values = centered_values * inv_std

        output_values = normalized_values * gamma_values + beta_values

        output_thread_tile.store(output_values.to(output_tensor.element_type))



@cute.jit
def launch_ln_v2(
    input_flat_tensor: cute.Tensor,
    gamma: cute.Tensor,
    beta: cute.Tensor,
    output_tensor: cute.Tensor,
    stream: CUstream,
    HIDDEN_SIZE: cutlass.Constexpr,
    WARPS_PER_BLOCK: cutlass.Constexpr,
    VECTOR_SIZE: cutlass.Constexpr,
    EPSILON: cutlass.Constexpr,
):
    WARP_SIZE = 32

    num_rows = input_flat_tensor.shape[0] // HIDDEN_SIZE

    num_blocks = cute.ceil_div(
        num_rows,
        WARPS_PER_BLOCK,
    )

    threads_per_block = WARPS_PER_BLOCK * WARP_SIZE

    layernorm_v2(
        input_flat_tensor,
        gamma,
        beta,
        output_tensor,
        HIDDEN_SIZE,
        WARPS_PER_BLOCK,
        VECTOR_SIZE,
        EPSILON,
    ).launch(
        grid=[num_blocks, 1, 1],
        block=[threads_per_block, 1, 1],
        stream=stream,
    )

/**
 *  Copyright (c) 2021 by Contributors
 * @file array/cuda/rowwise_sampling.cu
 * @brief uniform rowwise sampling
 */

#include <curand_kernel.h>
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/tensordispatch.h>

#include <cub/cub.cuh>
#include <numeric>
#include<vector>

#include "../../array/cuda/atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

using namespace dgl::cuda;
using namespace dgl::aten::cuda;
using TensorDispatcher = dgl::runtime::TensorDispatcher;

namespace dgl {
namespace aten {
namespace impl {

namespace tempsample{

constexpr int BLOCK_SIZE = 128;

/**
 * @brief Compute the size of each row in the sampled CSR, without replacement.
 *
 * @tparam IdType The type of node and edge indexes.
 * @param num_picks The number of non-zero entries to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The index where each row's edges start.
 * @param out_deg The size of each row in the sampled matrix, as indexed by
 * `in_rows` (output).
 */
template <typename IdType>
__global__ void _CSRRowWiseSampleDegreeKernel(
    const int64_t num_picks, const int64_t num_rows,
    const IdType* const in_rows, const IdType* const in_ptr,
    IdType* const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int in_row = in_rows[tIdx];
    const int out_row = tIdx;
    out_deg[out_row] = min(
        static_cast<IdType>(num_picks), in_ptr[in_row + 1] - in_ptr[in_row]);

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}

/**
 * @brief Compute the size of each row in the sampled CSR, with replacement.
 *
 * @tparam IdType The type of node and edge indexes.
 * @param num_picks The number of non-zero entries to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The index where each row's edges start.
 * @param out_deg The size of each row in the sampled matrix, as indexed by
 * `in_rows` (output).
 */
template <typename IdType>
__global__ void _CSRRowWiseSampleDegreeReplaceKernel(
    const int64_t num_picks, const int64_t num_rows,
    const IdType* const in_rows, const IdType* const in_ptr,
    IdType* const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int64_t in_row = in_rows[tIdx];
    const int64_t out_row = tIdx;

    if (in_ptr[in_row + 1] - in_ptr[in_row] == 0) {
      out_deg[out_row] = 0;
    } else {
      out_deg[out_row] = static_cast<IdType>(num_picks);
    }

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}

/**
 * @brief Perform row-wise uniform sampling on a CSR matrix,
 * and generate a COO matrix, without replacement.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam TILE_SIZE The number of rows covered by each threadblock.
 * @param rand_seed The random seed to use.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param in_index The indices array of the input CSR.
 * @param data The data array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param out_rows The rows of the output COO (output).
 * @param out_cols The columns of the output COO (output).
 * @param out_idxs The data array of the output COO (output).
 */
template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType* const in_rows, const IdType* const in_ptr,
    const IdType* const in_index, const IdType* const data,
    const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
    IdType* const out_idxs, const bool shared_flag) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  extern __shared__ int64_t sm_sampled_result[];
  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
        out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        if(!shared_flag)
          out_idxs[out_row_start + idx] = idx;
        else 
          sm_sampled_result[out_row_start + idx]=idx;
      }
      __syncthreads();

      for (int idx = num_picks + threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const int num = curand(&rng) % (idx + 1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          if(!shared_flag)
            AtomicMax(out_idxs + out_row_start + num, idx);
          else
            AtomicMax(sm_sampled_result + out_row_start + num, idx);
        }
      }
      __syncthreads();

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        IdType perm_idx;
        if(!shared_flag)
          perm_idx= out_idxs[out_row_start + idx] + in_row_start;
        else
          perm_idx=sm_sampled_result[out_row_start + idx] + in_row_start;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[perm_idx];
        out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
      }
    }
    out_row += 1;
  }
}

/**
 * @brief Perform row-wise uniform sampling on a CSR matrix,
 * and generate a COO matrix, with replacement.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam TILE_SIZE The number of rows covered by each threadblock.
 * @param rand_seed The random seed to use.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param in_index The indices array of the input CSR.
 * @param data The data array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param out_rows The rows of the output COO (output).
 * @param out_cols The columns of the output COO (output).
 * @param out_idxs The data array of the output COO (output).
 */
template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformReplaceKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType* const in_rows, const IdType* const in_ptr,
    const IdType* const in_index, const IdType* const data,
    const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
    IdType* const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t out_row_start = out_ptr[out_row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;

    if (deg > 0) {
      // each thread then blindly copies in rows only if deg > 0.
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const int64_t edge = curand(&rng) % deg;
        const int64_t out_idx = out_row_start + idx;
        out_rows[out_idx] = row;
        out_cols[out_idx] = in_index[in_row_start + edge];
        out_idxs[out_idx] =
            data ? data[in_row_start + edge] : in_row_start + edge;
      }
    }
    out_row += 1;
  }
}

template <typename IdType, int TILE_SIZE>
void processRows(
  const IdType* const rows,IdType* const out_rows,const int64_t num_rows,
  const int64_t block_threshold){
  
  int block_nums=(num_rows + TILE_SIZE - 1) / TILE_SIZE;
  constexpr int BLOCK_SIZE = 128;
  int warp_nums=BLOCK_SIZE/32;
  if(num_rows<block_threshold){
    for(int i=0;i<num_rows;++i){
      int block_index=i%block_nums;
      int index=i/block_nums;
      if(index%2==1)
        block_index=block_nums-block_index-1;
      out_rows[block_index*TILE_SIZE+index]=rows[i];
    }
  } else {
    for(int i=0;i<num_rows;++i){
      int block_index=i/TILE_SIZE;
      int warp_index=(i-block_index*TILE_SIZE)%warp_nums;
      int index=(i-block_index*TILE_SIZE)/warp_nums;
      if(index%2==1)
        warp_index=warp_nums-warp_index-1;
      out_rows[block_index*TILE_SIZE+warp_index*TILE_SIZE/warp_nums+index]=rows[i];
    }
  }
}

}  // namespace

///////////////////////////// CSR sampling //////////////////////////

template <DGLDeviceType XPU, typename IdType>
COOMatrix _CSRRowWiseSamplingUniform(
    CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace, const bool shared_flag) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  const int64_t num_rows = rows->shape[0];
  const IdType* const slice_rows = static_cast<const IdType*>(rows->data);

  IdArray picked_row =
      NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col =
      NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx =
      NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));
  const IdType* in_cols = static_cast<IdType*>(GetDevicePointer(mat.indices));
  const IdType* data = CSRHasData(mat)
                           ? static_cast<IdType*>(GetDevicePointer(mat.data))
                           : nullptr;

  // compute degree
  IdType* out_deg = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
        tempsample::_CSRRowWiseSampleDegreeReplaceKernel, grid, block, 0, stream, num_picks,
        num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
        tempsample::_CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
        num_rows, slice_rows, in_ptr, out_deg);
  }

  // fill out_ptr
  IdType* out_ptr = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  NDArray new_len_tensor;
  if (TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty(
        {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty(
        {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  }

  // copy using the internal current stream
  CUDA_CALL(cudaMemcpyAsync(
      new_len_tensor->data, out_ptr + num_rows, sizeof(IdType),
      cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  // the number of rows each thread block will cover
  constexpr int BLOCK_SIZE = 128;
  constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  if (replace) {  // with replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
        (tempsample::_CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE>), grid, block,
        0, stream, random_seed, num_picks, num_rows, slice_rows, in_ptr,
        in_cols, data, out_ptr, out_rows, out_cols, out_idxs);
  } else {  // without replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
        (tempsample::_CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE>), grid, block, sizeof(IdType) * num_rows * num_picks,
        stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
        data, out_ptr, out_rows, out_cols, out_idxs, shared_flag);
  }
  device->FreeWorkspace(ctx, out_ptr);

  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

  const IdType new_len = static_cast<const IdType*>(new_len_tensor->data)[0];
  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  return COOMatrix(
      mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform(
    CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {
  if (num_picks == -1) {
    // Basically this is UnitGraph::InEdges().
    COOMatrix coo = CSRToCOO(CSRSliceRows(mat, rows), false);
    IdArray sliced_rows = IndexSelect(rows, coo.row);
    return COOMatrix(
        mat.num_rows, mat.num_cols, sliced_rows, coo.col, coo.data);
  } else {
    // get ctx
    const auto& ctx = rows->ctx;
    auto device = runtime::DeviceAPI::Get(ctx);
    cudaStream_t stream = runtime::getCurrentCUDAStream();
    const int64_t num_rows = rows->shape[0];
    const IdType* const slice_rows = static_cast<const IdType*>(rows->data);
    IdType* const sorted_rows = static_cast<IdType*>(
      device->AllocWorkspace(ctx, num_rows * sizeof(IdType)));
    const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));

    //sorted vertex
    IdType* degrees = static_cast<IdType*>(
      device->AllocWorkspace(ctx, num_rows * sizeof(IdType)));
    for(int i=0;i<num_rows;++i){
      IdType vertex=slice_rows[i];
      degrees[i]=in_ptr[vertex]-in_ptr[vertex+1];
    }
    IdType* sorted_degrees = static_cast<IdType*>(
      device->AllocWorkspace(ctx, num_rows * sizeof(IdType)));
    size_t temp_storage_size=0;
    void* temp_storage=nullptr;
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      temp_storage, temp_storage_size, degrees, sorted_degrees, slice_rows, sorted_rows, num_rows, stream=stream));
    temp_storage=device->AllocWorkspace(ctx, temp_storage_size);
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      temp_storage, temp_storage_size, degrees, sorted_degrees, slice_rows, sorted_rows, num_rows, stream=stream));
    device->FreeWorkspace(ctx, temp_storage);
    device->FreeWorkspace(ctx, degrees);

    //divide two partion
    const int64_t degree_threshold=100;
    int64_t high_degree_num=0;
    for(int i=0;i<num_rows;i++){
      if(sorted_degrees[i]<=-1*degree_threshold)
        high_degree_num++;
      else break;
    }

    IdType* const high_degree_rows = static_cast<IdType*>(
      device->AllocWorkspace(ctx, high_degree_num * sizeof(IdType)));
    IdType* const low_degree_rows = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows-high_degree_num) * sizeof(IdType)));
    for(int i=0;i<num_rows;i++){
      if(sorted_degrees[i]<=degree_threshold)
        high_degree_rows[i]=sorted_rows[i];
      else
        low_degree_rows[i-high_degree_num]=sorted_rows[i];
    }

    //block schedule
    IdType* const rows1 = static_cast<IdType*>(
      device->AllocWorkspace(ctx, high_degree_num * sizeof(IdType)));
    IdType* const rows2 = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows-high_degree_num) * sizeof(IdType)));
    constexpr int BLOCK_SIZE = 128;
    constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
    tempsample::processRows<IdType,TILE_SIZE>(
      high_degree_rows,rows1,high_degree_num,256);
    tempsample::processRows<IdType,TILE_SIZE>(
      low_degree_rows,rows2,num_rows-high_degree_num,256);
    device->FreeWorkspace(ctx, sorted_rows);
    device->FreeWorkspace(ctx, sorted_degrees);
    device->FreeWorkspace(ctx, high_degree_rows);
    device->FreeWorkspace(ctx, low_degree_rows);

    // 采样图
    NDArray sampled_rows1=NDArray::Empty(
        {high_degree_num}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCUDA, 0});
    CUDA_CALL(cudaMemcpyAsync(
      sampled_rows1->data, rows1, sizeof(IdType)*high_degree_num,
      cudaMemcpyHostToDevice, stream));
    device->FreeWorkspace(ctx, rows1);
    COOMatrix res1= _CSRRowWiseSamplingUniform<XPU, IdType>(
      mat, sampled_rows1, num_picks, replace, true);
    NDArray sampled_rows2=NDArray::Empty(
        {num_rows-high_degree_num}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCUDA, 0});
    CUDA_CALL(cudaMemcpyAsync(
      sampled_rows2->data, rows2, sizeof(IdType)*(num_rows-high_degree_num),
      cudaMemcpyHostToDevice, stream));
    device->FreeWorkspace(ctx, rows2);
    COOMatrix res2= _CSRRowWiseSamplingUniform<XPU, IdType>(
      mat, sampled_rows2, num_picks, replace, false);
    std::vector<COOMatrix> coos;
    coos.push_back(res1);
    coos.push_back(res2);

    //合并图
    return aten::UnionCoo(coos);
  }
}

template COOMatrix CSRRowWiseSamplingUniform<kDGLCUDA, int32_t>(
    CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDGLCUDA, int64_t>(
    CSRMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

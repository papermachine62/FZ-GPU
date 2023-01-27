/**
 * @file lorenzo_var.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-09-29
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef E2BEA52A_4D2E_4966_9135_6CE8B8E05762
#define E2BEA52A_4D2E_4966_9135_6CE8B8E05762

#include <cstddef>

#if __has_include(<cub/cub.cuh>)
// #pragma message __FILE__ ": (CUDA 11 onward), cub from system path"
#include <cub/cub.cuh>
#else
// #pragma message __FILE__ ": (CUDA 10 or earlier), cub from git submodule"
#include "../../external/cub/cub/cub.cuh"
#endif

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#define TIX threadIdx.x
#define TIY threadIdx.y
#define TIZ threadIdx.z
#define BIX blockIdx.x
#define BIY blockIdx.y
#define BIZ blockIdx.z
#define BDX blockDim.x
#define BDY blockDim.y
#define BDZ blockDim.z

#include "../utils/cuda_err.cuh"
#include "../utils/timer.hh"

namespace cusz {
namespace experimental {

template <typename Data, typename ErrCtrl, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void
pred1d(Data thread_scope[SEQ], volatile bool* shmem_signum, volatile ErrCtrl* shmem_delta, Data from_last_stripe = 0)
{
    if CONSTEXPR (FIRST_POINT) {  // i == 0
        Data delta                  = thread_scope[0] - from_last_stripe;
        ErrCtrl tmp = static_cast<ErrCtrl>(fabs(delta));
        tmp |= ((delta < 0) << 15);
        shmem_delta[0 + TIX * SEQ]  = tmp;
    }
    else {
#pragma unroll
        for (auto i = 1; i < SEQ; i++) {
            Data delta                  = thread_scope[i] - thread_scope[i - 1];
            shmem_signum[i + TIX * SEQ] = delta < 0;  // signum
            shmem_delta[i + TIX * SEQ]  = static_cast<ErrCtrl>(fabs(delta));
        }
        __syncthreads();
    }
}

template <typename Data, typename FP, int NTHREAD, int SEQ>
__forceinline__ __device__ void load1d(
    Data*          data,
    unsigned int   dimx,
    unsigned int   id_base,
    volatile Data* shmem_data,
    Data           thread_scope[SEQ],
    Data&          from_last_stripe,
    FP             ebx2_r)
{
#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
        auto id = id_base + TIX + i * NTHREAD;
        if (id < dimx) { shmem_data[TIX + i * NTHREAD] = round(data[id] * ebx2_r); }
    }
    __syncthreads();

    for (auto i = 0; i < SEQ; i++) thread_scope[i] = shmem_data[TIX * SEQ + i];

    if (TIX > 0) from_last_stripe = shmem_data[TIX * SEQ - 1];
    __syncthreads();
}

template <typename ErrCtrl, int NTHREAD, int SEQ>
__forceinline__ __device__ void write1d(
    volatile bool*    shmem_signum,
    bool*             signum,
    unsigned int      dimx,
    unsigned int      id_base,
    volatile ErrCtrl* shmem_delta = nullptr,
    ErrCtrl*          delta       = nullptr)
{
#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
        auto id = id_base + TIX + i * NTHREAD;
        if (id < dimx) {
            signum[id] = shmem_signum[TIX + i * NTHREAD];
            delta[id]  = shmem_delta[TIX + i * NTHREAD];
        }
    }
}

template <typename Data, typename FP, int YSEQ>
__forceinline__ __device__ void load2d_prequant(
    Data*        data,
    Data         center[YSEQ + 1],
    unsigned int dimx,
    unsigned int dimy,
    unsigned int stridey,
    unsigned int gix,
    unsigned int giy_base,
    FP           ebx2_r)
{
    auto get_gid = [&](auto i) { return (giy_base + i) * stridey + gix; };

#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
        if (gix < dimx and giy_base + i < dimy) center[i + 1] = round(data[get_gid(i)] * ebx2_r);
    }
    auto tmp = __shfl_up_sync(0xffffffff, center[YSEQ], 16);  // same-warp, next-16
    if (TIY == 1) center[0] = tmp;
}

template <typename Data, typename FP, int YSEQ>
__forceinline__ __device__ void pred2d(Data center[YSEQ + 1])
{
    /* prediction
         original form:  Data delta = center[i] - center[i - 1] + west[i] - west[i - 1];
            short form:  Data delta = center[i] - west[i];
       */
#pragma unroll
    for (auto i = YSEQ; i > 0; i--) {
        center[i] -= center[i - 1];
        auto west = __shfl_up_sync(0xffffffff, center[i], 1, 16);
        if (TIX > 0) center[i] -= west;
    }
    __syncthreads();
}

template <typename Data, typename ErrCtrl, int YSEQ>
__forceinline__ __device__ void postquant_write2d(
    Data         center[YSEQ + 1],
    ErrCtrl*     delta,
    bool*        signum,
    unsigned int dimx,
    unsigned int dimy,
    unsigned int stridey,
    unsigned int gix,
    unsigned int giy_base)
{
    /********************************************************************************
     * Depending on whether postquant is delayed in compression, deside separating
     * data-type signum and uint-type quantcode when writing to DRAM (or not).
     ********************************************************************************/
    auto get_gid = [&](auto i) { return (giy_base + i) * stridey + gix; };

#pragma unroll
    for (auto i = 1; i < YSEQ + 1; i++) {
        auto gid = get_gid(i - 1);

        if (gix < dimx and giy_base + i - 1 < dimy) {
            ErrCtrl tmp = static_cast<ErrCtrl>(fabs(center[i]));
            tmp |= ((center[i] < 0) << 15);
            delta[gid]  = tmp;
        }
    }
}

template <
    typename Data,
    typename ErrCtrl,
    typename FP,
    int BLOCK,
    int SEQ>
__global__ void c_lorenzo_1d1l(  //
    Data*    data,
    ErrCtrl* delta,
    bool*    signum,
    dim3     len3,
    dim3     stride3,
    FP       ebx2_r)
{
    constexpr auto NTHREAD = BLOCK / SEQ;

    __shared__ struct {
        Data    data[BLOCK];
        ErrCtrl delta[BLOCK];
        bool    signum[BLOCK];
    } shmem;

    auto id_base = BIX * BLOCK;

    Data thread_scope[SEQ];
    Data from_last_stripe{0};

    /********************************************************************************
     * load from DRAM using striped layout, perform prequant
     ********************************************************************************/
    load1d<Data, FP, NTHREAD, SEQ>(data, len3.x, id_base, shmem.data, thread_scope, from_last_stripe, ebx2_r);

    /********************************************************************************
     * delta and signum
     ********************************************************************************/
    pred1d<Data, ErrCtrl, SEQ, true>(thread_scope, shmem.signum, shmem.delta, from_last_stripe);
    pred1d<Data, ErrCtrl, SEQ, false>(thread_scope, shmem.signum, shmem.delta);
    write1d<ErrCtrl, NTHREAD, SEQ>(shmem.signum, signum, len3.x, id_base, shmem.delta, delta);
}

template <typename Data = float, typename ErrCtrl = uint16_t, typename FP = float>
__global__ void c_lorenzo_2d1l_16x16data_mapto16x2(
    Data*    data,    // input
    ErrCtrl* delta,   // output
    bool*    signum,  // output
    dim3     len3,
    dim3     stride3,
    FP       ebx2_r)
{
    constexpr auto BLOCK = 16;
    constexpr auto YSEQ  = 8;

    Data center[YSEQ + 1] = {0};  // nw  n
                                  //  w  center

    auto gix      = BIX * BDX + TIX;           // BDX == 16
    auto giy_base = BIY * BLOCK + TIY * YSEQ;  // BDY * YSEQ = BLOCK == 16
                                               // clang-format off
    load2d_prequant<Data, FP, YSEQ>(data, center, len3.x, len3.y, stride3.y, gix, giy_base, ebx2_r);
    pred2d<Data, FP, YSEQ>(center);
    postquant_write2d<Data, ErrCtrl, YSEQ >(center, delta, signum, len3.x, len3.y, stride3.y,  gix, giy_base);
    // clang-format on
}

template <typename Data, typename ErrCtrl = uint16_t, typename FP = float>
__global__ void c_lorenzo_3d1l_32x8x8data_mapto32x1x8(
    Data*    data,    // input
    ErrCtrl* delta,   // output
    bool*    signum,  // output
    dim3     len3,
    dim3     stride3,
    FP       ebx2_r)
{
    constexpr auto  BLOCK = 8;
    __shared__ Data shmem[8][8][32];

    auto z = TIZ;

    auto gix      = BIX * (BLOCK * 4) + TIX;
    auto giy_base = BIY * BLOCK;
    auto giz      = BIZ * BLOCK + z;
    auto base_id  = gix + giy_base * stride3.y + giz * stride3.z;

    /********************************************************************************
     * load from DRAM, perform prequant
     ********************************************************************************/
    if (gix < len3.x and giz < len3.z) {
        for (auto y = 0; y < BLOCK; y++) {
            if (giy_base + y < len3.y) {
                shmem[z][y][TIX] = round(data[base_id + y * stride3.y] * ebx2_r);  // prequant (fp presence)
            }
        }
    }
    __syncthreads();  // necessary to ensure correctness

    auto x = TIX % 8;

    for (auto y = 0; y < BLOCK; y++) {
        Data delta_val;

        // prediction
        delta_val = shmem[z][y][TIX] - ((z > 0 and y > 0 and x > 0 ? shmem[z - 1][y - 1][TIX - 1] : 0)  // dist=3
                                        - (y > 0 and x > 0 ? shmem[z][y - 1][TIX - 1] : 0)              // dist=2
                                        - (z > 0 and x > 0 ? shmem[z - 1][y][TIX - 1] : 0)              //
                                        - (z > 0 and y > 0 ? shmem[z - 1][y - 1][TIX] : 0)              //
                                        + (x > 0 ? shmem[z][y][TIX - 1] : 0)                            // dist=1
                                        + (y > 0 ? shmem[z][y - 1][TIX] : 0)                            //
                                        + (z > 0 ? shmem[z - 1][y][TIX] : 0));                          //

        auto id = base_id + (y * stride3.y);

        // delta and signum
        if (gix < len3.x and (giy_base + y) < len3.y and giz < len3.z) {
            ErrCtrl tmp = static_cast<ErrCtrl>(fabs(delta_val));
            tmp |= ((delta_val < 0) << 15);
            delta[id]  = tmp;
        }
    }
    /* EOF */
}

template <typename Data = float, typename ErrCtrl = uint16_t, typename FP = float, int BLOCK = 256, int SEQ = 8>
__global__ void x_lorenzo_1d1l(  //
    bool*    signum,
    ErrCtrl* delta,
    Data*    xdata,
    dim3     len3,
    dim3     stride3,
    FP       ebx2)
{
    constexpr auto block_dim = BLOCK / SEQ;  // dividable

    // coalesce-load (warp-striped) and transpose in shmem (similar for store)
    typedef cub::BlockLoad<bool, block_dim, SEQ, cub::BLOCK_LOAD_WARP_TRANSPOSE>    BlockLoadT_signum;
    typedef cub::BlockLoad<ErrCtrl, block_dim, SEQ, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT_delta;
    typedef cub::BlockStore<Data, block_dim, SEQ, cub::BLOCK_STORE_WARP_TRANSPOSE>  BlockStoreT_xdata;
    typedef cub::BlockScan<Data, block_dim, cub::BLOCK_SCAN_RAKING_MEMOIZE>
        BlockScanT_xdata;  // TODO autoselect algorithm

    __shared__ union TempStorage {  // overlap shared memory space
        typename BlockLoadT_signum::TempStorage load_signum;
        typename BlockLoadT_delta::TempStorage  load_delta;
        typename BlockStoreT_xdata::TempStorage store_xdata;
        typename BlockScanT_xdata::TempStorage  scan_xdata;
    } temp_storage;

    // thread-scope tiled data
    struct ThreadData {
        Data xdata[SEQ];
        bool signum[SEQ];
    } thread_scope;
    ErrCtrl thread_scope_delta[SEQ];

    /********************************************************************************
     * load to thread-private array (fuse at the same time)
     * (BIX * BDX * SEQ) denotes the start of the data chunk that belongs to this thread block
     ********************************************************************************/
    BlockLoadT_delta(temp_storage.load_delta).Load(delta + (BIX * BDX) * SEQ, thread_scope_delta);
    __syncthreads();  // barrier for shmem reuse
    BlockLoadT_signum(temp_storage.load_signum).Load(signum + (BIX * BDX) * SEQ, thread_scope.signum);
    __syncthreads();  // barrier for shmem reuse

#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
        auto id               = (BIX * BDX + TIX) * SEQ + i;
        thread_scope.xdata[i] = id < len3.x  //
                                    ? (thread_scope.signum[i] ? -1 : 1) * static_cast<Data>(thread_scope_delta[i])
                                    : 0;
    }
    __syncthreads();

    /********************************************************************************
     * perform partial-sum using cub::InclusiveSum
     ********************************************************************************/
    BlockScanT_xdata(temp_storage.scan_xdata).InclusiveSum(thread_scope.xdata, thread_scope.xdata);
    __syncthreads();  // barrier for shmem reuse

    /********************************************************************************
     * scale by ebx2 and write to DRAM
     ********************************************************************************/
#pragma unroll
    for (auto i = 0; i < SEQ; i++) thread_scope.xdata[i] *= ebx2;
    __syncthreads();  // barrier for shmem reuse

    BlockStoreT_xdata(temp_storage.store_xdata).Store(xdata + (BIX * BDX) * SEQ, thread_scope.xdata);
}

template <typename Data = float, typename ErrCtrl = uint16_t, typename FP = float>
__global__ void
x_lorenzo_2d1l_16x16data_mapto16x2(bool* signum, ErrCtrl* delta, Data* xdata, dim3 len3, dim3 stride3, FP ebx2)
{
    constexpr auto BLOCK = 16;
    constexpr auto YSEQ  = BLOCK / 2;  // sequentiality in y direction
    static_assert(BLOCK == 16, "In one case, we need BLOCK for 2D == 16");

    __shared__ Data intermediate[BLOCK];  // TODO use warp shuffle to eliminate this
    Data            thread_scope[YSEQ];
    /*
      .  ------> gix (x)
      |  t00    t01    t02    t03    ... t0f
      |  ts00_0 ts00_0 ts00_0 ts00_0
     giy ts00_1 ts00_1 ts00_1 ts00_1
     (y)  |      |      |      |
         ts00_7 ts00_7 ts00_7 ts00_7

      |  t10    t11    t12    t13    ... t1f
      |  ts00_0 ts00_0 ts00_0 ts00_0
     giy ts00_1 ts00_1 ts00_1 ts00_1
     (y)  |      |      |      |
         ts00_7 ts00_7 ts00_7 ts00_7
     */

    auto gix      = BIX * BLOCK + TIX;
    auto giy_base = BIY * BLOCK + TIY * YSEQ;  // BDY * YSEQ = BLOCK == 16
    auto get_gid  = [&](auto i) { return (giy_base + i) * stride3.y + gix; };

    /********************************************************************************
     * load to thread-private array (fuse at the same time)
     ********************************************************************************/
#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
        auto gid = get_gid(i);
        // even if we hit the else branch, all threads in a warp hit the y-boundary simultaneously
        if (gix < len3.x and giy_base + i < len3.y)
            thread_scope[i] = (signum[gid] ? -1 : 1) * static_cast<Data>(delta[gid]);  // fuse
        else
            thread_scope[i] = 0;  // TODO set as init state?
    }

    /********************************************************************************
     * partial-sum along y-axis, sequantially
     ********************************************************************************/
    for (auto i = 1; i < YSEQ; i++) thread_scope[i] += thread_scope[i - 1];
    // two-pass: store for cross-threadscope update
    if (TIY == 0) intermediate[TIX] = thread_scope[YSEQ - 1];
    __syncthreads();
    // two-pass: load and update
    if (TIY == 1) {
        auto tmp = intermediate[TIX];
#pragma unroll
        for (auto& i : thread_scope) i += tmp;
    }

    /********************************************************************************
     * in-warp partial-sum along x-axis
     ********************************************************************************/
#pragma unroll
    for (auto& i : thread_scope) {
        for (auto d = 1; d < BLOCK; d *= 2) {
            Data n = __shfl_up_sync(0xffffffff, i, d, 16);
            if (TIX >= d) i += n;
        }
        i *= ebx2;
    }

    /********************************************************************************
     * write to DRAM
     ********************************************************************************/
#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
        auto gid = get_gid(i);
        if (gix < len3.x and giy_base + i < len3.y) xdata[gid] = thread_scope[i];
    }
}

template <typename Data = float, typename ErrCtrl = uint16_t, typename FP = float>
__global__ void
x_lorenzo_3d1l_32x8x8data_mapto32x1x8(bool* signum, ErrCtrl* delta, Data* xdata, dim3 len3, dim3 stride3, FP ebx2)
{
    constexpr auto BLOCK = 8;
    constexpr auto YSEQ  = BLOCK;
    static_assert(BLOCK == 8, "In one case, we need BLOCK for 3D == 8");

    __shared__ Data intermediate[BLOCK][4][8];
    Data            thread_scope[YSEQ];

    auto seg_id  = TIX / 8;
    auto seg_tix = TIX % 8;

    auto gix = BIX * (4 * BLOCK) + TIX, giy_base = BIY * BLOCK, giz = BIZ * BLOCK + TIZ;
    auto get_gid = [&](auto y) { return giz * stride3.z + (giy_base + y) * stride3.y + gix; };

    /********************************************************************************
     * load to thread-private array (fuse at the same time)
     ********************************************************************************/
#pragma unroll
    for (auto y = 0; y < YSEQ; y++) {
        auto gid = get_gid(y);
        if (gix < len3.x and giy_base + y < len3.y and giz < len3.z)
            thread_scope[y] = (signum[gid] ? -1 : 1) * static_cast<Data>(delta[gid]);
        else
            thread_scope[y] = 0;
    }

    /********************************************************************************
     * partial-sum along y-axis, sequantially
     ********************************************************************************/
    for (auto y = 1; y < YSEQ; y++) thread_scope[y] += thread_scope[y - 1];

    /********************************************************************************
     * ND partial-sums along x- and z-axis
     * in-warp shuffle used: in order to perform, it's transposed after X-partial sum
     ********************************************************************************/
    auto dist = 1;
    Data addend;

#pragma unroll
    for (auto i = 0; i < BLOCK; i++) {
        Data val = thread_scope[i];

        for (dist = 1; dist < BLOCK; dist *= 2) {
            addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        // x-z transpose
        intermediate[TIZ][seg_id][seg_tix] = val;
        __syncthreads();
        val = intermediate[seg_tix][seg_id][TIZ];
        __syncthreads();

        for (dist = 1; dist < BLOCK; dist *= 2) {
            addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        intermediate[TIZ][seg_id][seg_tix] = val;
        __syncthreads();
        val = intermediate[seg_tix][seg_id][TIZ];
        __syncthreads();

        thread_scope[i] = val;
    }

    /********************************************************************************
     * write to DRAM
     ********************************************************************************/
#pragma unroll
    for (auto y = 0; y < YSEQ; y++) {
        if (gix < len3.x and giy_base + y < len3.y and giz < len3.z) { xdata[get_gid(y)] = thread_scope[y] * ebx2; }
    }
    /* EOF */
}

template <typename T, typename DeltaT, typename FP>
void launch_construct_LorenzoI_var(
    T*           data,
    DeltaT*      delta,
    bool*        signum,
    dim3 const   len3,
    double const eb,
    float&       time_elapsed,
    cudaStream_t stream)
{
    auto pardeg3 = [](dim3 len, dim3 sublen) {
        return dim3(
            (len.x - 1) / sublen.x + 1,  //
            (len.y - 1) / sublen.y + 1,  //
            (len.z - 1) / sublen.z + 1);
    };

    auto ndim = [&]() {
        if (len3.z == 1 and len3.y == 1)
            return 1;
        else if (len3.z == 1 and len3.y != 1)
            return 2;
        else
            return 3;
    };

    constexpr auto SUBLEN_1D = 256;
    constexpr auto SEQ_1D    = 4;  // x-sequentiality == 4
    constexpr auto BLOCK_1D  = dim3(256 / 4, 1, 1);
    auto           GRID_1D   = pardeg3(len3, SUBLEN_1D);

    constexpr auto SUBLEN_2D = dim3(16, 16, 1);
    // constexpr auto SEQ_2D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_2D = dim3(16, 2, 1);
    auto           GRID_2D  = pardeg3(len3, SUBLEN_2D);

    constexpr auto SUBLEN_3D = dim3(32, 8, 8);
    // constexpr auto SEQ_3D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_3D = dim3(32, 1, 8);
    auto           GRID_3D  = pardeg3(len3, SUBLEN_3D);

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = dim3(1, len3.x, len3.x * len3.y);

    auto outlier = data;

    cuda_timer_t timer;
    timer.timer_start(stream);

    if (ndim() == 1) {
        cusz::experimental::c_lorenzo_1d1l<T, DeltaT, FP, SUBLEN_1D, SEQ_1D>  //
            <<<GRID_1D, BLOCK_1D, 0, stream>>>                                //
            (data, delta, signum, len3, leap3, ebx2_r);
    }
    else if (ndim() == 2) {
        cusz::experimental::c_lorenzo_2d1l_16x16data_mapto16x2<T, DeltaT, FP>  //
            <<<GRID_2D, BLOCK_2D, 0, stream>>>                                 //
            (data, delta, signum, len3, leap3, ebx2_r);
    }
    else if (ndim() == 3) {
        cusz::experimental::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, DeltaT, FP>  //
            <<<GRID_3D, BLOCK_3D, 0, stream>>>                                    //
            (data, delta, signum, len3, leap3, ebx2_r);
    }
    else {
        throw std::runtime_error("Lorenzo only works for 123-D.");
    }

    timer.timer_end(stream);
    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());

    time_elapsed = timer.get_time_elapsed();
}

template <typename T, typename DeltaT, typename FP>
void launch_reconstruct_LorenzoI_var(
    bool*        signum,
    DeltaT*      delta,
    T*           xdata,
    dim3 const   len3,
    double const eb,
    float&       time_elapsed,
    cudaStream_t stream)
{
    auto pardeg3 = [](dim3 len, dim3 sublen) {
        return dim3(
            (len.x - 1) / sublen.x + 1,  //
            (len.y - 1) / sublen.y + 1,  //
            (len.z - 1) / sublen.z + 1);
    };

    auto ndim = [&]() {
        if (len3.z == 1 and len3.y == 1)
            return 1;
        else if (len3.z == 1 and len3.y != 1)
            return 2;
        else
            return 3;
    };

    constexpr auto SUBLEN_1D = 256;
    constexpr auto SEQ_1D    = 8;  // x-sequentiality == 8
    constexpr auto BLOCK_1D  = dim3(256 / 8, 1, 1);
    auto           GRID_1D   = pardeg3(len3, SUBLEN_1D);

    constexpr auto SUBLEN_2D = dim3(16, 16, 1);
    // constexpr auto SEQ_2D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_2D = dim3(16, 2, 1);
    auto           GRID_2D  = pardeg3(len3, SUBLEN_2D);

    constexpr auto SUBLEN_3D = dim3(32, 8, 8);
    // constexpr auto SEQ_3D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_3D = dim3(32, 1, 8);
    auto           GRID_3D  = pardeg3(len3, SUBLEN_3D);

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = dim3(1, len3.x, len3.x * len3.y);

    auto outlier = xdata;

    cuda_timer_t timer;
    timer.timer_start(stream);

    if (ndim() == 1) {
        cusz::experimental::x_lorenzo_1d1l<T, DeltaT, FP, 256, 8>  //
            <<<GRID_1D, BLOCK_1D, 0, stream>>>                     //
            (signum, delta, xdata, len3, leap3, ebx2);
    }
    else if (ndim() == 2) {
        cusz::experimental::x_lorenzo_2d1l_16x16data_mapto16x2<T, DeltaT, FP>  //
            <<<GRID_2D, BLOCK_2D, 0, stream>>>                                 //
            (signum, delta, xdata, len3, leap3, ebx2);
    }
    else {
        cusz::experimental::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, DeltaT, FP>  //
            <<<GRID_3D, BLOCK_3D, 0, stream>>>                                    //
            (signum, delta, xdata, len3, leap3, ebx2);
    }

    timer.timer_end(stream);
    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());

    time_elapsed = timer.get_time_elapsed();
}

//
}  // namespace experimental
}  // namespace cusz

#undef TIX
#undef TIY
#undef TIZ
#undef BIX
#undef BIY
#undef BIZ
#undef BDX
#undef BDY
#undef BDZ

#endif /* E2BEA52A_4D2E_4966_9135_6CE8B8E05762 */

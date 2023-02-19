#include <cuda_runtime.h>
#include <dirent.h>
#include <stdint.h>
#include <sys/stat.h>
#include <thrust/copy.h>
#include <chrono>
#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "include/kernel/lorenzo_var.cuh"
#include "include/utils/cuda_err.cuh"
#include "include/utils/io.hh"

#define UINT32_BIT_LEN 32


long GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int         rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

template <typename T>
T* read_binary_to_new_array(const std::string& fname, size_t dtype_len)
{
    std::ifstream ifs(fname.c_str(), std::ios::binary | std::ios::in);
    if (not ifs.is_open()) {
        std::cerr << "fail to open " << fname << std::endl;
        exit(1);
    }
    auto _a = new T[dtype_len]();
    ifs.read(reinterpret_cast<char*>(_a), std::streamsize(dtype_len * sizeof(T)));
    ifs.close();
    return _a;
}

template <typename T>
void write_array_to_binary(const std::string& fname, T* const _a, size_t const dtype_len)
{
    std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::out);
    if (not ofs.is_open()) return;
    ofs.write(reinterpret_cast<const char*>(_a), std::streamsize(dtype_len * sizeof(T)));
    ofs.close();
}

__global__ void encodeDebug(int blockSize, uint8_t* d_in, uint8_t* d_out, uint32_t* preSum)
{
    __shared__ uint32_t sumArr[33];
    int                 tid = threadIdx.x + blockIdx.x * blockDim.x;
    sumArr[0]               = preSum[tid];
    sumArr[threadIdx.x + 1] = preSum[tid + 1];
    __syncthreads();
    if (sumArr[threadIdx.x + 1] != sumArr[threadIdx.x]) {
        for (int i = 0; i < blockSize; i++) { d_out[sumArr[threadIdx.x] * blockSize + i] = d_in[tid * blockSize + i]; }
    }
}

__global__ void bitshuffleAndBitflag(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out,
    int       blockSize,
    uint32_t* d_bitFlagArray,
    uint32_t*  d_byteFlagArray)
{
    __shared__ uint32_t smem[32][33];
    __shared__ uint32_t bitflagArr[8];
    __shared__ uint32_t  byteFlagArray[256];

    uint32_t            v;
    v = in[threadIdx.x + threadIdx.y * 32 + blockIdx.x * 1024];
    
#pragma unroll 32
    for (int i = 0; i < 32; i++)
    {
        smem[threadIdx.y][i] = __ballot_sync(0xFFFFFFFFU, v & (1U << i));
    }
    __syncthreads(); 

    out[threadIdx.x + threadIdx.y * 32 + blockIdx.x * 1024] = smem[threadIdx.x][threadIdx.y];

    if (threadIdx.x < 8)
    {
        for (int i = 1; i < 4; i++)
        { 
            smem[threadIdx.x * 4][threadIdx.y] |= smem[threadIdx.x * 4 + i][threadIdx.y];
        }
        byteFlagArray[threadIdx.y * 8 + threadIdx.x] = (smem[threadIdx.x * 4][threadIdx.y] > 0);
    }
    __syncthreads();

    uint32_t buffer;
    if (threadIdx.y < 8)
    {
        buffer = byteFlagArray[threadIdx.y * 32 + threadIdx.x];
        bitflagArr[threadIdx.y] = __ballot_sync(0xFFFFFFFFU, buffer);
    }
    __syncthreads();

    if (threadIdx.y < 8)
    {
        d_byteFlagArray[blockIdx.x * 256 + threadIdx.y * 32 + threadIdx.x] = byteFlagArray[threadIdx.y * 32 + threadIdx.x];
    }

    if (threadIdx.x < 8 && threadIdx.y == 0)
    {
        d_bitFlagArray[blockIdx.x * 8 + threadIdx.x] = bitflagArr[threadIdx.x];
    }
}

void runSzs(std::string fileName, int x, int y, int z, double eb)
{
    // prepare for pre-quantization
    auto len3     = dim3(x, y, z);
    int  fileSize = GetFileSize(fileName);
    auto len      = int(fileSize / sizeof(float));

    float*    data;
    float*    h_data;
    bool*     signum;
    uint16_t* quant;
    float     time_elapsed;
    uint16_t* bitshuffleOut;

    uint32_t* d_bitFlagArray;
    uint32_t*  d_byteFlagArray;
    uint8_t* d_out;
    uint32_t* d_preSumArray;
    uint32_t* d_byteFlagArrayTest;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    h_data = read_binary_to_new_array<float>(fileName, len);
    float range = *std::max_element(h_data , h_data + len) - *std::min_element(h_data , h_data + len);

    CHECK_CUDA(cudaMalloc((void**)&data, sizeof(float) * len));
    CHECK_CUDA(cudaMalloc((void**)&quant, sizeof(uint16_t) * len));
    CHECK_CUDA(cudaMalloc((void**)&signum, sizeof(bool) * len));

    CHECK_CUDA(cudaMemcpy(data, h_data, sizeof(float) * len, cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemset(quant, 0, sizeof(uint16_t) * len));
    

    // launch pre-quantization kernel
    cusz::experimental::launch_construct_LorenzoI_var<float, uint16_t, float>(data, quant, signum, len3, eb * range, time_elapsed, stream);

    CHECK_CUDA(cudaFree(data));
    CHECK_CUDA(cudaFree(signum));

    CHECK_CUDA(cudaMalloc((void**)&bitshuffleOut, sizeof(uint16_t) * len));
    CHECK_CUDA(cudaMemcpy(bitshuffleOut, quant, sizeof(uint16_t) * len, cudaMemcpyDeviceToDevice));

    int  blockSize    = 16;
    auto newLen       = len * 2;  // bitshuffle result length in byte unit
    newLen            = newLen % 8192 == 0 ? newLen : newLen - newLen % 8192 + 8192;
    int dataChunkSize = newLen % (blockSize * UINT32_BIT_LEN) == 0 ? newLen / (blockSize * UINT32_BIT_LEN)
                                                                   : int(newLen / (blockSize * UINT32_BIT_LEN)) + 1;
    
    CHECK_CUDA(cudaMalloc((void**)&d_bitFlagArray, sizeof(uint32_t) * dataChunkSize));
    CHECK_CUDA(cudaMalloc((void**)&d_byteFlagArray, sizeof(uint32_t) * dataChunkSize * UINT32_BIT_LEN));
    CHECK_CUDA(cudaMalloc((void**)&d_out, sizeof(uint8_t) * dataChunkSize * blockSize * UINT32_BIT_LEN));

    CHECK_CUDA(cudaMemset(d_bitFlagArray, 0, sizeof(uint32_t) * dataChunkSize));
    CHECK_CUDA(cudaMemset(d_byteFlagArray, 0, sizeof(uint32_t) * dataChunkSize * UINT32_BIT_LEN));
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(uint8_t) * dataChunkSize * blockSize * UINT32_BIT_LEN));

    // launch bitshuffle kernel
    dim3 threads(32, 32);
    dim3 grid(floor(len / 2048));  
    bitshuffleAndBitflag<<<grid, threads>>>((uint32_t*)quant, (uint32_t*)bitshuffleOut, blockSize, d_bitFlagArray, d_byteFlagArray);
    CHECK_CUDA(cudaFree(quant));

    newLen = len * 2;
    dataChunkSize = newLen % (blockSize * UINT32_BIT_LEN) == 0 ? newLen / (blockSize * UINT32_BIT_LEN) : int(newLen / (blockSize * UINT32_BIT_LEN)) + 1;
    
    CHECK_CUDA(cudaMalloc((void**)&d_preSumArray, sizeof(uint32_t) * dataChunkSize * UINT32_BIT_LEN + sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_preSumArray, 0, sizeof(uint32_t) * dataChunkSize * UINT32_BIT_LEN + sizeof(uint32_t)));
   
    CHECK_CUDA(cudaMalloc((void**)&d_byteFlagArrayTest, sizeof(uint32_t) * dataChunkSize * UINT32_BIT_LEN));
    CHECK_CUDA(cudaMemcpy(d_byteFlagArrayTest, d_byteFlagArray, sizeof(uint32_t) * dataChunkSize * UINT32_BIT_LEN, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaFree(d_byteFlagArray));

    // prefix sum with CUB library implementation
    void*  d_temp_storage     = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_byteFlagArrayTest, d_preSumArray, dataChunkSize * UINT32_BIT_LEN);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_byteFlagArrayTest, d_preSumArray, dataChunkSize * UINT32_BIT_LEN);
    
    uint32_t* lastSum = (uint32_t*)malloc(sizeof(uint32_t));
    CHECK_CUDA(cudaMemcpy(lastSum, d_preSumArray + dataChunkSize * UINT32_BIT_LEN - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t* lastFlag = (uint32_t*)malloc(sizeof(uint32_t));
    CHECK_CUDA(cudaMemcpy(lastFlag, d_byteFlagArrayTest + dataChunkSize * UINT32_BIT_LEN - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t* result = (uint32_t*)malloc(sizeof(uint32_t));
    *result = *lastSum + *lastFlag;

    CHECK_CUDA(cudaMemcpy(d_preSumArray + dataChunkSize * UINT32_BIT_LEN, result, sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFree(d_byteFlagArrayTest));
    CHECK_CUDA(cudaFree(d_temp_storage));

    encodeDebug<<<dataChunkSize, 32>>>(blockSize, (uint8_t*)bitshuffleOut, d_out, d_preSumArray);  

    printf("original size: %d\n", fileSize);
    printf("compressed size: %d\n", sizeof(uint32_t) * dataChunkSize + blockSize * (*result));
    printf("compression ratio: %f\n", float(fileSize) / float(sizeof(uint32_t) * dataChunkSize + blockSize * (*result)));

    cudaStreamDestroy(stream);
    CHECK_CUDA(cudaFree(bitshuffleOut));
    CHECK_CUDA(cudaFree(d_bitFlagArray));
    CHECK_CUDA(cudaFree(d_preSumArray));
    CHECK_CUDA(cudaFree(d_out));

    delete[] h_data;

    free(result);
    free(lastSum);
    free(lastFlag);

    return;
}

int main(int argc, char* argv[])
{
    using T = float;
    std::string fileName;
    fileName  = std::string(argv[1]);
    int    x  = std::stoi(argv[2]);
    int    y  = std::stoi(argv[3]);
    int    z  = std::stoi(argv[4]);
    double eb = std::stod(argv[5]);

    runSzs(fileName, x, y, z, eb);
    return 0;
}

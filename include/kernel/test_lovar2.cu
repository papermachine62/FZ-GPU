/**
 * @file test_lorenzo_var.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-09-29
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <iostream>
#include <string>

#include "../cli/quality_viewer.hh"
#include "../utils/cuda_err.cuh"
#include "../utils/io.hh"
#include "lorenzo_var.cuh"

using std::cerr;
using std::cout;
using std::endl;

template <typename DeltaT = uint16_t>
int f(std::string fname, size_t x, size_t y, size_t z, double eb, size_t start = 10000)
{
    float*  h_data;
    float*  data;
    float*  xdata;
    bool*   signum;
    DeltaT* delta;

    dim3   len3    = dim3(x, y, z);
    dim3   stride3 = dim3(1, x, x * y);
    size_t len     = x * y * z;

    cudaMallocHost(&h_data, len * sizeof(float));

    cudaMalloc(&data, len * sizeof(float));
    cudaMalloc(&xdata, len * sizeof(float));
    cudaMalloc(&signum, len * sizeof(bool));
    cudaMalloc(&delta, len * sizeof(DeltaT));

    io::read_binary_to_array<float>(fname, h_data, len);
    cudaMemcpy(data, h_data, len * sizeof(float), cudaMemcpyHostToDevice);

    {
        printf("data\n");
        thrust::for_each(thrust::device, data + start, data + start + 20, [=] __device__ __host__(const float i) {
            printf("%.3e\t", i);
        });
        printf("\n");
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float time_comp;

    cusz::experimental::launch_construct_LorenzoI_var<float, DeltaT, float>(
        data, delta, signum, len3, eb, time_comp, stream);

    {
        printf("signum\n");
        thrust::for_each(thrust::device, signum + start, signum + start + 20, [=] __device__ __host__(const bool i) {
            printf("%d\t", (int)i);
        });
        printf("\n");
        printf("delta\n");
        thrust::for_each(thrust::device, delta + start, delta + start + 20, [=] __device__ __host__(const DeltaT i) {
            printf("%u\t", (uint32_t)i);
        });
        printf("\n");
    }

    cout << "comp time\t" << time_comp << endl;

    float time_decomp;
    cusz::experimental::launch_reconstruct_LorenzoI_var<float, DeltaT, float>(
        signum, delta, xdata, len3, eb, time_decomp, stream);

    cout << "decomp time\t" << time_decomp << endl;

    {
        printf("xdata\n");
        thrust::for_each(thrust::device, xdata + start, xdata + start + 20, [=] __device__ __host__(const float i) {
            printf("%.3e\t", i);
        });
        printf("\n");
    }

    /* perform evaluation */ cusz::QualityViewer::echo_metric_gpu(xdata, data, len);

    cudaFreeHost(h_data);
    cudaFree(data);
    cudaFree(xdata);
    cudaFree(signum);
    cudaFree(delta);

    cudaStreamDestroy(stream);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 5) {
        cout << "                       default: ui16" << endl;
        cout << "                       ui8,ui16,ui32" << endl;
        cout << "PROG fname x y z [eb] [delta type] [print offset]" << endl;
        cout << "0    1     2 3 4 [5]  [6]          [7]" << endl;

        return 1;
    }

    auto fname = std::string(argv[1]);
    auto x     = atoi(argv[2]);
    auto y     = atoi(argv[3]);
    auto z     = atoi(argv[4]);

    double      eb          = 1e-4;
    std::string delta_type  = "ui16";
    size_t      print_start = 10000;

    if (argc >= 6) eb = atof(argv[5]);

    if (argc >= 7) delta_type = std::string(argv[6]);

    if (argc >= 8) print_start = atoi(argv[7]);

    if (delta_type == "ui8")
        f<uint8_t>(fname, x, y, z, eb, print_start);
    else if (delta_type == "ui16")
        f<uint16_t>(fname, x, y, z, eb, print_start);
    else if (delta_type == "ui32")
        f<uint32_t>(fname, x, y, z, eb, print_start);

    return 0;
}
/**
 * @file sp_path.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-29
 * meged file created on 2021-06-06
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_SP_PATH_CUH
#define CUSZ_SP_PATH_CUH

#include "base_compressor.cuh"
#include "binding.hh"
#include "componments.hh"

/******************************************************************************
                            macros for shorthand writing
 ******************************************************************************/

#define D2D_CPY(VAR, FIELD)                                                                            \
    {                                                                                                  \
        auto dst = d_reserved_compressed + header.entry[HEADER::FIELD];                                \
        auto src = reinterpret_cast<BYTE*>(d_##VAR);                                                   \
        CHECK_CUDA(cudaMemcpyAsync(dst, src, nbyte[HEADER::FIELD], cudaMemcpyDeviceToDevice, stream)); \
    }

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_compressed + header.entry[HEADER::SYM])

/******************************************************************************
                               class definition
******************************************************************************/

template <class BINDING>
class SpPathCompressor : public BaseCompressor<typename BINDING::PREDICTOR> {
    using Predictor = typename BINDING::PREDICTOR;
    using Spcodec   = typename BINDING::SPCODEC;

    using T    = typename Predictor::Origin;   // wrong in type inference
    using E    = typename Predictor::ErrCtrl;  // wrong in type inference
    using BYTE = uint8_t;                      // non-interpreted type; bytestream

    unsigned int len;
    dim3         data_size;

    Predictor* predictor;
    Spcodec*   spcodec;

    BYTE* d_reserved_compressed{nullptr};

   public:
    struct header_t {
        static const int HEADER = 0;
        static const int ANCHOR = 1;
        static const int SPFMT  = 2;
        static const int END    = 3;

        int      header_nbyte : 16;
        uint32_t entry[END + 1];

        uint32_t file_size() const { return entry[END]; }
    };
    using HEADER = struct header_t;

   public:
    uint32_t get_len_data() { return data_size.x * data_size.y * data_size.z; }
    uint32_t get_len_anchor() { return predictor->get_len_anchor(); }

   private:
   public:
    SpPathCompressor() = default;

    /**
     * @brief Given the input data size, determine the internal memory footprint and allocate accordingly.
     *
     * @param xyz 3D unsigned integer
     */
    void init(dim3 xyz, int dummy_coarse_pardeg = -1, int sp_factor = 4, bool dbg_print = false)
    {
        predictor = new Predictor;
        spcodec   = new Spcodec;

        data_size = xyz;

        (*predictor).init(xyz, dbg_print);

        // TODO encapsulate more
        auto spcodec_in_len = (*predictor).get_alloclen_quant();

        (*spcodec).init(spcodec_in_len, sp_factor, dbg_print);

        CHECK_CUDA(cudaMalloc(&d_reserved_compressed, (*predictor).get_len_data() * sizeof(T) / 2));
    }

    /**
     * @brief
     *
     * @param uncompressed device pointer, uncompressed input
     * @param eb host variable, error bound
     * @param radius host variable, in many cases, it is 0
     * @param compressed device pointer, modified inside the method, output cusz archive binary
     * @param compressed_len host variable, modified inside the method, output compressed length
     * @param stream CUDA stream
     */
    void compress(
        T*           uncompressed,
        double const eb,
        int const    radius,
        BYTE*&       compressed,
        size_t&      compressed_len,
        cudaStream_t stream = nullptr)
    {
        T*     d_anchor{nullptr};
        E*     d_errctrl{nullptr};
        BYTE*  d_spcodec_out{nullptr};
        size_t spcodec_out_len{0};

        HEADER header;

        auto subfile_collect = [&]() {
            header.header_nbyte = sizeof(HEADER);

            uint32_t nbyte[HEADER::END];
            nbyte[HEADER::HEADER] = 128;
            nbyte[HEADER::ANCHOR] = sizeof(T) * (*predictor).get_len_anchor();
            nbyte[HEADER::SPFMT]  = sizeof(BYTE) * spcodec_out_len;

            header.entry[0] = 0;
            // *.END + 1; need to know the ending position
            for (auto i = 1; i < HEADER::END + 1; i++) { header.entry[i] = nbyte[i - 1]; }
            for (auto i = 1; i < HEADER::END + 1; i++) { header.entry[i] += header.entry[i - 1]; }

            auto debug_header_entry = [&]() {
                printf("\nsubfile collect in compressor:\n");
                for (auto i = 0; i < HEADER::END + 1; i++) printf("%d, header entry: %d\n", i, header.entry[i]);
            };
            debug_header_entry();

            CHECK_CUDA(cudaMemcpyAsync(d_reserved_compressed, &header, sizeof(header), cudaMemcpyHostToDevice, stream));

            D2D_CPY(anchor, ANCHOR)
            D2D_CPY(spcodec_out, SPFMT)

            /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));
        };

        (*predictor).construct(data_size, uncompressed, d_anchor, d_errctrl, eb, radius, stream);
        auto spcodec_in_len = (*predictor).get_alloclen_quant();
        (*spcodec).encode(d_errctrl, spcodec_in_len, d_spcodec_out, spcodec_out_len, stream);

        /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

        subfile_collect();
        compressed_len = header.file_size();
        compressed     = d_reserved_compressed;

        auto compress_report = [&]() {
            auto get_cr        = [&]() { return get_len_data() * sizeof(T) * 1.0 / compressed_len; };
            auto byte_to_gbyte = [&](double bytes) { return bytes / 1024 / 1024 / 1024; };
            auto ms_to_s       = [&](double ms) { return ms / 1000; };

            auto bytes = get_len_data() * sizeof(T);

            auto time_p = predictor->get_time_elapsed();
            auto tp_p   = byte_to_gbyte(bytes) / ms_to_s(time_p);

            auto time_s = spcodec->get_time_elapsed();
            auto tp_s   = byte_to_gbyte(bytes) / ms_to_s(time_s);

            auto tp_total = byte_to_gbyte(bytes) / ms_to_s(time_p + time_s);

            printf("\n(c) COMPRESSION REPORT\n");
            printf("%-*s: %.2f\n", 20, "compression ratio", get_cr());
            printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "predictor time", time_p, tp_p);
            printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "spcodec time", time_s, tp_s);
            printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "total time", time_p + time_s, tp_total);
            printf("\n");
        };

        compress_report();
    }

    /**
     * @brief
     *
     */
    void clear_buffer()
    {  //
        (*predictor).clear_buffer();
        (*spcodec).clear_buffer();
    }

    /**
     * @brief High-level decompress method for this compressor
     *
     * @param in_compressed device pointer, the cusz archive bianry
     * @param eb host variable, error bound
     * @param radius host variable, in this case, it is 0
     * @param out_decompressed device pointer, output decompressed data
     * @param stream CUDA stream
     */
    void decompress(
        BYTE*        in_compressed,
        double const eb,
        int const    radius,
        T*           out_decompressed,
        cudaStream_t stream = nullptr)
    {
        HEADER header;
        CHECK_CUDA(cudaMemcpyAsync(&header, in_compressed, sizeof(header), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        auto d_anchor     = ACCESSOR(ANCHOR, T);
        auto d_spcodec_in = ACCESSOR(SPFMT, BYTE);

        auto d_errctrl = (*predictor).expose_quant();  // reuse

        (*spcodec).decode(d_spcodec_in, d_errctrl, stream);
        (*predictor).reconstruct(data_size, d_anchor, d_errctrl, eb, radius, out_decompressed, stream);

        auto decompress_report = [&]() {
            auto byte_to_gbyte = [&](double bytes) { return bytes / 1024 / 1024 / 1024; };
            auto ms_to_s       = [&](double ms) { return ms / 1000; };

            auto bytes = get_len_data() * sizeof(T);

            auto time_p = predictor->get_time_elapsed();
            auto tp_p   = byte_to_gbyte(bytes) / ms_to_s(time_p);

            auto time_s = spcodec->get_time_elapsed();
            auto tp_s   = byte_to_gbyte(bytes) / ms_to_s(time_s);

            auto tp_total = byte_to_gbyte(bytes) / ms_to_s(time_p + time_s);

            printf("\n(d) deCOMPRESSION REPORT\n");
            printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "spcodec time", time_s, tp_s);
            printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "predictor time", time_p, tp_p);
            printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "total time", time_p + time_s, tp_total);
            printf("\n");
        };

        decompress_report();
    }

    ~SpPathCompressor()
    {
        delete predictor;
        delete spcodec;
    }
};

/******************************************************************************
                              config with defaults
******************************************************************************/

struct SparsityAwarePath {
   private:
    using DATA    = DataTrait<4>::type;
    using ERRCTRL = ErrCtrlTrait<4, true>::type;
    using FP      = FastLowPrecisionTrait<true>::type;

   public:
    using DefaultBinding = PredictorReducerBinding<  //
        cusz::Spline3<DATA, ERRCTRL, FP>,
        cusz::CSR11<ERRCTRL>>;

    using DefaultCompressor      = class SpPathCompressor<DefaultBinding>;
    using LorenzoBasedCompressor = class SpPathCompressor<DefaultBinding>;

    using FallbackBinding = PredictorReducerBinding<  //
        cusz::PredictionUnified<DATA, ERRCTRL, FP>,
        cusz::CSR11<ERRCTRL>>;

    using FallbackCompressor = class SpPathCompressor<FallbackBinding>;
};

#undef ACCESSOR
#undef D2D_CPY

#endif

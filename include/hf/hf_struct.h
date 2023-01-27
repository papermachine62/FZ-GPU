/**
 * @file hf_struct.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-09-14
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef DA6883A3_A70F_4690_A4FA_56644987725A
#define DA6883A3_A70F_4690_A4FA_56644987725A

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

// raw pointer array; regardless of being on host or device
typedef struct hf_book {
    uint32_t* freq;
    // undertermined on definition; could be uint32_t* and uint64_t*
    void* book;
    int   booklen;
} hf_book;

typedef struct hf_revbook {
} hf_revbook;

typedef struct hf_chunk {
    void* bits;     // how many bits each chunk
    void* cells;    // how many cells each chunk
    void* entries;  // jump to the chunk
} hf_chunk;

typedef struct hf_bitstream {
    void*     buffer;
    void*     bitstream;
    hf_chunk* d_metadata;
    hf_chunk* h_metadata;
    int       sublen;  // data chunksize
    int       pardeg;  // runtime paralleism degree
    int       numSMs;  // number of streaming multiprocessor
} hf_bitstream;

#ifdef __cplusplus
}
#endif

#endif /* DA6883A3_A70F_4690_A4FA_56644987725A */

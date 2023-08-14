/******************************************************************************
 * The following codes are modified from the original FlashAttn library: https://github.com/Dao-AILab/flash-attention
 ******************************************************************************/

// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<> void run_mha_fwd_<cutlass::bfloat16_t, 256>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256<cutlass::bfloat16_t>(params, stream);
}

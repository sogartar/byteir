// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up
// compilation.

#include "flash_fwd_launch_template.h"
namespace brt {
namespace cuda {
namespace kernel {
// template<>
// void run_mha_fwd_<cutlass::bfloat16_t, 192>(Flash_fwd_params &params,
// cudaStream_t stream) {
//     using elem_type = cutlass::bfloat16_t;
//     BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
//         run_flash_fwd<Flash_fwd_kernel_traits<192, 64, 64, 4, false, false,
//         elem_type>, Is_dropout>(params, stream);
//     });
// }
template <>
void run_mha_fwd_<cutlass::bfloat16_t, 192>(Flash_fwd_params &params,
                                            cudaStream_t stream) {
  run_mha_fwd_hdim192<cutlass::bfloat16_t>(params, stream);
}
} // namespace kernel
} // namespace cuda
} // namespace brt
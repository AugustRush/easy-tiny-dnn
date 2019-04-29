//
//  fully_connected_op_apple.h
//  tiny-dnn
//
//  Created by pingwei liu on 2019/4/29.
//  Copyright © 2019 pingwei liu. All rights reserved.
//

#pragma once

#include "fully_params.h"
#include "apple_math.h"

namespace tiny_dnn {
    namespace kernels {
        
        inline void fully_connected_op_apple(const tensor_t &in_data,
                                             const vec_t &W,
                                             const vec_t &bias,
                                             tensor_t &out_data,
                                             const core::fully_params &params,
                                             const bool layer_parallelize) {
            for_i(layer_parallelize, in_data.size(), [&](size_t sample) {
                const vec_t &in = in_data[sample];
                vec_t &out      = out_data[sample];
                int in_size = static_cast<int>(params.in_size_);
                int out_size = static_cast<int>(params.out_size_);
                apple_mmul(out.data(), in.data(), W.data(), 1, out_size, in_size);
                if (params.has_bias_) {
                    apple_vadd(out.data(), out.data(), bias.data(), out_size);
                }
            });
        }
        
        inline void fully_connected_op_apple(const tensor_t &prev_out,
                                             const vec_t &W,
                                             tensor_t &dW,
                                             tensor_t &db,
                                             tensor_t &curr_delta,
                                             tensor_t &prev_delta,
                                             const core::fully_params &params,
                                             const bool layer_parallelize) {
            for (size_t sample = 0; sample < prev_out.size(); sample++) {
                int in_size = static_cast<int>(params.in_size_);
                int out_size = static_cast<int>(params.out_size_);
                apple_sgemm(prev_delta[sample].data(), W.data(), 111, curr_delta[sample].data(), 111, in_size, 1, out_size, out_size, 1, 1, 1, 1);

                for_(layer_parallelize, 0, params.out_size_, [&](const blocked_range &r) {
                    // accumulate weight-step using delta
                    // dW[c * out_size + i] += current_delta[i] * prev_out[c]
                    int size = static_cast<int>(r.end() - r.begin());
                    apple_sgemm(dW[sample].data(), curr_delta[sample].data(), 111, prev_out[sample].data(), 111, size, in_size, 1, 1, in_size, in_size, 1, 1);
                    //
                    if (params.has_bias_) {
                        // vec_t& db = *in_grad[2];
                        apple_vadd(db[sample].data(), db[sample].data(), curr_delta[sample].data(), size);
                    }
                });
            }
        }
        
    }  // namespace kernels
}  // namespace tiny_dnn


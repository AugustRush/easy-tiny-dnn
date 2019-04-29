/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>
#include <utility>

#include "activation_layer.h"
#include "layer.h"

namespace tiny_dnn {

class tanh_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  std::string layer_type() const override { return "tanh-activation"; }

  void forward_activation(const vec_t &x, vec_t &y) override {
#ifdef CNN_USE_APPLE_ACCELERATE
      apple_tanh(y.data(), x.data(), (int)x.size());
#else
    for (size_t j = 0; j < x.size(); j++) {
      y[j] = std::tanh(x[j]);
    }
#endif
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) override {
#ifdef CNN_USE_APPLE_ACCELERATE
      int size = static_cast<int>(x.size());
      apple_vsq(dx.data(), y.data(), size);
      apple_vneg(dx.data(), dx.data(), size);
      apple_vsadd(dx.data(), dx.data(), 1.f, size);
      apple_vmul(dx.data(), dx.data(), dy.data(), size);
#else
    for (size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of tanh)
      dx[j] = dy[j] * (float_t(1) - sqr(y[j]));
    }
#endif
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(-0.8), float_t(0.8));
  }

  friend struct serialization_buddy;
};

}  // namespace tiny_dnn

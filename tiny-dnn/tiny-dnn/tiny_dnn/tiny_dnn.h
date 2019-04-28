/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "config.h"
#include "network.h"
#include "nodes.h"

#include "tensor.h"

#include "device.h"
#include "program_manager.h"

#include "asinh_layer.h"
#include "elu_layer.h"
#include "leaky_relu_layer.h"
#include "relu_layer.h"
#include "selu_layer.h"
#include "sigmoid_layer.h"
#include "softmax_layer.h"
#include "softplus_layer.h"
#include "softsign_layer.h"
#include "tanh_layer.h"
#include "tanh_p1m2_layer.h"
#include "arithmetic_layer.h"
#include "average_pooling_layer.h"
#include "average_unpooling_layer.h"
#include "batch_normalization_layer.h"
#include "cell.h"
#include "cells.h"
#include "concat_layer.h"
#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "dropout_layer.h"
#include "fully_connected_layer.h"
#include "global_average_pooling_layer.h"
#include "input_layer.h"
#include "l2_normalization_layer.h"
#include "lrn_layer.h"
#include "lrn_layer.h"
#include "max_pooling_layer.h"
#include "max_unpooling_layer.h"
#include "power_layer.h"
#include "quantized_convolutional_layer.h"
#include "quantized_deconvolutional_layer.h"
#include "recurrent_layer.h"
#include "slice_layer.h"
#include "zero_pad_layer.h"
#include "files.h"

#ifdef CNN_USE_GEMMLOWP
#include "quantized_fully_connected_layer.h"
#endif  // CNN_USE_GEMMLOWP

#include "lossfunctions/loss_function.h"
#include "optimizers/optimizer.h"

#include "deform.h"
#include "graph_visualizer.h"
#include "product.h"
#include "weight_init.h"
#include "nms.h"

#include "io/cifar10_parser.h"
#include "io/display.h"
#include "io/layer_factory.h"
#include "io/mnist_parser.h"

#ifdef DNN_USE_IMAGE_API
#include "image.h"
#endif  // DNN_USE_IMAGE_API

#ifndef CNN_NO_SERIALIZATION
#include "deserialization_helper.h"
#include "serialization_helper.h"
// to allow upcasting
CEREAL_REGISTER_TYPE(tiny_dnn::elu_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::leaky_relu_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::relu_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::sigmoid_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::softmax_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::softplus_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::softsign_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::tanh_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::tanh_p1m2_layer)
#endif  // CNN_NO_SERIALIZATION

// shortcut version of layer names
namespace tiny_dnn {
namespace layers {

using conv = tiny_dnn::convolutional_layer;

using q_conv = tiny_dnn::quantized_convolutional_layer;

using max_pool = tiny_dnn::max_pooling_layer;

using ave_pool = tiny_dnn::average_pooling_layer;

using fc = tiny_dnn::fully_connected_layer;

using dense = tiny_dnn::fully_connected_layer;

using zero_pad = tiny_dnn::zero_pad_layer;

// using rnn_cell = tiny_dnn::rnn_cell_layer;

#ifdef CNN_USE_GEMMLOWP
using q_fc = tiny_dnn::quantized_fully_connected_layer;
#endif

using add = tiny_dnn::elementwise_add_layer;

using dropout = tiny_dnn::dropout_layer;

using input = tiny_dnn::input_layer;

using linear = linear_layer;

using lrn = tiny_dnn::lrn_layer;

using concat = tiny_dnn::concat_layer;

using deconv = tiny_dnn::deconvolutional_layer;

using max_unpool = tiny_dnn::max_unpooling_layer;

using ave_unpool = tiny_dnn::average_unpooling_layer;

}  // namespace layers

namespace activation {

using sigmoid = tiny_dnn::sigmoid_layer;

using asinh = tiny_dnn::asinh_layer;

using tanh = tiny_dnn::tanh_layer;

using relu = tiny_dnn::relu_layer;

using rectified_linear = tiny_dnn::relu_layer;

using softmax = tiny_dnn::softmax_layer;

using leaky_relu = tiny_dnn::leaky_relu_layer;

using elu = tiny_dnn::elu_layer;

using selu = tiny_dnn::selu_layer;

using tanh_p1m2 = tiny_dnn::tanh_p1m2_layer;

using softplus = tiny_dnn::softplus_layer;

using softsign = tiny_dnn::softsign_layer;

}  // namespace activation

#include "models/alexnet.h"

using batch_norm = tiny_dnn::batch_normalization_layer;

using l2_norm = tiny_dnn::l2_normalization_layer;

using slice = tiny_dnn::slice_layer;

using power = tiny_dnn::power_layer;

}  // namespace tiny_dnn

#ifdef CNN_USE_CAFFE_CONVERTER
// experimental / require google protobuf
#include "io/caffe/layer_factory.h"
#endif

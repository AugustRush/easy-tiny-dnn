//
//  main.m
//  tiny-dnn
//
//  Created by pingwei liu on 2019/4/28.
//  Copyright © 2019 pingwei liu. All rights reserved.
//

#import <Foundation/Foundation.h>
#include <iostream>
#include "tiny_dnn.h"
#include "apple_math.h"

void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn) {
    
    using q_conv = tiny_dnn::quantized_convolutional_layer;
    using ave_pool = tiny_dnn::average_pooling_layer;
    using q_fc = tiny_dnn::quantized_fully_connected_layer;
    

    nn << tiny_dnn::convolutional_layer(32,32,5,5,1,6,tiny_dnn::padding::valid)
    << tiny_dnn::tanh_layer(28, 28, 6)
    << ave_pool(28, 28, 6, 2)             // S2, 6@28x28-in, 6@14x14-out
    << tiny_dnn::tanh_layer(14, 14, 6)
    << tiny_dnn::convolutional_layer(14,14,5,5,6,16,tiny_dnn::padding::valid)
    << tiny_dnn::tanh_layer(10, 10, 16)
    << ave_pool(10, 10, 16, 2)            // S4, 16@10x10-in, 16@5x5-out
    << tiny_dnn::tanh_layer(5, 5, 16)
    << tiny_dnn::convolutional_layer(5,5,5,5,16,120,tiny_dnn::padding::valid)
    << tiny_dnn::tanh_layer(120)
    << tiny_dnn::fully_connected_layer(120,10)
    << tiny_dnn::tanh_layer(10);
    // clang-format on
}

void construct_net1(tiny_dnn::network<tiny_dnn::sequential> &nn) {
//     connection table [Y.Lecun, 1998 Table.1]
    #define O true
    #define X false
        // clang-format off
        static const bool tbl[] = {
            O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O, O, X, X, X, O, O, O,
            X, X, O, O, O, O, X, O, O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
            X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O, X, X, O, O, O, X, X, O,
            O, O, O, X, O, O, X, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O};
    #undef O
    #undef X
    
    using q_conv = tiny_dnn::quantized_convolutional_layer;
    using ave_pool = tiny_dnn::average_pooling_layer;
    using q_fc = tiny_dnn::quantized_fully_connected_layer;
    
        tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();
    // construct nets
    //
    // C : convolution
    // S : sub-sampling
    // F : fully connected
    nn << q_conv(32, 32, 5, 1, 6, tiny_dnn::padding::valid, true, 1, 1,
                     backend_type)                    // C1, 1@32x32-in, 6@28x28-out
    << tiny_dnn::tanh_layer(28, 28, 6)
    << ave_pool(28, 28, 6, 2)             // S2, 6@28x28-in, 6@14x14-out
    << tiny_dnn::tanh_layer(14, 14, 6)
    << q_conv(14, 14, 5, 6, 16, tiny_dnn::core::connection_table(tbl, 6, 16),
              tiny_dnn::padding::valid, true, 1, 1,
              backend_type)               // C3, 6@14x14-in, 16@10x10-in
    << tiny_dnn::tanh_layer(10, 10, 16)
    << ave_pool(10, 10, 16, 2)            // S4, 16@10x10-in, 16@5x5-out
    << tiny_dnn::tanh_layer(5, 5, 16)
    << q_conv(5, 5, 5, 16, 120, tiny_dnn::padding::valid, true, 1, 1,
              backend_type)               // C5, 16@5x5-in, 120@1x1-out
    << tiny_dnn::tanh_layer(120)
    << q_fc(120, 10, true, backend_type)
    // F6, 120-in, 10-out
    << tiny_dnn::tanh_layer(10);
    // clang-format on
}

static void train_lenet(const std::string &data_dir_path) {
    // specify loss-function and learning strategy
    tiny_dnn::network<tiny_dnn::sequential> nn;
    tiny_dnn::adagrad optimizer;
    
    std::string path = "/Users/pingweiliu/Desktop/LeNet-model";
    
    if (tiny_dnn::files::exists(path)) {
        nn.load(path);
    } else {
        construct_net(nn);
    }
    
    std::cout << "load models..." << std::endl;
    
    // load MNIST dataset
    std::vector<tiny_dnn::label_t> train_labels, test_labels;
    std::vector<tiny_dnn::vec_t> train_images, test_images;
    
    tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels-idx1-ubyte",
                                 &train_labels);
    tiny_dnn::parse_mnist_images(data_dir_path + "/train-images-idx3-ubyte",
                                 &train_images, -1.0, 1.0, 2, 2);
    tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels-idx1-ubyte",
                                 &test_labels);
    tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images-idx3-ubyte",
                                 &test_images, -1.0, 1.0, 2, 2);
    
    std::cout << "start training" << std::endl;
    
    tiny_dnn::progress_display disp(train_images.size());
    tiny_dnn::timer t;
    int minibatch_size = 10;
    int num_epochs     = 10;
    
    optimizer.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(minibatch_size));
    
    // create callback
    auto on_enumerate_epoch = [&]() {
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_dnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;
        
        disp.restart(train_images.size());
        t.restart();
    };
    
    auto on_enumerate_minibatch = [&]() { disp += minibatch_size; };
    
    // training
    nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, minibatch_size,
                            num_epochs, on_enumerate_minibatch,
                            on_enumerate_epoch);
    
    std::cout << "end training." << std::endl;
    
    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);
    
    // save network model & trained weights
    nn.save(path);
}

void apple_math_test() {
    int size = 10;
    tiny_dnn::vec_t numbers(size);
    
    apple_fill(numbers.data(), 1.356, 1, size);
    printf("numbers 7 is %f\n", numbers[7]);
    
    for (int i = 0; i < size; i++) {
        printf("%f ",tan(numbers[i]));
    }
    printf("\n");
    apple_tanh(numbers.data(), numbers.data(), size);
    for (int i = 0; i < size; i++) {
        printf("%f ",numbers[i]);
    }
    
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // insert code here...
//        apple_math_test();
        train_lenet("/Users/pingweiliu/Downloads/MINIST");
    }
    return 0;
}

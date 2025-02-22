//===- flash_attn_bwd_test.cc -------------------------------*--- C++-*-===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#if BRT_ENABLE_FLASH_ATTENSION

#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "gtest/gtest.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>

static std::string test_file_flash_attn_bwd =
    "test/test_files/flash_attn_bwd.mlir";
// ground_truth_file and input files are generated by running
// generate_flash_attn_ground_truth.py at test/test_files/
static std::string input_q_file = "test/test_files/flash_attn_inputs_q.data";
static std::string input_k_file = "test/test_files/flash_attn_inputs_k.data";
static std::string input_v_file = "test/test_files/flash_attn_inputs_v.data";
static std::string input_dout_file =
    "test/test_files/flash_attn_inputs_dout.data";
static std::string ground_truth_file_dq =
    "test/test_files/flash_attn_bwd_outputs_dq.data";
static std::string ground_truth_file_dk =
    "test/test_files/flash_attn_bwd_outputs_dk.data";
static std::string ground_truth_file_dv =
    "test/test_files/flash_attn_bwd_outputs_dv.data";

using namespace brt;
using namespace brt::cuda;
using namespace brt::test;

TEST(SM80CUDATestFlashAttnBwd, Basic) {

  size_t b = 1;
  size_t seq_len = 128;
  size_t num_heads = 3;
  size_t head_dims = 32;
  size_t input_len = b * seq_len * num_heads * head_dims;
  size_t softmax_len = b * seq_len * num_heads;
  size_t dq_accum_len = input_len;

  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load = session.Load(test_file_flash_attn_bwd, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  __half *d_o;
  __half *d_q;
  __half *d_k;
  __half *d_v;

  __half *d_do;
  __half *d_dq;
  __half *d_dk;
  __half *d_dv;

  float *d_softmax_lse;
  float *d_dsoftmax;

  float *d_dq_accum;

  cudaMalloc(&d_o, input_len * sizeof(__half));
  cudaMalloc(&d_q, input_len * sizeof(__half));
  cudaMalloc(&d_k, input_len * sizeof(__half));
  cudaMalloc(&d_v, input_len * sizeof(__half));
  cudaMalloc(&d_do, input_len * sizeof(__half));
  cudaMalloc(&d_dq, input_len * sizeof(__half));
  cudaMalloc(&d_dk, input_len * sizeof(__half));
  cudaMalloc(&d_dv, input_len * sizeof(__half));
  cudaMalloc(&d_softmax_lse, softmax_len * sizeof(float));
  cudaMalloc(&d_dsoftmax, softmax_len * sizeof(float));
  cudaMalloc(&d_dq_accum, dq_accum_len * sizeof(float));

  ReadCUDAFloatValues(d_q, input_len, input_q_file);
  ReadCUDAFloatValues(d_k, input_len, input_k_file);
  ReadCUDAFloatValues(d_v, input_len, input_v_file);
  ReadCUDAFloatValues(d_do, input_len, input_dout_file);
  AssignCUDABuffer(d_dq, input_len, static_cast<__half>(0.f));
  AssignCUDABuffer(d_dk, input_len, static_cast<__half>(0.f));
  AssignCUDABuffer(d_dv, input_len, static_cast<__half>(0.f));
  AssignCUDABuffer(d_o, input_len, static_cast<__half>(0.f));

  AssignCUDABuffer(d_softmax_lse, softmax_len, 0.f);
  AssignCUDABuffer(d_dsoftmax, softmax_len, 0.f);

  AssignCUDABuffer(d_dq_accum, dq_accum_len, 0.f);

  // for (size_t i = 0; i < input_len; i++) {
  //   h_o[i] = static_cast<__half>(0.f);
  //   h_q[i] = static_cast<__half>(i / 300000.f);
  //   h_k[i] = static_cast<__half>(i / 400000.f);
  //   h_v[i] = static_cast<__half>(i / 500000.f);
  //   h_dq[i] = static_cast<__half>(0.f);
  //   h_dk[i] = static_cast<__half>(0.f);
  //   h_dv[i] = static_cast<__half>(0.f);
  //   h_dq_accum[i] = static_cast<float>(0.f);
  //   h_do[i] = static_cast<float>(1.f/20.f);
  // }

  // for (size_t i = 0; i < softmax_len; i++) {
  //   h_softmax_lse[i] = static_cast<float>(0.f);
  //   h_dsoftmax[i] = static_cast<float>(0.f);
  // }

  // cudaMemcpy(d_do, h_do, input_len * sizeof(__half), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_o, h_o, input_len * sizeof(__half), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_q, h_q, input_len * sizeof(__half), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_k, h_k, input_len * sizeof(__half), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_v, h_v, input_len * sizeof(__half), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_dq, h_dq, input_len * sizeof(__half), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_dk, h_dk, input_len * sizeof(__half), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_dv, h_dv, input_len * sizeof(__half), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_softmax_lse, h_softmax_lse, softmax_len * sizeof(float),
  //            cudaMemcpyHostToDevice);
  // cudaMemcpy(d_dsoftmax, h_dsoftmax, softmax_len * sizeof(float),
  //            cudaMemcpyHostToDevice);
  // cudaMemcpy(d_dq_accum, h_dq_accum, dq_accum_len * sizeof(float),
  //            cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  // PrintCUDAValues(d_q, input_len, input_len);

  // PrintCUDAValues(d_o, input_len, input_len);
  // PrintCUDAValues(d_q, input_len, input_len);
  // PrintCUDAValues(d_k, input_len, input_len);
  // PrintCUDAValues(d_v, input_len, input_len);
  // PrintCUDAValues(d_softmax_lse, softmax_len, 10);

  request->BindArg(0, d_do);
  request->BindArg(1, d_q);
  request->BindArg(2, d_k);
  request->BindArg(3, d_v);
  request->BindArg(4, d_o);
  request->BindArg(5, d_softmax_lse);
  request->BindArg(6, d_dq);
  request->BindArg(7, d_dk);
  request->BindArg(8, d_dv);
  request->BindArg(10, d_dsoftmax);
  request->BindArg(11, d_dq_accum);

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  // PrintCUDAValues(d_do, input_len, 10);
  // PrintCUDAValues(d_q, input_len, 10);
  // PrintCUDAValues(d_k, input_len, 10);
  // PrintCUDAValues(d_v, input_len, 10);
  // PrintCUDAValues(d_o, input_len, 10);
  // PrintCUDAValues(d_dq, input_len, 10);
  // PrintCUDAValues(d_dk, input_len, 10);
  // PrintCUDAValues(d_dv, input_len, 10);

  CheckCUDABuffer<__half>(
      (__half *)d_dq, /* size */ input_len, [&](__half *h_ptr) {
        __half *ground_truth = new __half[input_len];
        std::ifstream inFile;
        inFile.open(ground_truth_file_dq);
        if (inFile.is_open()) {
          float num;
          for (size_t i = 0; i < input_len; i++) {
            inFile >> num;
            // std::cout << "read:" << num << std::endl;
            ground_truth[i] = static_cast<__half>(num);
          }
        } else {
          ASSERT_TRUE(false)
              << "cannot open ground truth file of flash attn fwd output.";
        }
        inFile.close();
        float max_diff = 0.f;
        for (size_t i = 0; i < input_len; ++i) {
          if (abs(h_ptr[i] - ground_truth[i]) / abs(ground_truth[i]) >
              max_diff) {
            max_diff = abs(h_ptr[i] - ground_truth[i]) / abs(ground_truth[i]);
          }
          if (abs(h_ptr[i] - ground_truth[i]) / abs(ground_truth[i]) > 2e-6f) {
            std::cout << "index:" << i << " output:"
                      << "" << h_ptr[i] << " expect:" << ground_truth[i]
                      << " diff (ratio):"
                      << abs(h_ptr[i] - ground_truth[i]) / abs(ground_truth[i])
                      << std::endl;
            EXPECT_TRUE(false);
          }
        }
        std::cout << "dq max_diff (ratio):" << max_diff << std::endl;
      });

  CheckCUDABuffer<__half>(
      (__half *)d_dk, /* size */ input_len, [&](__half *h_ptr) {
        __half *ground_truth = new __half[input_len];
        std::ifstream inFile;
        inFile.open(ground_truth_file_dk);
        if (inFile.is_open()) {
          float num;
          for (size_t i = 0; i < input_len; i++) {
            inFile >> num;
            // std::cout << "read:" << num << std::endl;
            ground_truth[i] = static_cast<__half>(num);
          }
        } else {
          ASSERT_TRUE(false)
              << "cannot open ground truth file of flash attn fwd output.";
        }
        inFile.close();
        float max_diff = 0.f;
        for (size_t i = 0; i < input_len; ++i) {
          if (abs(h_ptr[i] - ground_truth[i]) / abs(ground_truth[i]) >
              max_diff) {
            max_diff = abs(h_ptr[i] - ground_truth[i]) / abs(ground_truth[i]);
          }
          if (abs(h_ptr[i] - ground_truth[i]) / abs(ground_truth[i]) > 2e-6f) {
            std::cout << "index:" << i << " output:"
                      << "" << h_ptr[i] << " expect:" << ground_truth[i]
                      << " diff (ratio):"
                      << abs(h_ptr[i] - ground_truth[i]) / abs(ground_truth[i])
                      << std::endl;
            EXPECT_TRUE(false);
          }
        }
        std::cout << "dk max_diff (ratio):" << max_diff << std::endl;
      });

  CheckCUDABuffer<__half>(
      (__half *)d_dv, /* size */ input_len, [&](__half *h_ptr) {
        __half *ground_truth = new __half[input_len];
        std::ifstream inFile;
        inFile.open(ground_truth_file_dv);
        if (inFile.is_open()) {
          float num;
          for (size_t i = 0; i < input_len; i++) {
            inFile >> num;
            // std::cout << "read:" << num << std::endl;
            ground_truth[i] = static_cast<__half>(num);
          }
        } else {
          ASSERT_TRUE(false)
              << "cannot open ground truth file of flash attn fwd output.";
        }
        inFile.close();
        float max_diff = 0.f;
        for (size_t i = 0; i < input_len; ++i) {
          if (abs(h_ptr[i] - ground_truth[i]) / abs(ground_truth[i]) >
              max_diff) {
            max_diff = abs(h_ptr[i] - ground_truth[i]) / abs(ground_truth[i]);
          }
          if (abs(h_ptr[i] - ground_truth[i]) / abs(ground_truth[i]) > 2e-6f) {
            std::cout << "index:" << i << " output:"
                      << "" << h_ptr[i] << " expect:" << ground_truth[i]
                      << " diff (ratio):"
                      << abs(h_ptr[i] - ground_truth[i]) / abs(ground_truth[i])
                      << std::endl;
            EXPECT_TRUE(false);
          }
        }
        std::cout << "dv max_diff (ratio):" << max_diff << std::endl;
      });
}

#endif // BRT_ENABLE_FLASH_ATTENSION

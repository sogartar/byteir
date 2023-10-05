// RUN: byteir-opt -test-print-composite-op-sharding-interface -split-input-file %s | FileCheck %s


// CHECK-LABEL: Running test-print-composite-op-sharding-interface on function mlp
//       CHECK: Loop iterator types : (parallel, parallel, reduction_sum, invalid, parallel, )
//       CHECK: Indexing maps : 
//       CHECK: (d0, d1, d2, d3, d4) -> (d0, d1, d2)
//       CHECK: (d0, d1, d2, d3, d4) -> (d2, d3)
//       CHECK: (d0, d1, d2, d3, d4) -> (d3, d4)
func.func @mlp(%arg0: tensor<2x4x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<32x8xf32>)
  -> tensor<2x4x8xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [2], 
                                      rhs_contracting_dimensions = [0]>, 
                                      precision_config = 
                                      [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<2x4x8xf32>, tensor<8x32xf32>) -> tensor<2x4x32xf32>
  %1 = mhlo.constant dense<0.000000e+00> : tensor<2x4x32xf32>
  %2 = mhlo.maximum %0, %1 : tensor<2x4x32xf32>
  %3 = "mhlo.dot_general"(%2, %arg2) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [2],
                                      rhs_contracting_dimensions = [0]>, 
                                      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<2x4x32xf32>, tensor<32x8xf32>) -> tensor<2x4x8xf32>
  return %3 : tensor<2x4x8xf32>
}

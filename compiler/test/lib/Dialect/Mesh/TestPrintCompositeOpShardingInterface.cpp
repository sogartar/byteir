/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Modifications Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "byteir/Dialect/Mesh/IR/MeshOps.h"
#include "byteir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace mesh {

namespace {

void printLoopIeratorTypes(
    const SmallVector<ShardingIteratorType> &iteratorTypes) {
  llvm::outs() << "(";
  for (ShardingIteratorType iteratorType : iteratorTypes) {
    llvm::outs() << iteratorType << ", ";
  }
  llvm::outs() << ")";
}

void printIndexingMaps(const SmallVector<AffineMap> &indexingMaps) {
  for (AffineMap affineMap : indexingMaps) {
    llvm::outs() << affineMap << "\n";
  }
}

struct TestPrintCompositeOpShardingInterface
    : public PassWrapper<TestPrintCompositeOpShardingInterface,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestPrintCompositeOpShardingInterface)

  StringRef getArgument() const final {
    return "test-print-composite-op-sharding-interface";
  }

  StringRef getDescription() const final {
    return "Print the automatically deduced loop iterators and indexing maps "
           "of a function operation.";
  }

  void runOnOperation() override {
    llvm::outs()
        << "Running test-print-composite-op-sharding-interface on function "
        << getOperation().getName() << "\n";
    ShardingInterface shardingInterface =
        llvm::dyn_cast<ShardingInterface>(getOperation().getOperation());
    assert(shardingInterface);

    SmallVector<ShardingIteratorType> loopIteratorTypes =
        shardingInterface.getLoopIteratorTypes();
    llvm::outs() << "Loop iterator types : ";
    printLoopIeratorTypes(loopIteratorTypes);
    llvm::outs() << "\n";

    SmallVector<AffineMap> indexingMaps = shardingInterface.getIndexingMaps();
    llvm::outs() << "Indexing maps : \n";
    printIndexingMaps(indexingMaps);
    llvm::outs() << "\n";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<MeshDialect, func::FuncDialect>();
  }
};

} // namespace

void registerTestPrintCompositeOpShardingInterfacePass() {
  PassRegistration<TestPrintCompositeOpShardingInterface>();
}
} // namespace mesh
} // namespace mlir

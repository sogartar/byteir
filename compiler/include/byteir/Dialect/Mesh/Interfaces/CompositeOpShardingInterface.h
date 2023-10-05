#ifndef BYTEIR_DIALECT_MESH_INTERFACES_COMPOSITEOPSHARDINGINTERFACE_H
#define BYTEIR_DIALECT_MESH_INTERFACES_COMPOSITEOPSHARDINGINTERFACE_H

#include "byteir/Dialect/Mesh/IR/MeshOps.h"
#include "byteir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <numeric>
#include <stdint.h>
#include <tuple>

#define DEBUG_TYPE "composite-op-sharding-interface"

namespace mlir {
namespace mesh {

namespace detail {

using LoopIteratorIndex = int32_t;
using DimensionIndex = int16_t;
using ValueDimLoopIteratorMultimap =
    llvm::DenseMap<std::tuple<Value, DimensionIndex>,
                   SmallVector<LoopIteratorIndex>>;
using ValueDimSet = llvm::SmallDenseSet<std::tuple<Value, DimensionIndex>>;
using LoopIteratorTypeMap =
    llvm::DenseMap<LoopIteratorIndex, ShardingIteratorType>;
using LoopIteratorValueDimMultimap =
    llvm::DenseMap<LoopIteratorIndex, ValueDimSet>;

inline ShardingIteratorType combineIteratorTypes(ShardingIteratorType t1,
                                                 ShardingIteratorType t2) {
  if (t1 == t2) {
    return t1;
  } else {
    return ShardingIteratorType::invalid;
  }
}

inline bool isPassThroughOperation(Operation *op) {
  if (llvm::dyn_cast<func::ReturnOp>(op)) {
    return true;
  } else if (llvm::dyn_cast<mesh::AnnotateOp>(op)) {
    return true;
  }

  return false;
}

struct CompositeOpIndexingMapsAndLoopTypesDeduction {

  CompositeOpIndexingMapsAndLoopTypesDeduction(
      ValueDimSet &valueDimsWithUnresolvedLoopIterators,
      ValueDimLoopIteratorMultimap &valueDimLoopIteratorMultimap,
      LoopIteratorValueDimMultimap &loopIteratorValueDimMultimap,
      LoopIteratorTypeMap &loopIteratorTypeMap,
      LoopIteratorIndex &nextFreeLoopIteratorIndex)
      : valueDimsWithUnresolvedLoopIterators(
            valueDimsWithUnresolvedLoopIterators),
        valueDimLoopIteratorMultimap(valueDimLoopIteratorMultimap),
        loopIteratorValueDimMultimap(loopIteratorValueDimMultimap),
        loopIteratorTypeMap(loopIteratorTypeMap),
        nextFreeLoopIteratorIndex(nextFreeLoopIteratorIndex) {}

  void insertValueDim(Value value, DimensionIndex dim,
                      LoopIteratorIndex loopIterator) {
    auto valueDimTuple = std::make_tuple(value, dim);
    auto &loopIterators = valueDimLoopIteratorMultimap[valueDimTuple];
    loopIterators.push_back(loopIterator);
    loopIteratorValueDimMultimap[loopIterator].insert(valueDimTuple);
    if (loopIterators.size() > 1) {
      valueDimsWithUnresolvedLoopIterators.insert(valueDimTuple);
    }

    LLVM_DEBUG(checkStateIsConsistent());
  }

  template <typename LoopIteratorIndexIt>
  void insertValue(Value value, AffineMap indexingMap,
                   LoopIteratorIndexIt loopIteratorIndexBegin) {
    auto rankedTensorType = value.getType().dyn_cast<RankedTensorType>();
    if (!rankedTensorType) {
      return;
    }

    for (int64_t dim = 0; dim < rankedTensorType.getRank(); ++dim) {
      insertValueDim(
          value, dim,
          loopIteratorIndexBegin
              [indexingMap.getResult(dim).cast<AffineDimExpr>().getPosition()]);
    }
  }

  void insertOperationWithoutShardingInterface(Operation *op) {
    assert(isPassThroughOperation(op));
    for (size_t i = 0; i < op->getResults().size(); ++i) {
      RankedTensorType rankedTesnorType =
          op->getResults()[i].getType().dyn_cast<RankedTensorType>();
      if (!rankedTesnorType) {
        continue;
      }

      for (int64_t dim = 0; dim < rankedTesnorType.getRank(); ++dim) {
        auto operandDimTuple = std::make_tuple(op->getOperands()[i], dim);
        auto resultDimTuple = std::make_tuple(op->getResults()[i], dim);
        valueDimLoopIteratorMultimap[resultDimTuple] =
            valueDimLoopIteratorMultimap[operandDimTuple];
        valueDimsWithUnresolvedLoopIterators.insert(resultDimTuple);
      }
    }
  }

  // Insertion order must respect dependencies.
  void insertOperation(Operation *op) {
    ShardingInterface shardingInterface = llvm::dyn_cast<ShardingInterface>(op);
    if (!shardingInterface) {
      insertOperationWithoutShardingInterface(op);
      return;
    }
    SmallVector<ShardingIteratorType> opLoopIteratorTypes =
        shardingInterface.getLoopIteratorTypes();
    SmallVector<AffineMap> opIndexingMaps = shardingInterface.getIndexingMaps();

    SmallVector<LoopIteratorIndex, 64> loopIteratorIndices(
        opLoopIteratorTypes.size());
    std::iota(loopIteratorIndices.begin(), loopIteratorIndices.end(),
              nextFreeLoopIteratorIndex);
    nextFreeLoopIteratorIndex += loopIteratorIndices.size();

    for (size_t i = 0; i < opLoopIteratorTypes.size(); ++i) {
      loopIteratorTypeMap[loopIteratorIndices[i]] = opLoopIteratorTypes[i];
    }

    size_t valueIdx = 0;
    for (Value operand : op->getOperands()) {
      insertValue(operand, opIndexingMaps[valueIdx],
                  loopIteratorIndices.begin());
      ++valueIdx;
    }
    for (Value result : op->getResults()) {
      insertValue(result, opIndexingMaps[valueIdx],
                  loopIteratorIndices.begin());
      ++valueIdx;
    }
  }

  void insertBlockArgument(BlockArgument argument) {
    if (!argument.getType().isa<RankedTensorType>()) {
      return;
    }
    RankedTensorType rankedTensor = argument.getType().cast<RankedTensorType>();
    for (DimensionIndex dim = 0; dim < rankedTensor.getRank(); ++dim) {
      auto it =
          valueDimLoopIteratorMultimap.find(std::make_tuple(argument, dim));
      if (it == valueDimLoopIteratorMultimap.end()) {
        loopIteratorTypeMap[nextFreeLoopIteratorIndex] =
            ShardingIteratorType::invalid;
        insertValueDim(argument, dim, nextFreeLoopIteratorIndex);
        ++nextFreeLoopIteratorIndex;
      }
    }
  }

  void combineLoopIterators(LoopIteratorIndex i1, LoopIteratorIndex i2) {
    LLVM_DEBUG(llvm::dbgs()
               << "combineLoopIterators " << i1 << ", " << i2 << "\n");

    ShardingIteratorType newLoopIteratorType =
        combineIteratorTypes(loopIteratorTypeMap[i1], loopIteratorTypeMap[i2]);
    loopIteratorTypeMap[i1] = newLoopIteratorType;
    loopIteratorTypeMap.erase(i2);

    auto loopIteratorValueDimMultimapIt1 =
        loopIteratorValueDimMultimap.find(i1);
    assert(loopIteratorValueDimMultimapIt1 !=
           loopIteratorValueDimMultimap.end());
    auto loopIteratorValueDimMultimapIt2 =
        loopIteratorValueDimMultimap.find(i2);
    assert(loopIteratorValueDimMultimapIt2 !=
           loopIteratorValueDimMultimap.end());

    for (std::tuple<Value, DimensionIndex> &valueDim :
         loopIteratorValueDimMultimapIt2->second) {
      auto valueDimLoopIteratorMultimapIt =
          valueDimLoopIteratorMultimap.find(valueDim);
      assert(valueDimLoopIteratorMultimapIt !=
             valueDimLoopIteratorMultimap.end());
      auto it = std::find(valueDimLoopIteratorMultimapIt->second.begin(),
                          valueDimLoopIteratorMultimapIt->second.end(), i2);
      assert(it != valueDimLoopIteratorMultimapIt->second.end());
      if (std::find(valueDimLoopIteratorMultimapIt->second.begin(),
                    valueDimLoopIteratorMultimapIt->second.end(),
                    i1) == valueDimLoopIteratorMultimapIt->second.end()) {
        *it = i1;
      } else {
        valueDimLoopIteratorMultimapIt->second.erase(it);
      }
      if (valueDimLoopIteratorMultimapIt->second.size() == 1) {
        valueDimsWithUnresolvedLoopIterators.erase(
            valueDimLoopIteratorMultimapIt->first);
      }
    }

    // Move value-dims from one iterator to the other.
    loopIteratorValueDimMultimapIt1->second.insert(
        loopIteratorValueDimMultimapIt2->second.begin(),
        loopIteratorValueDimMultimapIt2->second.end());
    loopIteratorValueDimMultimap.erase(loopIteratorValueDimMultimapIt2);

    LLVM_DEBUG(printState());
    LLVM_DEBUG(checkStateIsConsistent());
  }

  void
  resolveLoopIterators(const std::tuple<Value, DimensionIndex> &valueDimTuple) {
    auto valueDimLoopIteratorMultimapIt =
        valueDimLoopIteratorMultimap.find(valueDimTuple);
    assert(valueDimLoopIteratorMultimapIt !=
           valueDimLoopIteratorMultimap.end());
    assert(valueDimLoopIteratorMultimapIt->second.size() > 1);
    while (valueDimLoopIteratorMultimapIt->second.size() > 1) {
      combineLoopIterators(valueDimLoopIteratorMultimapIt->second.front(),
                           valueDimLoopIteratorMultimapIt->second.back());
    }
  }

  void resolveLoopIterators() {
    while (!valueDimsWithUnresolvedLoopIterators.empty()) {
      resolveLoopIterators(*valueDimsWithUnresolvedLoopIterators.begin());
    }
  }

  void getBlockIndexingMapsAndLoopIteratorTypes(
      Block &block, SmallVector<AffineMap> &outIndexingMaps,
      SmallVector<ShardingIteratorType> &outLoopIteratorTypes) {
    for (Operation &op : block) {
      insertOperation(&op);
    }
    resolveLoopIterators();

    SmallVector<Value> argumentResultValues;
    for (BlockArgument argument : block.getArguments()) {
      argumentResultValues.push_back(argument);
      insertBlockArgument(argument);
    }
    for (Value result : block.getTerminator()->getResults()) {
      argumentResultValues.push_back(result);
    }

    MLIRContext *mlirCtx = block.getParent()->getContext();

    // Compute remapping of iterators to range [0, 1, 2, ...).
    llvm::DenseMap<LoopIteratorIndex, LoopIteratorIndex> loopIteratorRemapping;
    for (Value value : argumentResultValues) {
      auto rankedTensor = value.getType().dyn_cast<RankedTensorType>();
      if (rankedTensor) {
        for (DimensionIndex dim = 0; dim < rankedTensor.getRank(); ++dim) {
          auto valueDimTuple = std::make_tuple(value, dim);
          loopIteratorRemapping.try_emplace(
              valueDimLoopIteratorMultimap[valueDimTuple].front(),
              loopIteratorRemapping.size());
        }
      }
    }

    // Populate output indexing maps.
    SmallVector<AffineExpr> affineExprs;
    for (Value value : argumentResultValues) {
      auto rankedTensor = value.getType().dyn_cast<RankedTensorType>();
      if (!rankedTensor) {
        outIndexingMaps.push_back(AffineMap::get(mlirCtx));
      } else {
        for (DimensionIndex dim = 0; dim < rankedTensor.getRank(); ++dim) {
          auto valueDimTuple = std::make_tuple(value, dim);
          AffineExpr affineExpr = getAffineDimExpr(
              loopIteratorRemapping[valueDimLoopIteratorMultimap[valueDimTuple]
                                        .front()],
              block.getParent()->getContext());
          affineExprs.push_back(affineExpr);
        }
        outIndexingMaps.push_back(AffineMap::get(loopIteratorRemapping.size(),
                                                 0, affineExprs, mlirCtx));
        affineExprs.clear();
      }
    }

    // Populate output loop iterator types.
    outLoopIteratorTypes.resize(loopIteratorRemapping.size());
    for (auto &loopIteratorRemapPair : loopIteratorRemapping) {
      outLoopIteratorTypes[loopIteratorRemapPair.second] =
          loopIteratorTypeMap[loopIteratorRemapPair.first];
    }
  }

  void checkStateIsConsistent() {
    SmallVector<LoopIteratorIndex> loopIterators;
    for (auto &valueDimIteratorsPair : valueDimLoopIteratorMultimap) {
      for (auto iteratorIdx : valueDimIteratorsPair.second) {
        auto loopIteratorValueDimMultimapIt =
            loopIteratorValueDimMultimap.find(iteratorIdx);
        assert(loopIteratorValueDimMultimapIt !=
               loopIteratorValueDimMultimap.end());
        assert(std::find(loopIteratorValueDimMultimapIt->second.begin(),
                         loopIteratorValueDimMultimapIt->second.end(),
                         valueDimIteratorsPair.first) !=
               loopIteratorValueDimMultimapIt->second.end());
        loopIterators.push_back(iteratorIdx);
      }

      // Check fo duplicates.
      std::sort(loopIterators.begin(), loopIterators.end());
      assert(std::adjacent_find(loopIterators.begin(), loopIterators.end()) ==
             loopIterators.end());
      loopIterators.clear();
    }

    for (auto &loopIteratorValueDims : loopIteratorValueDimMultimap) {
      assert(loopIteratorTypeMap.contains(loopIteratorValueDims.first));
      for (auto &valueDimTuple : loopIteratorValueDims.second) {
        auto valueDimLoopIteratorMultimapIt =
            valueDimLoopIteratorMultimap.find(valueDimTuple);
        assert(valueDimLoopIteratorMultimapIt !=
               valueDimLoopIteratorMultimap.end());
        assert(std::find(valueDimLoopIteratorMultimapIt->second.begin(),
                         valueDimLoopIteratorMultimapIt->second.end(),
                         loopIteratorValueDims.first) !=
               valueDimLoopIteratorMultimapIt->second.end());
      }
    }

    for (auto &loopIteratorTypePair : loopIteratorTypeMap) {
      assert(loopIteratorTypePair.first < nextFreeLoopIteratorIndex);
    }

    for (auto &valueDimTuple : valueDimsWithUnresolvedLoopIterators) {
      auto valueDimLoopIteratorMultimapIt =
          valueDimLoopIteratorMultimap.find(valueDimTuple);
      assert(valueDimLoopIteratorMultimapIt !=
             valueDimLoopIteratorMultimap.end());
      assert(valueDimLoopIteratorMultimapIt->second.size() > 1);
    }
  }

  void printValue(Value v) {
    std::string name;
    if (auto *op = v.getDefiningOp()) {
      std::string str;
      llvm::raw_string_ostream stream(str);
      op->print(stream);
      name = str.substr(0, str.find('='));
      name.erase(std::remove_if(name.begin(), name.end(), ::isspace),
                 name.end());
    } else {
      BlockArgument arg = llvm::cast<BlockArgument>(v);
      llvm::raw_string_ostream stream(name);
      stream << "%arg" << arg.getArgNumber();
    }
    llvm::dbgs() << name;
  }

  void printValueDimTuple(const std::tuple<Value, DimensionIndex> &vd) {
    llvm::dbgs() << "(";
    printValue(std::get<0>(vd));
    llvm::dbgs() << ", " << std::get<1>(vd) << ")";
  }

  template <typename It> void printValueDimTupleRange(It begin, It end) {
    llvm::dbgs() << "[";
    for (It it = begin; it != end; ++it) {
      printValueDimTuple(*it);
      llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]";
  }

  template <typename It> void printRange(It begin, It end) {
    llvm::dbgs() << "[";
    for (It it = begin; it != end; ++it) {
      llvm::dbgs() << *it << ", ";
    }
    llvm::dbgs() << "]";
  }

  void printState() {
    llvm::dbgs() << "CompositeOpIndexingMapsAndLoopTypesDeduction state\n";

    llvm::dbgs() << "valueDimLoopIteratorMultimap = "
                 << "\n";
    for (auto &kv : valueDimLoopIteratorMultimap) {
      printValueDimTuple(kv.first);
      llvm::dbgs() << " -> ";
      printRange(kv.second.begin(), kv.second.end());
      llvm::dbgs() << "\n";
    }

    llvm::dbgs() << "loopIteratorValueDimMultimap = "
                 << "\n";
    for (auto &kv : loopIteratorValueDimMultimap) {
      llvm::dbgs() << kv.first << " -> ";
      printValueDimTupleRange(kv.second.begin(), kv.second.end());
      llvm::dbgs() << "\n";
    }

    llvm::dbgs() << "\n";
  }

  ValueDimSet &valueDimsWithUnresolvedLoopIterators;
  ValueDimLoopIteratorMultimap &valueDimLoopIteratorMultimap;
  LoopIteratorValueDimMultimap &loopIteratorValueDimMultimap;
  LoopIteratorTypeMap &loopIteratorTypeMap;
  LoopIteratorIndex &nextFreeLoopIteratorIndex;
};

inline void getBlockIndexingMapsAndLoopIteratorTypes(
    Block &block, SmallVector<AffineMap> &outIndexingMaps,
    SmallVector<ShardingIteratorType> &outLoopIteratorTypes) {
  ValueDimSet valueDimsWithUnresolvedLoopIterators;
  ValueDimLoopIteratorMultimap valueDimLoopIteratorMultimap;
  LoopIteratorValueDimMultimap loopIteratorValueDimMultimap;
  LoopIteratorTypeMap loopIteratorTypeMap;
  LoopIteratorIndex nextFreeLoopIteratorIndex = 0;
  CompositeOpIndexingMapsAndLoopTypesDeduction
      compositeOpIndexingMapsAndLoopTypesDeduction(
          valueDimsWithUnresolvedLoopIterators, valueDimLoopIteratorMultimap,
          loopIteratorValueDimMultimap, loopIteratorTypeMap,
          nextFreeLoopIteratorIndex);
  compositeOpIndexingMapsAndLoopTypesDeduction
      .getBlockIndexingMapsAndLoopIteratorTypes(block, outIndexingMaps,
                                                outLoopIteratorTypes);
}

} // namespace detail

template <typename CompositeOp> Block &entryBlock(CompositeOp op) {
  assert(!op.getOperation()->getRegions().empty());
  assert(!op.getOperation()->getRegion(0).empty());
  return op.getOperation()->getRegion(0).front();
}

// Requires implementation of
// Block& entryBlock(CompositeOp);
template <typename CompositeOp>
struct CompositeOpShardingInteface
    : public ShardingInterface::ExternalModel<
          CompositeOpShardingInteface<CompositeOp>, CompositeOp> {
  SmallVector<ShardingIteratorType> getLoopIteratorTypes(Operation *op) const {
    SmallVector<AffineMap> outIndexingMaps;
    SmallVector<ShardingIteratorType> outLoopIteratorTypes;
    CompositeOp compositeOp = cast<CompositeOp>(op);
    detail::getBlockIndexingMapsAndLoopIteratorTypes(
        entryBlock(compositeOp), outIndexingMaps, outLoopIteratorTypes);
    return outLoopIteratorTypes;
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    SmallVector<AffineMap> outIndexingMaps;
    SmallVector<ShardingIteratorType> outLoopIteratorTypes;
    CompositeOp compositeOp = cast<CompositeOp>(op);
    detail::getBlockIndexingMapsAndLoopIteratorTypes(
        entryBlock(compositeOp), outIndexingMaps, outLoopIteratorTypes);
    return outIndexingMaps;
  }
};

} // namespace mesh
} // namespace mlir

#undef DEBUG_TYPE

#endif // BYTEIR_DIALECT_MESH_INTERFACES_COMPOSITEOPSHARDINGINTERFACE_H

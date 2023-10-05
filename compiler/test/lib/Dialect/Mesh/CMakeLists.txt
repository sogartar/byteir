add_mlir_library(MLIRMeshTest
  TestPrintCompositeOpShardingInterface.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRMeshDialect
  ByteIRMhloShardingInterfaceImpl
  )

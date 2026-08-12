#pragma once

#include <cstdint>

enum class IRBaseType {
    F32,
    F64,
    I32,
    I64,
    F32Ptr,
    F64Ptr,
    I32Ptr,
    I64Ptr,
};

template <IRBaseType K>
struct MapType;

template <>
struct MapType<IRBaseType::F32> {
    using type = float*;
};

template <>
struct MapType<IRBaseType::F64> {
    using type = double*;
};

template <>
struct MapType<IRBaseType::F32Ptr> {
    using type = float*;
};

template <>
struct MapType<IRBaseType::F64Ptr> {
    using type = double*;
};

template <>
struct MapType<IRBaseType::I32> {
    using type = int32_t;
};

template <>
struct MapType<IRBaseType::I64> {
    using type = int64_t;
};

template <>
struct MapType<IRBaseType::I32Ptr> {
    using type = int32_t;
};

template <>
struct MapType<IRBaseType::I64Ptr> {
    using type = int64_t;
};

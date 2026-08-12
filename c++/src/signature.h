#pragma once

#include "type_list.h"
#include "type_mapping.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace pyggpu {

using SignatureID = uint64_t;

// The allowed types for kernel arguments
using AllowedTypes = type_list::type_list<int, float, float*>;
constexpr size_t MAX_ARG_TYPES = type_list::size<AllowedTypes>::value;

template<typename... Args>
struct Signature {
    static std::string asString() {
        return "void(" + ((std::string(typeid(Args).name()) + ", ") + ...) + ")";
    }
};

template <IRBaseType... Ks>
struct MakeSignature {
    using type = Signature<typename MapType<Ks>::type...>;
};

enum class TypeCode : uint8_t {
    Float      = 1,
    Double     = 2,
    Int32      = 3,
    Int64      = 4,
    FloatPtr   = 5,
    DoublePtr  = 6,
    Int32Ptr   = 7,
    Int64Ptr   = 8
};

inline TypeCode irType2TypeCode(IRBaseType type) {
    switch (type) {
        case IRBaseType::F32: return TypeCode::Float;
        case IRBaseType::F64: return TypeCode::Double;
        case IRBaseType::I32: return TypeCode::Int32;
        case IRBaseType::I64: return TypeCode::Int64;
        case IRBaseType::F32Ptr: return TypeCode::FloatPtr;
        case IRBaseType::F64Ptr: return TypeCode::DoublePtr;
        case IRBaseType::I32Ptr: return TypeCode::Int32Ptr;
        case IRBaseType::I64Ptr: return TypeCode::Int64Ptr;
        default:
            throw std::runtime_error("Unsupported IRBaseType");
    }
}

template<typename T> struct type_code;

template<> struct type_code<float>      { static constexpr TypeCode value = TypeCode::Float; };
template<> struct type_code<double>     { static constexpr TypeCode value = TypeCode::Double; };
template<> struct type_code<int32_t>    { static constexpr TypeCode value = TypeCode::Int32; };
template<> struct type_code<int64_t>    { static constexpr TypeCode value = TypeCode::Int64; };

template<> struct type_code<float*>     { static constexpr TypeCode value = TypeCode::FloatPtr; };
template<> struct type_code<double*>    { static constexpr TypeCode value = TypeCode::DoublePtr; };
template<> struct type_code<int32_t*>   { static constexpr TypeCode value = TypeCode::Int32Ptr; };
template<> struct type_code<int64_t*>   { static constexpr TypeCode value = TypeCode::Int64Ptr; };

template<typename Sig>
struct ComputeIdImpl;

template<typename... Ts>
struct ComputeIdImpl<Signature<Ts...>> {
    static constexpr SignatureID value = [] {
        uint64_t id = 0;

        ((id = id * MAX_ARG_TYPES + static_cast<uint64_t>(type_code<Ts>::value)), ...);

        return id;
    }();
};

template<typename Sig>
constexpr SignatureID computeId() {
    return ComputeIdImpl<Sig>::value;
}

inline SignatureID computeIdFromIR(const std::vector<IRBaseType>& arg_types) {
    uint64_t id = 0;
    for (const auto& type : arg_types) {
        id = id * MAX_ARG_TYPES + static_cast<uint64_t>(irType2TypeCode(type));
    }
    return id;
}

} // namespace pyggpu
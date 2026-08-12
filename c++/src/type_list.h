#pragma once

#include <cstddef>
#include <boost/mp11.hpp>

namespace pyggpu::type_list {

// The type list 
template<typename... Ts>
struct type_list {};

template<typename List>
struct size;

template<typename... Ts>
struct size<type_list<Ts...>> {
    static constexpr std::size_t value = sizeof...(Ts);
};


// Prepend a type to a type list
template<typename T, typename List>
struct prepend;

template<typename T, typename... Ts>
struct prepend<T, type_list<Ts...>> {
    using type = type_list<T, Ts...>;
};


// Concatenate two type lists
template<typename List1, typename List2>
struct concat;

template<typename... Ts1, typename... Ts2>
struct concat<type_list<Ts1...>, type_list<Ts2...>> {
    using type = type_list<Ts1..., Ts2...>;
};

template <typename T1, typename T2>
using concat_t = typename concat<T1, T2>::type;

template<typename... Lists>
using cartesian_product = boost::mp11::mp_product<type_list, Lists...>;

} // namespace pyggpu::type_list
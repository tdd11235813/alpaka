#include "Defines.hpp"
#pragma once

#define ALPAKA_TEST_MATH_OP_FUNCTOR(NAME, ARITY, FUNCTION, ...)\
class NAME\
{\
public:\
    const Range ranges [static_cast<int>(ARITY)] = {__VA_ARGS__};\
    static constexpr Arity arity = ARITY;\
    \
    template<\
        typename TAcc,\
        typename... TArgs,\
        typename std::enable_if<!std::is_same<TAcc, std::nullptr_t>::value, int>::type = 0>\
    ALPAKA_FN_ACC\
    auto operator()(TAcc const & acc, TArgs... args) const-> double\
    {\
        return alpaka::math::FUNCTION(acc, args...);\
    }\
    \
    template<\
        typename TAcc = std::nullptr_t,\
        typename... TArgs,\
        typename std::enable_if<std::is_same<TAcc, std::nullptr_t>::value, int>::type = 0>\
    ALPAKA_FN_HOST_ACC\
    auto operator()(TAcc const & acc,TArgs... args ) const-> double\
    {\
        alpaka::ignore_unused(acc);\
        return std::FUNCTION( args... );\
    }\
};


ALPAKA_TEST_MATH_OP_FUNCTOR( OpAbs, Arity::UNARY, abs,Range::UNRESTRICTED )
ALPAKA_TEST_MATH_OP_FUNCTOR( OpRemainder, Arity::BINARY, remainder, Range::UNRESTRICTED, Range::NOT_ZERO)
ALPAKA_TEST_MATH_OP_FUNCTOR( OpPow, Arity::BINARY, pow, Range::POSITIVE_AND_ZERO, Range::UNRESTRICTED)

using BinaryFunctors = std::tuple<
    OpPow,
    OpRemainder
    >;

using UnaryFunctors = std::tuple<
    OpAbs
    >;
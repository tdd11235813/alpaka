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
    auto operator()(TAcc const & acc, TArgs... args) const\
    -> decltype( alpaka::math::FUNCTION(acc, args...) )\
    {\
        return alpaka::math::FUNCTION(acc, args...);\
    }\
    \
    template<\
        typename TAcc = std::nullptr_t,\
        typename... TArgs,\
        typename std::enable_if<std::is_same<TAcc, std::nullptr_t>::value, int>::type = 0>\
    ALPAKA_FN_HOST_ACC\
    auto operator()( TAcc const & acc,TArgs... args ) const\
    -> decltype( std::FUNCTION( args... ) )\
    {\
        alpaka::ignore_unused(acc);\
        return std::FUNCTION( args... );\
    }\
    friend std::ostream & operator << (std::ostream &out, const NAME &op) \
    {\
      out << #NAME;\
    alpaka::ignore_unused(op); \
    return out;\
    }\
};


ALPAKA_TEST_MATH_OP_FUNCTOR( OpAbs,
    Arity::UNARY,
    abs,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpAcos,
    Arity::UNARY,
    acos,
    Range::ONE_NEIGHBOURHOOD )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpAsin,
    Arity::UNARY,
    asin,
    Range::ONE_NEIGHBOURHOOD )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpAtan,
    Arity::UNARY,
    atan,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpCbrt,
    Arity::UNARY,
    cbrt,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpCeil,
    Arity::UNARY,
    ceil,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpCos,
    Arity::UNARY,
    cos,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpErf,
    Arity::UNARY,
    erf,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpExp,
    Arity::UNARY,
    exp,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpFloor,
    Arity::UNARY,
    floor,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpLog,
    Arity::UNARY,
    log,
    Range::POSITIVE_ONLY )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpRound,
    Arity::UNARY,
    round,
    Range::UNRESTRICTED )
//ALPAKA_TEST_MATH_OP_FUNCTOR( OpRsqrt, Arity::UNARY, rsqr ,Range::UNRESTRICTED )
ALPAKA_TEST_MATH_OP_FUNCTOR( OpSin,
    Arity::UNARY,
    sin,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpSqrt,
    Arity::UNARY,
    sqrt,
    Range::POSITIVE_AND_ZERO )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpTan,
    Arity::UNARY,
    tan,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpTrunc,
    Arity::UNARY,
    trunc,
    Range::UNRESTRICTED )

// All binary operators.
ALPAKA_TEST_MATH_OP_FUNCTOR( OpAtan2,
    Arity::BINARY,
    atan2,
    Range::NOT_ZERO,
    Range::NOT_ZERO )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpFmod,
    Arity::BINARY,
    fmod,
    Range::UNRESTRICTED,
    Range::NOT_ZERO )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpMax,
    Arity::BINARY,
    max,
    Range::UNRESTRICTED,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpMin,
    Arity::BINARY,
    min,
    Range::UNRESTRICTED,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpPow,
    Arity::BINARY,
    pow,
    Range::POSITIVE_AND_ZERO,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpRemainder,
    Arity::BINARY,
    remainder,
    Range::UNRESTRICTED,
    Range::NOT_ZERO )


using BinaryFunctors = std::tuple<
    OpAtan2,
    OpFmod,
//    OpMax,
//    OpMin,
    OpPow,
    OpRemainder
    >;

using UnaryFunctors = std::tuple<
    OpAbs,
    OpAcos,
    OpAsin,
    OpAtan,
    OpCbrt,
    OpCeil,
    OpCos,
    OpErf,
    OpExp,
    OpFloor,
    OpLog,
    OpRound,
    OpSin,
    OpSqrt,
    OpTan,
    OpTrunc
    >;
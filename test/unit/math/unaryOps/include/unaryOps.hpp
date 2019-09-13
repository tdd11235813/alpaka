/** Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <catch2/catch.hpp>
#include <iostream>

//! @def ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY
//! @brief a macro to generate all needed functors
#define ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(NAME, STD_OP, ALPAKA_OP, RANGE) \
class NAME \
{ \
public: \
static constexpr Range range = RANGE; \
template< \
    typename TAcc, \
    typename T> \
ALPAKA_FN_ACC \
auto operator() (TAcc const& acc, T const & arg) const -> decltype(ALPAKA_OP(acc, arg)) { return ALPAKA_OP(acc,arg);} \
\
template < typename T > \
ALPAKA_FN_HOST \
auto operator() (T const & arg) const -> decltype(STD_OP(arg)) { return STD_OP(arg);} \
friend std::ostream & operator << (std::ostream &out, const NAME &op) \
{\
  out << #NAME;\
alpaka::ignore_unused(op); \
return out;\
}\
};


//! @enum Range
//! @brief Possible definition ranges.
enum class Range
{
    POSITIVE_ONLY,
    POSITIVE_AND_ZERO,
    ONE_NEIGHBOURHOOD, // [-1, 1]
    UNRESTRICTED
};

// This function is not part of the STL.
template<typename T>
T rsqrt(T t){
    return 1 / std::sqrt(t);
}

// Generate all unary functors.
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncAbs,   std::abs,   alpaka::math::abs,    Range::UNRESTRICTED)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncAcos,  std::acos,  alpaka::math::acos,   Range::ONE_NEIGHBOURHOOD)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncAsin,  std::asin,  alpaka::math::asin,   Range::ONE_NEIGHBOURHOOD)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncAtan,  std::atan,  alpaka::math::atan,   Range::UNRESTRICTED)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncCbrt,  std::cbrt,  alpaka::math::cbrt,   Range::UNRESTRICTED)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncCeil,  std::ceil,  alpaka::math::ceil,   Range::UNRESTRICTED)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncCos,   std::cos,   alpaka::math::cos,    Range::UNRESTRICTED)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncErf,   std::erf,   alpaka::math::erf,    Range::UNRESTRICTED)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncExp,   std::exp,   alpaka::math::exp,    Range::UNRESTRICTED)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncFloor, std::floor, alpaka::math::floor,  Range::UNRESTRICTED)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncLog,   std::log,   alpaka::math::log,    Range::POSITIVE_ONLY)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncRound, std::round, alpaka::math::round,  Range::UNRESTRICTED)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncRsqrt, rsqrt,      alpaka::math::rsqrt,  Range::POSITIVE_ONLY)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncSin,   std::sin,   alpaka::math::sin,    Range::UNRESTRICTED)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncSqrt,  std::sqrt,  alpaka::math::sqrt,   Range::POSITIVE_AND_ZERO)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncTan,   std::tan,   alpaka::math::tan,    Range::UNRESTRICTED)
ALPAKA_TEST_MATH_OP_FUNCTOR_UNARY(OpFuncTrunc, std::trunc, alpaka::math::trunc,  Range::UNRESTRICTED)


using Operators = std::tuple<
    OpFuncAbs,
    OpFuncAcos,
    OpFuncAsin,
    OpFuncAtan,
    OpFuncCbrt,
    OpFuncCeil,
    OpFuncCos,
    OpFuncErf,
    OpFuncExp,
    OpFuncFloor,
    OpFuncLog,
    OpFuncRound,
    OpFuncRsqrt,
    OpFuncSin,
    OpFuncSqrt,
    OpFuncTan,
    OpFuncTrunc
    >;



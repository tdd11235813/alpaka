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
#define ALPAKA_TEST_MATH_OP_FUNCTOR_BINARY( NAME, STD_OP, ALPAKA_OP, RANGE )\
class NAME\
{\
public:\
    static constexpr Range range = RANGE;\
    template<\
        typename TAcc,\
        typename T>\
    ALPAKA_FN_ACC\
    auto operator() (\
        TAcc const& acc,\
        T const & x,\
        T const & y ) const\
        -> decltype( ALPAKA_OP( acc, x, y ) )\
    {\
        return ALPAKA_OP( acc, x, y );\
    }\
    template < typename T >\
    ALPAKA_FN_HOST\
    auto operator() (\
        T const & x,\
        T const & y ) const\
        -> decltype( STD_OP( x, y ) )\
    {\
        return STD_OP( x, y );\
    }\
    friend std::ostream & operator<< ( std::ostream &out, const NAME &op )\
    {\
    out << #NAME;\
    alpaka::ignore_unused( op );\
    return out;\
    }\
};

//! @enum Range
//! @brief Possible definition ranges.
enum class Range
{
    POSITIVE_ONLY,
    NOT_ZERO,
    X_NOT_ZERO,
    Y_NOT_ZERO,
    UNRESTRICTED
};

// Generate all unary functors.
ALPAKA_TEST_MATH_OP_FUNCTOR_BINARY(
    OpFuncAtan2,
    std::atan2,
    alpaka::math::atan2,
    Range::NOT_ZERO )

ALPAKA_TEST_MATH_OP_FUNCTOR_BINARY(
    OpFuncFmod,
    std::fmod,
    alpaka::math::fmod,
    Range::Y_NOT_ZERO )

ALPAKA_TEST_MATH_OP_FUNCTOR_BINARY(
    OpFuncMax,
    std::max,
    alpaka::math::max,
    Range::Y_NOT_ZERO )

ALPAKA_TEST_MATH_OP_FUNCTOR_BINARY(
    OpFuncMin,
    std::min,
    alpaka::math::min,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR_BINARY(
    OpFuncPow,
    std::pow,
    alpaka::math::pow,
    Range::POSITIVE_ONLY )

ALPAKA_TEST_MATH_OP_FUNCTOR_BINARY(
    OpFuncRemainder,
    std::remainder,
    alpaka::math::remainder,
    Range::POSITIVE_ONLY )

using Operators = std::tuple<
    OpFuncAtan2,
    OpFuncFmod,
    OpFuncMax,
    OpFuncMin,
    OpFuncPow,
    OpFuncRemainder
    >;

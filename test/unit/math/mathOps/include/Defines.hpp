/** Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <cmath>

namespace alpaka {
namespace test {
namespace unit {
namespace math {

    // New types need to be added to the switch-case in DataGen.hpp
    enum class Range
    {
        ONE_NEIGHBOURHOOD,
        POSITIVE_ONLY,
        POSITIVE_AND_ZERO,
        NOT_ZERO,
        UNRESTRICTED
    };

    // New types need to be added to the compute function in mathOps.cpp
    enum class Arity
    {
        UNARY = 1,
        BINARY = 2
    };

    template<typename T, Arity Tarity>
    struct ArgsItem{
        static constexpr Arity arity = Tarity;
        static constexpr int arity_nr = static_cast<int>(Tarity);

        T arg[arity_nr]; // represents arg0, arg1, ...
    };

    template< typename T >
    auto rsqrt( T const & arg ) -> decltype( std::sqrt( arg ) )
    {
        return 1 / std::sqrt( arg );
    }

} // math
} // unit
} // test
} // alpaka

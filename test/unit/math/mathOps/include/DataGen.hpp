/** Copyright 2019 Jakob Krude, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "Defines.hpp"
#include <random>
#include <limits>

namespace alpaka {
namespace test {
namespace unit {
namespace math {


    /**
     * Fills buffer with random numbers (host-only).
     *
     * @tparam TData The used data-type.
     * @tparam TBuffer Buffer from custom Buffer-class.
     * @tparam TFunctor The used Functor-type.
     * @param buffer The buffer that should be filled.
     * @param functor The Functor, needed for ranges.
     * @param seed The used seed.
     */
    template<
        typename TData,
        typename TArgs,
        typename TFunctor>
    auto fillWithRndArgs(
        TArgs  & args,
        TFunctor functor,
        unsigned long const & seed
    ) -> void
    {
        /*
         * Each "sub-buffer" if filled with zero and/or max and/or lowest,
         * depending on the specified range (at [0] - [2]).
         *
         * The default switch-case throws a runtime-error.
         *
         * This function is easily extendable. It is only necessary to add extra
         * definitions in the switch case, for more Range-types.
         */
        static_assert( TArgs::value_type::arity == TFunctor::arity,
                       "Buffer properties must match TFunctor::arity" );
        static_assert( TArgs::capacity > 2, "Set of args must provide >2 entries." );
        constexpr auto max = std::numeric_limits< TData >::max();
        constexpr auto low = std::numeric_limits< TData >::lowest();
        std::default_random_engine eng{ seed };

        // These pseudo-random numbers are implementation/platform specific!
        std::uniform_real_distribution< TData > dist(0,1000);
        std::uniform_real_distribution< TData > distOne(-1,1);

        for(size_t k = 0; k < TFunctor::arity_nr; ++k)
        {
            switch( functor.ranges[k] )
            {
                case Range::ONE_NEIGHBOURHOOD:
                    for(size_t i = 0; i < TArgs::capacity; ++i) {
                        args(i).arg[k] = distOne( eng );
                    }
                break;

                case Range::POSITIVE_ONLY:
                    args(0).arg[k] = max;
                    for(size_t i = 1; i < TArgs::capacity; ++i) {
                        args(i).arg[k] = dist( eng )+1.0;
                    }
                break;

                case Range::POSITIVE_AND_ZERO:
                    args(0).arg[k] = 0.0;
                    args(1).arg[k] = max;
                    for(size_t i = 2; i < TArgs::capacity; ++i) {
                        args(i).arg[k] = dist( eng );
                    }
                break;

                case Range::NOT_ZERO:
                    args(0).arg[k] = max;
                    args(1).arg[k] = low;
                    for(size_t i = 2; i < TArgs::capacity; ++i) {
                        TData arg;
                        do
                        {
                            arg = dist( eng );
                        } while( arg == static_cast<TData>(0.0) );
                        if(i % 2 == 0)
                            args(i).arg[k] = arg;
                        else
                            args(i).arg[k] = -arg;
                    }
                break;

                case Range::UNRESTRICTED:
                    args(0).arg[k] = 0.0;
                    args(1).arg[k] = max;
                    args(2).arg[k] = low;
                    for(size_t i = 3; i < TArgs::capacity; ++i) {
                        if(i % 2 == 0)
                            args(i).arg[k] = dist(eng);
                        else
                            args(i).arg[k] = -dist(eng);
                    }
                break;

                default:
                    throw std::runtime_error( "Unsupported Range" );
            }
        }
    }

} // math
} // unit
} // test
} // alpaka

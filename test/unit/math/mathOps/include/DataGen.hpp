/** Copyright 2019 Axel Huebl, Benjamin Worpitz
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

namespace test
{
    unsigned long generateSeed(); // Avoid clang, undeclared-function-warning.
    auto generateSeed () -> unsigned long
    {
        std::random_device rd {};
        auto seed = rd();
        return seed;
    }
/*+
 * @fn fillWithRndArgs
 * @tparam TData The used data-type.
 * @tparam TBuffer Buffer from custom Buffer-class.
 * @tparam TFunctor The used Functor-type.
 * @param buffer The buffer that should be filled.
 * @param functor The Functor, needed for ranges.
 * @param seed The used seed.
 */
    template<
        typename TData,
        typename TBuffer,
        typename TFunctor>
    auto fillWithRndArgs(
        TBuffer  & buffer,
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
        static_assert( TBuffer::count == static_cast< int >(
            TFunctor::arity ),
                "Each sub-buffer needs a range of type Range::");
        constexpr auto max = std::numeric_limits< TData >::max();
        constexpr auto low = std::numeric_limits< TData >::lowest();
        std::default_random_engine eng{ seed };

        // These pseudo-random numbers are implementation/platform specific!
        std::uniform_real_distribution< TData > dist(
            0,
            1000
        );

        std::uniform_real_distribution< TData > distOne(
            -1,
            1
        );

        for(size_t i = 0; i < TBuffer::count; ++i)
        {
            switch( functor.ranges[i] )
            {
                case Range::ONE_NEIGHBOURHOOD:
                    buffer.setArg( 0, 0, i );
                    for( size_t j = 1; j < TBuffer::extent; ++j )
                        buffer.setArg( distOne( eng ), j, i );
                break;

                case Range::POSITIVE_ONLY:
                    buffer.setArg( max, 0, i );
                    for( size_t j = 1; j < TBuffer::extent; ++j )
                        buffer.setArg( dist( eng ) + 1, j , i );
                break;

                case Range::POSITIVE_AND_ZERO:
                    buffer.setArg( 0, 0, i );
                    buffer.setArg( max, 1, i );
                    for( size_t j = 2; j < TBuffer::extent; ++j )
                        buffer.setArg( dist( eng ), j , i );
                break;

                case Range::NOT_ZERO:
                    buffer.setArg( max, 0, i );
                    buffer.setArg( low, 1, i );
                    for( size_t j = 2; j < TBuffer::extent; ++j )
                    {
                        TData arg;
                        do
                        {
                            arg = dist( eng );
                          // Avoid comparing floats/doubles warning.
                        } while( static_cast< int > ( arg ) == 0 );
                        if(j % 2 == 0)
                            buffer.setArg( arg, j , i );
                        else
                            buffer.setArg( -arg, j, i );
                    }
                break;

                case Range::UNRESTRICTED:
                    buffer.setArg( 0, 0, i );
                    buffer.setArg( max, 1, i );
                    buffer.setArg( low, 2, i );
                    for( size_t j = 3; j < TBuffer::extent; ++j )
                    {
                        if( j % 2 == 0 )
                            buffer.setArg( dist( eng ), j , i );
                        else
                            buffer.setArg( -dist( eng ), j, i );
                    }
                break;

                default:
                    throw std::runtime_error( "Unsupported Range" );
            }
        }
    }
}

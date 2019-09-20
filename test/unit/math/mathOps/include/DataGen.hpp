/** Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "Defines.hpp"
#include "Buffer.hpp"
#include <random>

/**
 * @namespace test
 * @brief Only contains fillWithRndArgs.
 * @fn fillWithRndArgs
 * @tparam Data The used Buffer-type.
 * @param buffer The buffer that should be filled.
 * @param size The size of the used buffer.
 * @param range The Range, around Zero, for the data.
 */

namespace test
{
    auto generateSeed () -> unsigned long
    {
        std::random_device rd {};
        auto seed = rd();
        return seed;
    }

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
        static_assert( TBuffer::count == static_cast<int>(TFunctor::arity), "");

        std::default_random_engine eng { seed };
        // These pseudo-random numbers are implementation/platform specific!
        int const distRange = 10;
        std::uniform_real_distribution< TData > dist(
            0,
            distRange
        );
        for(size_t i = 0; i < TBuffer::count; ++i)
        {
            switch( functor.ranges[i] )
            {
                case Range::POSITIVE_ONLY:
                    for( size_t j = 0; j < TBuffer::extent; ++j )
                        buffer.setArg( dist(eng) + 1, j , i );
                break;

                case Range::POSITIVE_AND_ZERO:
                    for( size_t j = 0; j < TBuffer::extent; ++j )
                        buffer.setArg( dist(eng), j , i );
                break;

                case Range::NOT_ZERO:
                    for( size_t j = 0; j < TBuffer::extent; ++j )
                    {
                        TData arg;
                        do{
                            arg = dist(eng);
                        }while(arg == 0);
                        buffer.setArg( arg, j , i );
                    }
                break;

                case Range::UNRESTRICTED:
                    for( size_t j = 0; j < TBuffer::extent; ++j )
                    {
                        buffer.setArg( dist(eng)- distRange / 2, j , i );
                    }
                break;

                default:
                    throw std::runtime_error( "Unsupported Range" );
            }
        }

    }
}
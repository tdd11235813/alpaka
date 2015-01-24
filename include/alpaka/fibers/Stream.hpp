/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/fibers/AccFibersFwd.hpp>   // AccFibers

#include <alpaka/host/Stream.hpp>           // StreamHost

namespace alpaka
{
    namespace fibers
    {
        namespace detail
        {
            //#############################################################################
            //! The fibers accelerator stream.
            //#############################################################################
            class StreamFibers :
                public host::detail::StreamHost
            {};
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The fibers accelerator stream accelerator type trait specialization.
            //#############################################################################
            template<>
            struct GetAcc<
                fibers::detail::StreamFibers>
            {
                using type = AccFibers;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The fibers accelerator stream type trait specialization.
            //#############################################################################
            template<>
            struct GetStream<
                AccFibers>
            {
                using type = alpaka::fibers::detail::StreamFibers;
            };
        }
    }
}

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

#ifdef ALPAKA_ACC_HIP_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/idx/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Hip.hpp>
#include <alpaka/core/Positioning.hpp>

//#include <boost/core/ignore_unused.hpp>   // boost::ignore_unused

namespace alpaka
{
    namespace idx
    {
        namespace gb
        {
            //#############################################################################
            //! The HIP accelerator ND index provider.
            template<
                typename TDim,
                typename TIdx>
            class IdxGbHipBuiltIn
            {
            public:
                using IdxGbBase = IdxGbHipBuiltIn;

                //-----------------------------------------------------------------------------
                IdxGbHipBuiltIn() = default;
                //-----------------------------------------------------------------------------
                IdxGbHipBuiltIn(IdxGbHipBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                IdxGbHipBuiltIn(IdxGbHipBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxGbHipBuiltIn const & ) -> IdxGbHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxGbHipBuiltIn &&) -> IdxGbHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxGbHipBuiltIn() = default;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator index dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                idx::gb::IdxGbHipBuiltIn<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator grid block index get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TIdx>
            struct GetIdx<
                idx::gb::IdxGbHipBuiltIn<TDim, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FN_ACC_HIP_ONLY static auto getIdx(
                    idx::gb::IdxGbHipBuiltIn<TDim, TIdx> const & /*idx*/,
                    TWorkDiv const &)
                -> vec::Vec<TDim, TIdx>
                {
                    //boost::ignore_unused(idx);
                    return vec::cast<TIdx>(offset::getOffsetVecEnd<TDim>(dim3(hipBlockIdx_x, hipBlockIdx_y, hipBlockIdx_z)));
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator grid block index idx type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::gb::IdxGbHipBuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif

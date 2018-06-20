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

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_*, __HIPCC__

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
        namespace bt
        {
            //#############################################################################
            //! The HIP accelerator ND index provider.
            //#############################################################################
            template<
                typename TDim,
                typename TIdx>
            class IdxBtHipBuiltIn
            {
            public:
                using IdxBtBase = IdxBtHipBuiltIn;

                //-----------------------------------------------------------------------------
                IdxBtHipBuiltIn() = default;
                //-----------------------------------------------------------------------------
                IdxBtHipBuiltIn(IdxBtHipBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                IdxBtHipBuiltIn(IdxBtHipBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxBtHipBuiltIn const & ) -> IdxBtHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxBtHipBuiltIn &&) -> IdxBtHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxBtHipBuiltIn() = default;
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
                idx::bt::IdxBtHipBuiltIn<TDim, TIdx>>
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
            //! The GPU HIP accelerator block thread index get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetIdx<
                idx::bt::IdxBtHipBuiltIn<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                ALPAKA_FN_ACC_HIP_ONLY static auto getIdx(
                    idx::bt::IdxBtHipBuiltIn<TDim, TIdx> const & /*idx*/,
                    TWorkDiv const &)
                -> vec::Vec<TDim, TIdx>
                {
                    //boost::ignore_unused(idx);
                    return vec::cast<TIdx>(offset::getOffsetVecEnd<TDim>(dim3(hipThreadIdx_x,hipThreadIdx_y,hipThreadIdx_z)));
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator block thread index idx type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::bt::IdxBtHipBuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif

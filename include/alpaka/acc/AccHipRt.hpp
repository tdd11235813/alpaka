/**
* \file
* Copyright 2014-2016 Benjamin Worpitz, Rene Widera
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

// Base classes.
#include <alpaka/workdiv/WorkDivHipBuiltIn.hpp>
#include <alpaka/idx/gb/IdxGbHipBuiltIn.hpp>
#include <alpaka/idx/bt/IdxBtHipBuiltIn.hpp>
#include <alpaka/atomic/AtomicHipBuiltIn.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/math/MathHipBuiltIn.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynHipBuiltIn.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStHipBuiltIn.hpp>
#include <alpaka/block/sync/BlockSyncHipBuiltIn.hpp>
#include <alpaka/rand/RandHipRand.hpp>
#include <alpaka/time/TimeHipBuiltIn.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/dev/DevHipRt.hpp>
#include <alpaka/core/Hip.hpp>

#include <boost/predef.h>

#include <typeinfo>

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecGpuHipRt;
    }
    namespace acc
    {
        //#############################################################################
        //! The GPU HIP accelerator.
        //!
        //! This accelerator allows parallel kernel execution on devices supporting HIP or HCC
        template<
            typename TDim,
            typename TIdx>
        class AccHipRt final :
            public workdiv::WorkDivHipBuiltIn<TDim, TIdx>,
            public idx::gb::IdxGbHipBuiltIn<TDim, TIdx>,
            public idx::bt::IdxBtHipBuiltIn<TDim, TIdx>,
            public atomic::AtomicHierarchy<
                atomic::AtomicHipBuiltIn, // grid atomics
                atomic::AtomicHipBuiltIn, // block atomics
                atomic::AtomicHipBuiltIn  // thread atomics
            >,
            public math::MathHipBuiltIn,
            public block::shared::dyn::BlockSharedMemDynHipBuiltIn,
            public block::shared::st::BlockSharedMemStHipBuiltIn,
            public block::sync::BlockSyncHipBuiltIn,
            public rand::RandHipRand,
            public time::TimeHipBuiltIn
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY AccHipRt(
                vec::Vec<TDim, TIdx> const & threadElemExtent) :
                    workdiv::WorkDivHipBuiltIn<TDim, TIdx>(threadElemExtent),
                    idx::gb::IdxGbHipBuiltIn<TDim, TIdx>(),
                    idx::bt::IdxBtHipBuiltIn<TDim, TIdx>(),
                    atomic::AtomicHierarchy<
                        atomic::AtomicHipBuiltIn, // atomics between grids
                        atomic::AtomicHipBuiltIn, // atomics between blocks
                        atomic::AtomicHipBuiltIn  // atomics between threads
                    >(),
                    math::MathHipBuiltIn(),
                    block::shared::dyn::BlockSharedMemDynHipBuiltIn(),
                    block::shared::st::BlockSharedMemStHipBuiltIn(),
                    block::sync::BlockSyncHipBuiltIn(),
                    rand::RandHipRand(),
                    time::TimeHipBuiltIn()
            {}

        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY AccHipRt(AccHipRt const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY AccHipRt(AccHipRt &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY auto operator=(AccHipRt const &) -> AccHipRt & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY auto operator=(AccHipRt &&) -> AccHipRt & = delete;
            //-----------------------------------------------------------------------------
            ~AccHipRt() = default;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct AccType<
                acc::AccHipRt<TDim, TIdx>>
            {
                using type = acc::AccHipRt<TDim, TIdx>;
            };
            //#############################################################################
            //! The GPU HIP accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccDevProps<
                acc::AccHipRt<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevHipRt const & dev)
                -> acc::AccDevProps<TDim, TIdx>
                {
                    hipDeviceProp_t hipDevProp;
                    ALPAKA_HIP_RT_CHECK(hipGetDeviceProperties(
                        &hipDevProp,
                        dev.m_iDevice));

                    return {
                        // m_multiProcessorCount
                        static_cast<TIdx>(hipDevProp.multiProcessorCount),
                        // m_gridBlockExtentMax
                        extent::getExtentVecEnd<TDim>(
                            vec::Vec<dim::DimInt<3u>, TIdx>(
                                static_cast<TIdx>(hipDevProp.maxGridSize[2]),
                                static_cast<TIdx>(hipDevProp.maxGridSize[1]),
                                static_cast<TIdx>(hipDevProp.maxGridSize[0]))),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        extent::getExtentVecEnd<TDim>(
                            vec::Vec<dim::DimInt<3u>, TIdx>(
                                static_cast<TIdx>(hipDevProp.maxThreadsDim[2]),
                                static_cast<TIdx>(hipDevProp.maxThreadsDim[1]),
                                static_cast<TIdx>(hipDevProp.maxThreadsDim[0]))),
                        // m_blockThreadCountMax
                        static_cast<TIdx>(hipDevProp.maxThreadsPerBlock),
                        // m_threadElemExtentMax
                        vec::Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max()};
                }
            };
            //#############################################################################
            //! The GPU Hip accelerator name trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccName<
                acc::AccHipRt<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccHipRt<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator device type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DevType<
                acc::AccHipRt<TDim, TIdx>>
            {
                using type = dev::DevHipRt;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                acc::AccHipRt<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace kernel
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator executor type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TWorkDiv,
                typename TKernelFnObj,
                typename... TArgs>
            struct CreateTaskExec<
                acc::AccHipRt<TDim, TIdx>,
                TWorkDiv,
                TKernelFnObj,
                TArgs...>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto createTaskExec(
                    TWorkDiv const & workDiv,
                    TKernelFnObj const & kernelFnObj,
                    TArgs const & ... args)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> exec::ExecGpuHipRt<
                    TDim,
                    TIdx,
                    TKernelFnObj,
                    TArgs...>
#endif
                {
                    return
                        exec::ExecGpuHipRt<
                            TDim,
                            TIdx,
                            TKernelFnObj,
                            TArgs...>(
                                workDiv,
                                kernelFnObj,
                                args...);
                }
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU HIP executor platform type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct PltfType<
                acc::AccHipRt<TDim, TIdx>>
            {
                using type = pltf::PltfHipRt;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                acc::AccHipRt<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
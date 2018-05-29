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

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/queue/QueueHipRtSync.hpp>
#include <alpaka/queue/QueueHipRtAsync.hpp>

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevHipRt.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/queue/QueueHipRtAsync.hpp>
#include <alpaka/queue/QueueHipRtSync.hpp>

#include <alpaka/core/Hip.hpp>

#include <cassert>

namespace alpaka
{
    namespace mem
    {
        namespace view
        {
            namespace hip
            {
                namespace detail
                {
                    //#############################################################################
                    //! The HIP memory copy trait.
                    //#############################################################################
                    template<
                        typename TDim,
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyHip;

                    //#############################################################################
                    //! The 1D HIP memory copy trait.
                    //#############################################################################
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyHip<
                        dim::DimInt<1>,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        // TODO: Maybe check for Size of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, typename std::remove_const<elem::Elem<TViewSrc>>::type>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Idx = idx::Idx<TExtent>;

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopyHip(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
                            hipMemcpyKind const & hipMemCpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_hipMemCpyKind(hipMemCpyKind),
                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_extentWidth(extent::getWidth(extent)),
                                m_dstWidth(static_cast<Idx>(extent::getWidth(viewDst))),
                                m_srcWidth(static_cast<Idx>(extent::getWidth(viewSrc))),
#endif
                                m_extentWidthBytes(extent::getWidth(extent) * static_cast<Idx>(sizeof(elem::Elem<TViewDst>))),
                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_extentWidth <= m_dstWidth);
                            assert(m_extentWidth <= m_srcWidth);
#endif
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " ddev: " << m_iDstDevice
                                << " ew: " << m_extentWidth
                                << " ewb: " << m_extentWidthBytes
                                << " dw: " << m_dstWidth
                                << " dptr: " << m_dstMemNative
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_srcWidth
                                << " sptr: " << m_srcMemNative
                                << std::endl;
                        }
#endif
                        hipMemcpyKind m_hipMemCpyKind;
                        int m_iDstDevice;
                        int m_iSrcDevice;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Idx m_extentWidth;
                        Idx m_dstWidth;
                        Idx m_srcWidth;
#endif
                        Idx m_extentWidthBytes;
                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };
                    //#############################################################################
                    //! The 2D HIP memory copy trait.
                    //#############################################################################
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyHip<
                        dim::DimInt<2>,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        // TODO: Maybe check for Size of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, typename std::remove_const<elem::Elem<TViewSrc>>::type>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Idx = idx::Idx<TExtent>;

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopyHip(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
                            hipMemcpyKind const & hipMemCpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_hipMemCpyKind(hipMemCpyKind),
                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_extentWidth(extent::getWidth(extent)),
#endif
                                m_extentWidthBytes(extent::getWidth(extent) * static_cast<Idx>(sizeof(elem::Elem<TViewDst>))),
                                m_dstWidth(static_cast<Idx>(extent::getWidth(viewDst))),      // required for 3D peer copy
                                m_srcWidth(static_cast<Idx>(extent::getWidth(viewSrc))),      // required for 3D peer copy

                                m_extentHeight(extent::getHeight(extent)),
                                m_dstHeight(static_cast<Idx>(extent::getHeight(viewDst))),    // required for 3D peer copy
                                m_srcHeight(static_cast<Idx>(extent::getHeight(viewSrc))),    // required for 3D peer copy

                                m_dstpitchBytesX(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - 1u>(viewDst))),
                                m_srcpitchBytesX(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - 1u>(viewSrc))),
                                m_dstPitchBytesY(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - (2u % dim::Dim<TViewDst>::value)>(viewDst))),
                                m_srcPitchBytesY(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - (2u % dim::Dim<TViewDst>::value)>(viewSrc))),

                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_extentWidth <= m_dstWidth);
                            assert(m_extentHeight <= m_dstHeight);
                            assert(m_extentWidth <= m_srcWidth);
                            assert(m_extentHeight <= m_srcHeight);
                            assert(m_extentWidthBytes <= m_dstpitchBytesX);
                            assert(m_extentWidthBytes <= m_srcpitchBytesX);
#endif
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " ew: " << m_extentWidth
                                << " eh: " << m_extentHeight
                                << " ewb: " << m_extentWidthBytes
                                << " ddev: " << m_iDstDevice
                                << " dw: " << m_dstWidth
                                << " dh: " << m_dstHeight
                                << " dptr: " << m_dstMemNative
                                << " dpitchb: " << m_dstpitchBytesX
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_srcWidth
                                << " sh: " << m_srcHeight
                                << " sptr: " << m_srcMemNative
                                << " spitchb: " << m_srcpitchBytesX
                                << std::endl;
                        }
#endif
                        hipMemcpyKind m_hipMemCpyKind;
                        int m_iDstDevice;
                        int m_iSrcDevice;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Idx m_extentWidth;
#endif
                        Idx m_extentWidthBytes;
                        Idx m_dstWidth;          // required for 3D peer copy
                        Idx m_srcWidth;          // required for 3D peer copy

                        Idx m_extentHeight;
                        Idx m_dstHeight;         // required for 3D peer copy
                        Idx m_srcHeight;         // required for 3D peer copy

                        Idx m_dstpitchBytesX;
                        Idx m_srcpitchBytesX;
                        Idx m_dstPitchBytesY;
                        Idx m_srcPitchBytesY;


                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };
                    //#############################################################################
                    //! The 3D HIP memory copy trait.
                    //#############################################################################
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyHip<
                        dim::DimInt<3>,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        // TODO: Maybe check for Size of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, typename std::remove_const<elem::Elem<TViewSrc>>::type>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Idx = idx::Idx<TExtent>;

                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopyHip(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
                            hipMemcpyKind const & hipMemCpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_hipMemCpyKind(hipMemCpyKind),

                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),

                                m_extentWidth(extent::getWidth(extent)),
                                m_extentWidthBytes(m_extentWidth * static_cast<Idx>(sizeof(elem::Elem<TViewDst>))),
                                m_dstWidth(static_cast<Idx>(extent::getWidth(viewDst))),
                                m_srcWidth(static_cast<Idx>(extent::getWidth(viewSrc))),

                                m_extentHeight(extent::getHeight(extent)),
                                m_dstHeight(static_cast<Idx>(extent::getHeight(viewDst))),
                                m_srcHeight(static_cast<Idx>(extent::getHeight(viewSrc))),

                                m_extentDepth(extent::getDepth(extent)),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_dstDepth(static_cast<Idx>(extent::getDepth(viewDst))),
                                m_srcDepth(static_cast<Idx>(extent::getDepth(viewSrc))),
#endif
                                m_dstpitchBytesX(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - 1u>(viewDst))),
                                m_srcpitchBytesX(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - 1u>(viewSrc))),
                                m_dstPitchBytesY(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - (2u % dim::Dim<TViewDst>::value)>(viewDst))),
                                m_srcPitchBytesY(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - (2u % dim::Dim<TViewDst>::value)>(viewSrc))),


                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_extentWidth <= m_dstWidth);
                            assert(m_extentHeight <= m_dstHeight);
                            assert(m_extentDepth <= m_dstDepth);
                            assert(m_extentWidth <= m_srcWidth);
                            assert(m_extentHeight <= m_srcHeight);
                            assert(m_extentDepth <= m_srcDepth);
                            assert(m_extentWidthBytes <= m_dstpitchBytesX);
                            assert(m_extentWidthBytes <= m_srcpitchBytesX);
#endif
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " ew: " << m_extentWidth
                                << " eh: " << m_extentHeight
                                << " ed: " << m_extentDepth
                                << " ewb: " << m_extentWidthBytes
                                << " ddev: " << m_iDstDevice
                                << " dw: " << m_dstWidth
                                << " dh: " << m_dstHeight
                                << " dd: " << m_dstDepth
                                << " dptr: " << m_dstMemNative
                                << " dpitchb: " << m_dstpitchBytesX
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_srcWidth
                                << " sh: " << m_srcHeight
                                << " sd: " << m_srcDepth
                                << " sptr: " << m_srcMemNative
                                << " spitchb: " << m_srcpitchBytesX
                                << std::endl;
                        }
#endif
                        hipMemcpyKind m_hipMemCpyKind;

                        int m_iDstDevice;
                        int m_iSrcDevice;

                        Idx m_extentWidth;
                        Idx m_extentWidthBytes;
                        Idx m_dstWidth;
                        Idx m_srcWidth;

                        Idx m_extentHeight;
                        Idx m_dstHeight;
                        Idx m_srcHeight;

                        Idx m_extentDepth;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Idx m_dstDepth;
                        Idx m_srcDepth;
#endif
                        Idx m_dstpitchBytesX;
                        Idx m_srcpitchBytesX;
                        Idx m_dstPitchBytesY;
                        Idx m_srcPitchBytesY;

                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };
                }
            }

            //-----------------------------------------------------------------------------
            // Trait specializations for CreateTaskCopy.
            //-----------------------------------------------------------------------------
            namespace traits
            {
                //#############################################################################
                //! The HIP to CPU memory copy trait specialization.
                //#############################################################################
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevCpu,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::hip::detail::TaskCopyHip<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const iDevice(
                            dev::getDev(viewSrc).m_iDevice);

                        return
                            mem::view::hip::detail::TaskCopyHip<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    hipMemcpyDeviceToHost,
                                    iDevice,
                                    iDevice);
                    }
                };
                //#############################################################################
                //! The CPU to HIP memory copy trait specialization.
                //#############################################################################
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevHipRt,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::hip::detail::TaskCopyHip<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const iDevice(
                            dev::getDev(viewDst).m_iDevice);

                        return
                            mem::view::hip::detail::TaskCopyHip<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    hipMemcpyHostToDevice,
                                    iDevice,
                                    iDevice);
                    }
                };
                //#############################################################################
                //! The HIP to HIP memory copy trait specialization.
                //#############################################################################
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevHipRt,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::hip::detail::TaskCopyHip<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        return
                            mem::view::hip::detail::TaskCopyHip<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    hipMemcpyDeviceToDevice,
                                    dev::getDev(viewDst).m_iDevice,
                                    dev::getDev(viewSrc).m_iDevice);
                    }
                };
            }
            namespace hip
            {
                namespace detail
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    //~ template<
                        //~ typename TExtent,
                        //~ typename TViewSrc,
                        //~ typename TViewDst>
                    //~ ALPAKA_FN_HOST static auto buildHipMemcpy3DParms(
                        //~ mem::view::hip::detail::TaskCopy<dim::DimInt<3>, TViewDst, TViewSrc, TExtent> const & task)
                    //~ -> hipMemcpy3DParms
                    //~ {
                        //~ ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        //~ auto const & extentWidthBytes(task.m_extentWidthBytes);
                        //~ auto const & dstWidth(task.m_dstWidth);
                        //~ auto const & srcWidth(task.m_srcWidth);

                        //~ auto const & extentHeight(task.m_extentHeight);
                        //~ //auto const & dstHeight(task.m_dstHeight);
                        //~ //auto const & srcHeight(task.m_srcHeight);

                        //~ auto const & extentDepth(task.m_extentDepth);

                        //~ auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        //~ auto const & srcPitchBytesX(task.m_srcpitchBytesX);
                        //~ auto const & dstPitchBytesY(task.m_dstPitchBytesY);
                        //~ auto const & srcPitchBytesY(task.m_srcPitchBytesY);

                        //~ auto const & dstNativePtr(task.m_dstMemNative);
                        //~ auto const & srcNativePtr(task.m_srcMemNative);

                        //~ // Fill HIP parameter structure.
                        //~ hipMemcpy3DParms hipMemCpy3DParms = {};
                        //~ //hipMemCpy3DParms.srcArray;     // Either srcArray or srcPtr.
                        //~ //hipMemCpy3DParms.srcPos;       // Optional. Offset in bytes.

                        //~ hipMemCpy3DParms.srcPtr =
                            //~ make_hipPitchedPtr(
                                //~ const_cast<void *>(srcNativePtr),
                                //~ static_cast<std::size_t>(srcPitchBytesX),
                                //~ static_cast<std::size_t>(srcWidth),
                                //~ static_cast<std::size_t>(srcPitchBytesY/srcPitchBytesX));
                        //~ //hipMemCpy3DParms.dstArray;     // Either dstArray or dstPtr.
                        //~ //hipMemCpy3DParms.dstPos;       // Optional. Offset in bytes.
                        //~ hipMemCpy3DParms.dstPtr =
                            //~ make_hipPitchedPtr(
                                //~ dstNativePtr,
                                //~ static_cast<std::size_t>(dstPitchBytesX),
                                //~ static_cast<std::size_t>(dstWidth),
                                //~ static_cast<std::size_t>(dstPitchBytesY/dstPitchBytesX));
                        //~ hipMemCpy3DParms.extent =
                            //~ make_hipExtent(
                                //~ static_cast<std::size_t>(extentWidthBytes),
                                //~ static_cast<std::size_t>(extentHeight),
                                //~ static_cast<std::size_t>(extentDepth));
                        //~ hipMemCpy3DParms.kind = task.m_hipMemCpyKind;

                        //~ return hipMemCpy3DParms;
                    //~ }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------

                    //~ template<
                        //~ typename TViewDst,
                        //~ typename TViewSrc,
                        //~ typename TExtent>
                    //~ ALPAKA_FN_HOST static auto buildHipMemcpy3DPeerParms(
                        //~ mem::view::hip::detail::TaskCopy<dim::DimInt<2>, TViewDst, TViewSrc, TExtent> const & task)
                    //~ -> hipMemcpy3DPeerParms
                    //~ {
                        //~ ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        //~ auto const & iDstDev(task.m_iDstDevice);
                        //~ auto const & iSrcDev(task.m_iSrcDevice);

                        //~ auto const & extentWidthBytes(task.m_extentWidthBytes);
                        //~ auto const & dstWidth(task.m_dstWidth);
                        //~ auto const & srcWidth(task.m_srcWidth);

                        //~ auto const & extentHeight(task.m_extentHeight);
                        //~ //auto const & dstHeight(task.m_dstHeight);
                        //~ //auto const & srcHeight(task.m_srcHeight);

                        //~ auto const extentDepth(1u);

                        //~ auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        //~ auto const & srcPitchBytesX(task.m_srcpitchBytesX);
                        //~ auto const & dstPitchBytesY(task.m_dstPitchBytesY);
                        //~ auto const & srcPitchBytesY(task.m_srcPitchBytesY);

                        //~ auto const & dstNativePtr(task.m_dstMemNative);
                        //~ auto const & srcNativePtr(task.m_srcMemNative);

                        //~ // Fill HIP parameter structure.
                        //~ hipMemcpy3DPeerParms hipMemCpy3DPeerParms = {};
                        //~ //hipMemCpy3DPeerParms.dstArray;     // Either dstArray or dstPtr.
                        //~ hipMemCpy3DPeerParms.dstDevice = iDstDev;
                        //~ //hipMemCpy3DPeerParms.dstPos;       // Optional. Offset in bytes.
                        //~ hipMemCpy3DPeerParms.dstPtr =
                            //~ make_hipPitchedPtr(
                                //~ dstNativePtr,
                                //~ static_cast<std::size_t>(dstPitchBytesX),
                                //~ static_cast<std::size_t>(dstWidth),
                                //~ static_cast<std::size_t>(dstPitchBytesY/dstPitchBytesX));
                        //~ hipMemCpy3DPeerParms.extent =
                            //~ make_hipExtent(
                                //~ static_cast<std::size_t>(extentWidthBytes),
                                //~ static_cast<std::size_t>(extentHeight),
                                //~ static_cast<std::size_t>(extentDepth));
                        //~ //hipMemCpy3DPeerParms.srcArray;     // Either srcArray or srcPtr.
                        //~ hipMemCpy3DPeerParms.srcDevice = iSrcDev;
                        //~ //hipMemCpy3DPeerParms.srcPos;       // Optional. Offset in bytes.
                        //~ hipMemCpy3DPeerParms.srcPtr =
                            //~ make_hipPitchedPtr(
                                //~ const_cast<void *>(srcNativePtr),
                                //~ static_cast<std::size_t>(srcPitchBytesX),
                                //~ static_cast<std::size_t>(srcWidth),
                                //~ static_cast<std::size_t>(srcPitchBytesY/srcPitchBytesX));

                        //~ return hipMemCpy3DPeerParms;
                    //~ }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    //~ template<
                        //~ typename TViewDst,
                        //~ typename TViewSrc,
                        //~ typename TExtent>
                    //~ ALPAKA_FN_HOST static auto buildHipMemcpy3DPeerParms(
                        //~ mem::view::hip::detail::TaskCopy<dim::DimInt<3>, TViewDst, TViewSrc, TExtent> const & task)
                    //~ -> hipMemcpy3DPeerParms
                    //~ {
                        //~ ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        //~ auto const & iDstDev(task.m_iDstDevice);
                        //~ auto const & iSrcDev(task.m_iSrcDevice);

                        //~ auto const & extentWidthBytes(task.m_extentWidthBytes);
                        //~ auto const & dstWidth(task.m_dstWidth);
                        //~ auto const & srcWidth(task.m_srcWidth);

                        //~ auto const & extentHeight(task.m_extentHeight);
                        //~ //auto const & dstHeight(task.m_dstHeight);
                        //~ //auto const & srcHeight(task.m_srcHeight);

                        //~ auto const & extentDepth(task.m_extentDepth);

                        //~ auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        //~ auto const & srcPitchBytesX(task.m_srcpitchBytesX);
                        //~ auto const & dstPitchBytesY(task.m_dstPitchBytesY);
                        //~ auto const & srcPitchBytesY(task.m_srcPitchBytesY);

                        //~ auto const & dstNativePtr(task.m_dstMemNative);
                        //~ auto const & srcNativePtr(task.m_srcMemNative);

                        //~ // Fill HIP parameter structure.
                        //~ hipMemcpy3DPeerParms hipMemCpy3DPeerParms = {};
                        //~ //hipMemCpy3DPeerParms.dstArray;     // Either dstArray or dstPtr.
                        //~ hipMemCpy3DPeerParms.dstDevice = iDstDev;
                        //~ //hipMemCpy3DPeerParms.dstPos;       // Optional. Offset in bytes.
                        //~ hipMemCpy3DPeerParms.dstPtr =
                            //~ make_hipPitchedPtr(
                                //~ dstNativePtr,
                                //~ static_cast<std::size_t>(dstPitchBytesX),
                                //~ static_cast<std::size_t>(dstWidth),
                                //~ static_cast<std::size_t>(dstPitchBytesY/dstPitchBytesX));
                        //~ hipMemCpy3DPeerParms.extent =
                            //~ make_hipExtent(
                                //~ static_cast<std::size_t>(extentWidthBytes),
                                //~ static_cast<std::size_t>(extentHeight),
                                //~ static_cast<std::size_t>(extentDepth));
                        //~ //hipMemCpy3DPeerParms.srcArray;     // Either srcArray or srcPtr.
                        //~ hipMemCpy3DPeerParms.srcDevice = iSrcDev;
                        //~ //hipMemCpy3DPeerParms.srcPos;       // Optional. Offset in bytes.
                        //~ hipMemCpy3DPeerParms.srcPtr =
                            //~ make_hipPitchedPtr(
                                //~ const_cast<void *>(srcNativePtr),
                                //~ static_cast<std::size_t>(srcPitchBytesX),
                                //~ static_cast<std::size_t>(srcWidth),
                                //~ static_cast<std::size_t>(srcPitchBytesY/srcPitchBytesX));

                        //~ return hipMemCpy3DPeerParms;
                    //~ }
                }
            }
        }
    }

    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP async device queue 1D copy enqueue trait specialization.
            //#############################################################################
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueHipRtAsync,
                mem::view::hip::detail::TaskCopyHip<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueHipRtAsync & queue,
                    mem::view::hip::detail::TaskCopyHip<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    auto const & extentWidthBytes(task.m_extentWidthBytes);

                    auto const & dstNativePtr(task.m_dstMemNative);
                    auto const & srcNativePtr(task.m_srcMemNative);

                    if(iDstDev == iSrcDev)
                    {
                        auto const & hipMemCpyKind(task.m_hipMemCpyKind);

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_HIP_RT_CHECK(
                            hipMemcpyAsync(
                                dstNativePtr,
                                srcNativePtr,
                                static_cast<std::size_t>(extentWidthBytes),
                                hipMemCpyKind,
                                queue.m_spQueueImpl->m_HipQueue));
                    }
                    else
                    {
                        // Initiate the memory copy.
                        ALPAKA_HIP_RT_CHECK(
                            hipMemcpyPeerAsync(
                                dstNativePtr,
                                iDstDev,
                                srcNativePtr,
                                iSrcDev,
                                static_cast<std::size_t>(extentWidthBytes),
                                queue.m_spQueueImpl->m_HipQueue));
                    }
                }
            };
            //#############################################################################
            //! The HIP sync device queue 1D copy enqueue trait specialization.
            //#############################################################################
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueHipRtSync,
                mem::view::hip::detail::TaskCopyHip<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueHipRtSync &,
                    mem::view::hip::detail::TaskCopyHip<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    auto const & extentWidthBytes(task.m_extentWidthBytes);

                    auto const & dstNativePtr(task.m_dstMemNative);
                    auto const & srcNativePtr(task.m_srcMemNative);

                    if(iDstDev == iSrcDev)
                    {
                        auto const & hipMemCpyKind(task.m_hipMemCpyKind);

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_HIP_RT_CHECK(
                            hipMemcpy(
                                dstNativePtr,
                                srcNativePtr,
                                static_cast<std::size_t>(extentWidthBytes),
                                hipMemCpyKind));
                    }
                    else
                    {
                        // Initiate the memory copy.
                        ALPAKA_HIP_RT_CHECK(
                            hipMemcpyPeer(
                                dstNativePtr,
                                iDstDev,
                                srcNativePtr,
                                iSrcDev,
                                static_cast<std::size_t>(extentWidthBytes)));
                    }
                }
            };
            //#############################################################################
            //! The HIP async device queue 2D copy enqueue trait specialization.
            //#############################################################################
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueHipRtAsync,
                mem::view::hip::detail::TaskCopyHip<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueHipRtAsync & queue,
                    mem::view::hip::detail::TaskCopyHip<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    if(iDstDev == iSrcDev)
                    {
                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & extentHeight(task.m_extentHeight);

                        auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        auto const & srcPitchBytesX(task.m_srcpitchBytesX);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        auto const & hipMemCpyKind(task.m_hipMemCpyKind);

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_HIP_RT_CHECK(
                            hipMemcpy2DAsync(
                                dstNativePtr,
                                static_cast<std::size_t>(dstPitchBytesX),
                                srcNativePtr,
                                static_cast<std::size_t>(srcPitchBytesX),
                                static_cast<std::size_t>(extentWidthBytes),
                                static_cast<std::size_t>(extentHeight),
                                hipMemCpyKind,
                                queue.m_spQueueImpl->m_HipQueue));
                    }
                    else
                    {
                        // There is no hipMemcpy2DPeerAsync, therefore we use hipMemcpy3DPeerAsync.
                        // Create the struct describing the copy.
                        //~ hipMemcpy3DPeerParms const hipMemCpy3DPeerParms(
                            //~ mem::view::hip::detail::buildHipMemcpy3DPeerParms(
                                //~ task));
                        // Initiate the memory copy.
                        //~ ALPAKA_HIP_RT_CHECK(
                            //~ hipMemcpy3DPeerAsync(
                                //~ &hipMemCpy3DPeerParms,
                                //~ queue.m_spQueueImpl->m_HipQueue));
                    }
                }
            };
            //#############################################################################
            //! The HIP sync device queue 2D copy enqueue trait specialization.
            //#############################################################################
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueHipRtSync,
                mem::view::hip::detail::TaskCopyHip<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueHipRtSync &,
                    mem::view::hip::detail::TaskCopyHip<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    if(iDstDev == iSrcDev)
                    {
                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & extentHeight(task.m_extentHeight);

                        auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        auto const & srcPitchBytesX(task.m_srcpitchBytesX);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        auto const & hipMemCpyKind(task.m_hipMemCpyKind);

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_HIP_RT_CHECK(
                            hipMemcpy2D(
                                dstNativePtr,
                                static_cast<std::size_t>(dstPitchBytesX),
                                srcNativePtr,
                                static_cast<std::size_t>(srcPitchBytesX),
                                static_cast<std::size_t>(extentWidthBytes),
                                static_cast<std::size_t>(extentHeight),
                                hipMemCpyKind));
                    }
                    else
                    {
                        //~ // There is no hipMemcpy2DPeerAsync, therefore we use hipMemcpy3DPeerAsync.
                        //~ // Create the struct describing the copy.
                        //~ hipMemcpy3DPeerParms const hipMemCpy3DPeerParms(
                            //~ mem::view::hip::detail::buildHipMemcpy3DPeerParms(
                                //~ task));
                        //~ // Initiate the memory copy.
                        //~ ALPAKA_HIP_RT_CHECK(
                            //~ hipMemcpy3DPeer(
                                //~ &hipMemCpy3DPeerParms));
                    }
                }
            };
            //#############################################################################
            //! The HIP async device queue 3D copy enqueue trait specialization.
            //#############################################################################
            //~ template<
                //~ typename TExtent,
                //~ typename TViewSrc,
                //~ typename TViewDst>
            //~ struct Enqueue<
                //~ queue::QueueHipRtAsync,
                //~ mem::view::hip::detail::TaskCopyHip<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent>>
            //~ {
                //~ //-----------------------------------------------------------------------------
                //~ //
                //~ //-----------------------------------------------------------------------------
                //~ ALPAKA_FN_HOST static auto enqueue(
                    //~ queue::QueueHipRtAsync & queue,
                    //~ mem::view::hip::detail::TaskCopyHip<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent> const & task)
                //~ -> void
                //~ {
                    //~ ALPAKA_DEBUG_FULL_LOG_SCOPE;

//~ #if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //~ task.printDebug();
//~ #endif
                    //~ auto const & iDstDev(task.m_iDstDevice);
                    //~ auto const & iSrcDev(task.m_iSrcDevice);

                    //~ if(iDstDev == iSrcDev)
                    //~ {
                        //~ // Create the struct describing the copy.
                        //~ hipMemcpy3DParms const hipMemCpy3DParms(
                            //~ mem::view::hip::detail::buildHipMemcpy3DParms(
                                //~ task));
                        //~ // Set the current device.
                        //~ ALPAKA_HIP_RT_CHECK(
                            //~ hipSetDevice(
                                //~ iDstDev));
                        //~ // Initiate the memory copy.
                        //~ ALPAKA_HIP_RT_CHECK(
                            //~ hipMemcpy3DAsync(
                                //~ &hipMemCpy3DParms,
                                //~ queue.m_spQueueImpl->m_HipQueue));
                    //~ }
                    //~ else
                    //~ {
                        //~ // Create the struct describing the copy.
                        //~ hipMemcpy3DPeerParms const hipMemCpy3DPeerParms(
                            //~ mem::view::hip::detail::buildHipMemcpy3DPeerParms(
                                //~ task));
                        //~ // Initiate the memory copy.
                        //~ ALPAKA_HIP_RT_CHECK(
                            //~ hipMemcpy3DPeerAsync(
                                //~ &hipMemCpy3DPeerParms,
                                //~ queue.m_spQueueImpl->m_HipQueue));
                    //~ }
                //~ }
            //~ };
            //#############################################################################
            //! The HIP sync device queue 3D copy enqueue trait specialization.
            //#############################################################################
            //~ template<
                //~ typename TExtent,
                //~ typename TViewSrc,
                //~ typename TViewDst>
            //~ struct Enqueue<
                //~ queue::QueueHipRtSync,
                //~ mem::view::hip::detail::TaskCopyHip<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent>>
            //~ {
                //~ //-----------------------------------------------------------------------------
                //~ //
                //~ //-----------------------------------------------------------------------------
                //~ ALPAKA_FN_HOST static auto enqueue(
                    //~ queue::QueueHipRtSync &,
                    //~ mem::view::hip::detail::TaskCopyHip<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent> const & task)
                //~ -> void
                //~ {
                    //~ ALPAKA_DEBUG_FULL_LOG_SCOPE;

//~ #if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //~ task.printDebug();
//~ #endif
                    //~ auto const & iDstDev(task.m_iDstDevice);
                    //~ auto const & iSrcDev(task.m_iSrcDevice);

                    //~ if(iDstDev == iSrcDev)
                    //~ {
                        //~ // Create the struct describing the copy.
                        //~ hipMemcpy3DParms const hipMemCpy3DParms(
                            //~ mem::view::hip::detail::buildHipMemcpy3DParms(
                                //~ task));
                        //~ // Set the current device.
                        //~ ALPAKA_HIP_RT_CHECK(
                            //~ hipSetDevice(
                                //~ iDstDev));
                        //~ // Initiate the memory copy.
                        //~ ALPAKA_HIP_RT_CHECK(
                            //~ hipMemcpy3D(
                                //~ &hipMemCpy3DParms));
                    //~ }
                    //~ else
                    //~ {
                        //~ // Create the struct describing the copy.
                        //~ hipMemcpy3DPeerParms const hipMemCpy3DPeerParms(
                            //~ mem::view::hip::detail::buildHipMemcpy3DPeerParms(
                                //~ task));
                        //~ // Initiate the memory copy.
                        //~ ALPAKA_HIP_RT_CHECK(
                            //~ hipMemcpy3DPeer(
                                //~ &hipMemCpy3DPeerParms));
                    //~ }
                //~ }
            //~ };
        }
    }
}

#endif


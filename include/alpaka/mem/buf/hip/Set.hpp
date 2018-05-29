/**
* \file
* Copyright 2014-2016 Benjamin Worpitz, Rene Widera, Erik Zenker
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

#include <alpaka/queue/QueueHipRtSync.hpp>
#include <alpaka/queue/QueueHipRtAsync.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/queue/Traits.hpp>

#include <alpaka/core/Hip.hpp>

#include <cassert>

namespace alpaka
{
    namespace dev
    {
        class DevHipRt;
    }
}

namespace alpaka
{
    //-----------------------------------------------------------------------------
    // Trait specializations for Set.
    //-----------------------------------------------------------------------------
    namespace mem
    {
        namespace view
        {
            namespace hip
            {
                namespace detail
                {
                    //#############################################################################
                    //! The HIP memory set trait.
                    //#############################################################################
                    template<
                        typename TDim,
                        typename TView,
                        typename TExtent>
                    struct TaskSetHip
                    {
                        //-----------------------------------------------------------------------------
                        //!
                        //-----------------------------------------------------------------------------
                        TaskSetHip(
                            TView & buf,
                            std::uint8_t const & byte,
                            TExtent const & extent) :
                                m_buf(buf),
                                m_byte(byte),
                                m_extent(extent),
                                m_iDevice(dev::getDev(buf).m_iDevice)
                        {
                            static_assert(
                                dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                                "The destination buffer and the extent are required to have the same dimensionality!");
                        }

                        TView & m_buf;
                        std::uint8_t const m_byte;
                        TExtent const m_extent;
                        std::int32_t const m_iDevice;
                    };
                }
            }
            namespace traits
            {
                //#############################################################################
                //! The HIP device memory set trait specialization.
                //#############################################################################
                template<
                    typename TDim>
                struct CreateTaskSet<
                    TDim,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TView>
                    ALPAKA_FN_HOST static auto createTaskSet(
                        TView & buf,
                        std::uint8_t const & byte,
                        TExtent const & extent)
                    -> mem::view::hip::detail::TaskSetHip<
                        TDim,
                        TView,
                        TExtent>
                    {
                        return
                            mem::view::hip::detail::TaskSetHip<
                                TDim,
                                TView,
                                TExtent>(
                                    buf,
                                    byte,
                                    extent);
                    }
                };
            }
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP async device queue 1D set enqueue trait specialization.
            //#############################################################################
            template<
                typename TView,
                typename TExtent>
            struct Enqueue<
                queue::QueueHipRtAsync,
                mem::view::hip::detail::TaskSetHip<dim::DimInt<1u>, TView, TExtent>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueHipRtAsync & queue,
                    mem::view::hip::detail::TaskSetHip<dim::DimInt<1u>, TView, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        dim::Dim<TView>::value == 1u,
                        "The destination buffer is required to be 1-dimensional for this specialization!");
                    static_assert(
                        dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");

                    using Idx = idx::Idx<TExtent>;

                    auto & buf(task.m_buf);
                    auto const & byte(task.m_byte);
                    auto const & extent(task.m_extent);
                    auto const & iDevice(task.m_iDevice);

                    auto const extentWidth(extent::getWidth(extent));
                    auto const extentWidthBytes(extentWidth * static_cast<Idx>(sizeof(elem::Elem<TView>)));
#if !defined(NDEBUG)
                    auto const dstWidth(extent::getWidth(buf));
#endif
                    auto const dstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(buf)));
                    assert(extentWidth <= dstWidth);

                    // Set the current device.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            iDevice));
                    // Initiate the memory set.
                    ALPAKA_HIP_RT_CHECK(
                        hipMemsetAsync(
                            dstNativePtr,
                            static_cast<int>(byte),
                            static_cast<size_t>(extentWidthBytes),
                            queue.m_spQueueImpl->m_HipQueue));
                }
            };
            //#############################################################################
            //! The HIP sync device queue 1D set enqueue trait specialization.
            //#############################################################################
            template<
                typename TView,
                typename TExtent>
            struct Enqueue<
                queue::QueueHipRtSync,
                mem::view::hip::detail::TaskSetHip<dim::DimInt<1u>, TView, TExtent>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueHipRtSync &,
                    mem::view::hip::detail::TaskSetHip<dim::DimInt<1u>, TView, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        dim::Dim<TView>::value == 1u,
                        "The destination buffer is required to be 1-dimensional for this specialization!");
                    static_assert(
                        dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");

                    using Idx = idx::Idx<TExtent>;

                    auto & buf(task.m_buf);
                    auto const & byte(task.m_byte);
                    auto const & extent(task.m_extent);
                    auto const & iDevice(task.m_iDevice);

                    auto const extentWidth(extent::getWidth(extent));
                    auto const extentWidthBytes(extentWidth * static_cast<Idx>(sizeof(elem::Elem<TView>)));
#if !defined(NDEBUG)
                    auto const dstWidth(extent::getWidth(buf));
#endif
                    auto const dstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(buf)));
                    assert(extentWidth <= dstWidth);

                    // Set the current device.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            iDevice));
                    // Initiate the memory set.
                    ALPAKA_HIP_RT_CHECK(
                        hipMemset(
                            dstNativePtr,
                            static_cast<int>(byte),
                            static_cast<size_t>(extentWidthBytes)));
                }
            };
            //#############################################################################
            //! The HIP async device queue 2D set enqueue trait specialization.
            //#############################################################################
            //~ template<
                //~ typename TView,
                //~ typename TExtent>
            //~ struct Enqueue<
                //~ queue::QueueHipRtAsync,
                //~ mem::view::hip::detail::TaskSetHip<dim::DimInt<2u>, TView, TExtent>>
            //~ {
                //~ //-----------------------------------------------------------------------------
                //~ //
                //~ //-----------------------------------------------------------------------------
                //~ ALPAKA_FN_HOST static auto enqueue(
                    //~ queue::QueueHipRtAsync & queue,
                    //~ mem::view::hip::detail::TaskSetHip<dim::DimInt<2u>, TView, TExtent> const & task)
                //~ -> void
                //~ {
                    //~ ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    //~ static_assert(
                        //~ dim::Dim<TView>::value == 2u,
                        //~ "The destination buffer is required to be 2-dimensional for this specialization!");
                    //~ static_assert(
                        //~ dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                        //~ "The destination buffer and the extent are required to have the same dimensionality!");

                    //~ using Idx = size::Idx<TExtent>;

                    //~ auto & buf(task.m_buf);
                    //~ auto const & byte(task.m_byte);
                    //~ auto const & extent(task.m_extent);
                    //~ auto const & iDevice(task.m_iDevice);

                    //~ auto const extentWidth(extent::getWidth(extent));
                    //~ auto const extentWidthBytes(extentWidth * static_cast<Idx>(sizeof(elem::Elem<TView>)));
                    //~ auto const extentHeight(extent::getHeight(extent));
//~ #if !defined(NDEBUG)
                    //~ auto const dstWidth(extent::getWidth(buf));
                    //~ auto const dstHeight(extent::getHeight(buf));
//~ #endif
                    //~ auto const dstPitchBytesX(mem::view::getPitchBytes<dim::Dim<TView>::value - 1u>(buf));
                    //~ auto const dstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(buf)));
                    //~ assert(extentWidth <= dstWidth);
                    //~ assert(extentHeight <= dstHeight);

                    //~ // Set the current device.
                    //~ ALPAKA_HIP_RT_CHECK(
                        //~ hipSetDevice(
                            //~ iDevice));
                    //~ // Initiate the memory set.
                    //~ ALPAKA_HIP_RT_CHECK(
                        //~ hipMemset2DAsync(
                            //~ dstNativePtr,
                            //~ static_cast<size_t>(dstPitchBytesX),
                            //~ static_cast<int>(byte),
                            //~ static_cast<size_t>(extentWidthBytes),
                            //~ static_cast<size_t>(extentHeight),
                            //~ queue.m_spQueueHipRtAsyncImpl->m_HipQueue));
                //~ }
            //~ };
            //#############################################################################
            //! The HIP sync device queue 2D set enqueue trait specialization.
            //#############################################################################
            //~ template<
                //~ typename TView,
                //~ typename TExtent>
            //~ struct Enqueue<
                //~ queue::QueueHipRtSync,
                //~ mem::view::hip::detail::TaskSetHip<dim::DimInt<2u>, TView, TExtent>>
            //~ {
                //~ //-----------------------------------------------------------------------------
                //~ //
                //~ //-----------------------------------------------------------------------------
                //~ ALPAKA_FN_HOST static auto enqueue(
                    //~ queue::QueueHipRtSync &,
                    //~ mem::view::hip::detail::TaskSetHip<dim::DimInt<2u>, TView, TExtent> const & task)
                //~ -> void
                //~ {
                    //~ ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    //~ static_assert(
                        //~ dim::Dim<TView>::value == 2u,
                        //~ "The destination buffer is required to be 2-dimensional for this specialization!");
                    //~ static_assert(
                        //~ dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                        //~ "The destination buffer and the extent are required to have the same dimensionality!");

                    //~ using Idx = size::Idx<TExtent>;

                    //~ auto & buf(task.m_buf);
                    //~ auto const & byte(task.m_byte);
                    //~ auto const & extent(task.m_extent);
                    //~ auto const & iDevice(task.m_iDevice);

                    //~ auto const extentWidth(extent::getWidth(extent));
                    //~ auto const extentWidthBytes(extentWidth * static_cast<Idx>(sizeof(elem::Elem<TView>)));
                    //~ auto const extentHeight(extent::getHeight(extent));
//~ #if !defined(NDEBUG)
                    //~ auto const dstWidth(extent::getWidth(buf));
                    //~ auto const dstHeight(extent::getHeight(buf));
//~ #endif
                    //~ auto const dstPitchBytesX(mem::view::getPitchBytes<dim::Dim<TView>::value - 1u>(buf));
                    //~ auto const dstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(buf)));
                    //~ assert(extentWidth <= dstWidth);
                    //~ assert(extentHeight <= dstHeight);

                    //~ // Set the current device.
                    //~ ALPAKA_HIP_RT_CHECK(
                        //~ hipSetDevice(
                            //~ iDevice));
                    //~ // Initiate the memory set.
                    //~ ALPAKA_HIP_RT_CHECK(
                        //~ hipMemset2D(
                            //~ dstNativePtr,
                            //~ static_cast<size_t>(dstPitchBytesX),
                            //~ static_cast<int>(byte),
                            //~ static_cast<size_t>(extentWidthBytes),
                            //~ static_cast<size_t>(extentHeight)));
                //~ }
            //~ };
            //#############################################################################
            //! The HIP async device queue 3D set enqueue trait specialization.
            //#############################################################################
            //~ template<
                //~ typename TView,
                //~ typename TExtent>
            //~ struct Enqueue<
                //~ queue::QueueHipRtAsync,
                //~ mem::view::hip::detail::TaskSetHip<dim::DimInt<3u>, TView, TExtent>>
            //~ {
                //~ //-----------------------------------------------------------------------------
                //~ //
                //~ //-----------------------------------------------------------------------------
                //~ ALPAKA_FN_HOST static auto enqueue(
                    //~ queue::QueueHipRtAsync & queue,
                    //~ mem::view::hip::detail::TaskSetHip<dim::DimInt<3u>, TView, TExtent> const & task)
                //~ -> void
                //~ {
                    //~ ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    //~ static_assert(
                        //~ dim::Dim<TView>::value == 3u,
                        //~ "The destination buffer is required to be 3-dimensional for this specialization!");
                    //~ static_assert(
                        //~ dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                        //~ "The destination buffer and the extent are required to have the same dimensionality!");

                    //~ using Elem = alpaka::elem::Elem<TView>;
                    //~ using Idx = size::Idx<TExtent>;

                    //~ auto & buf(task.m_buf);
                    //~ auto const & byte(task.m_byte);
                    //~ auto const & extent(task.m_extent);
                    //~ auto const & iDevice(task.m_iDevice);

                    //~ auto const extentWidth(extent::getWidth(extent));
                    //~ auto const extentHeight(extent::getHeight(extent));
                    //~ auto const extentDepth(extent::getDepth(extent));
                    //~ auto const dstWidth(extent::getWidth(buf));
//~ #if !defined(NDEBUG)
                    //~ auto const dstHeight(extent::getHeight(buf));
                    //~ auto const dstDepth(extent::getDepth(buf));
//~ #endif
                    //~ auto const dstPitchBytesX(mem::view::getPitchBytes<dim::Dim<TView>::value - 1u>(buf));
                    //~ auto const dstPitchBytesY(mem::view::getPitchBytes<dim::Dim<TView>::value - (2u % dim::Dim<TView>::value)>(buf));
                    //~ auto const dstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(buf)));
                    //~ assert(extentWidth <= dstWidth);
                    //~ assert(extentHeight <= dstHeight);
                    //~ assert(extentDepth <= dstDepth);

                    //~ // Fill HIP parameter structures.
                    //~ hipPitchedPtr const hipPitchedPtrVal(
                        //~ make_hipPitchedPtr(
                            //~ dstNativePtr,
                            //~ static_cast<size_t>(dstPitchBytesX),
                            //~ static_cast<size_t>(dstWidth * static_cast<Idx>(sizeof(Elem))),
                            //~ static_cast<size_t>(dstPitchBytesY/dstPitchBytesX)));

                    //~ hipExtent const hipExtentVal(
                        //~ make_hipExtent(
                            //~ static_cast<size_t>(extentWidth * static_cast<Idx>(sizeof(Elem))),
                            //~ static_cast<size_t>(extentHeight),
                            //~ static_cast<size_t>(extentDepth)));

                    //~ // Set the current device.
                    //~ ALPAKA_HIP_RT_CHECK(
                        //~ hipSetDevice(
                            //~ iDevice));
                    //~ // Initiate the memory set.
                    //~ ALPAKA_HIP_RT_CHECK(
                        //~ hipMemset3DAsync(
                            //~ hipPitchedPtrVal,
                            //~ static_cast<int>(byte),
                            //~ hipExtentVal,
                            //~ queue.m_spQueueHipRtAsyncImpl->m_HipQueue));
                //~ }
            //~ };
            //#############################################################################
            //! The HIP sync device queue 3D set enqueue trait specialization.
            //#############################################################################
            //~ template<
                //~ typename TView,
                //~ typename TExtent>
            //~ struct Enqueue<
                //~ queue::QueueHipRtSync,
                //~ mem::view::hip::detail::TaskSetHip<dim::DimInt<3u>, TView, TExtent>>
            //~ {
                //~ //-----------------------------------------------------------------------------
                //~ //
                //~ //-----------------------------------------------------------------------------
                //~ ALPAKA_FN_HOST static auto enqueue(
                    //~ queue::QueueHipRtSync &,
                    //~ mem::view::hip::detail::TaskSetHip<dim::DimInt<3u>, TView, TExtent> const & task)
                //~ -> void
                //~ {
                    //~ ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    //~ static_assert(
                        //~ dim::Dim<TView>::value == 3u,
                        //~ "The destination buffer is required to be 3-dimensional for this specialization!");
                    //~ static_assert(
                        //~ dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                        //~ "The destination buffer and the extent are required to have the same dimensionality!");

                    //~ using Elem = alpaka::elem::Elem<TView>;
                    //~ using Idx = size::Idx<TExtent>;

                    //~ auto & buf(task.m_buf);
                    //~ auto const & byte(task.m_byte);
                    //~ auto const & extent(task.m_extent);
                    //~ auto const & iDevice(task.m_iDevice);

                    //~ auto const extentWidth(extent::getWidth(extent));
                    //~ auto const extentHeight(extent::getHeight(extent));
                    //~ auto const extentDepth(extent::getDepth(extent));
                    //~ auto const dstWidth(extent::getWidth(buf));
//~ #if !defined(NDEBUG)
                    //~ auto const dstHeight(extent::getHeight(buf));
                    //~ auto const dstDepth(extent::getDepth(buf));
//~ #endif
                    //~ auto const dstPitchBytesX(mem::view::getPitchBytes<dim::Dim<TView>::value - 1u>(buf));
                    //~ auto const dstPitchBytesY(mem::view::getPitchBytes<dim::Dim<TView>::value - (2u % dim::Dim<TView>::value)>(buf));
                    //~ auto const dstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(buf)));
                    //~ assert(extentWidth <= dstWidth);
                    //~ assert(extentHeight <= dstHeight);
                    //~ assert(extentDepth <= dstDepth);

                    //~ // Fill HIP parameter structures.
                    //~ hipPitchedPtr const hipPitchedPtrVal(
                        //~ make_hipPitchedPtr(
                            //~ dstNativePtr,
                            //~ static_cast<size_t>(dstPitchBytesX),
                            //~ static_cast<size_t>(dstWidth * static_cast<Idx>(sizeof(Elem))),
                            //~ static_cast<size_t>(dstPitchBytesY/dstPitchBytesX)));

                    //~ hipExtent const hipExtentVal(
                        //~ make_hipExtent(
                            //~ static_cast<size_t>(extentWidth * static_cast<Idx>(sizeof(Elem))),
                            //~ static_cast<size_t>(extentHeight),
                            //~ static_cast<size_t>(extentDepth)));

                    //~ // Set the current device.
                    //~ ALPAKA_HIP_RT_CHECK(
                        //~ hipSetDevice(
                            //~ iDevice));
                    //~ // Initiate the memory set.
                    //~ ALPAKA_HIP_RT_CHECK(
                        //~ hipMemset3D(
                            //~ hipPitchedPtrVal,
                            //~ static_cast<int>(byte),
                            //~ hipExtentVal));
                //~ }
            //~ };
        }
    }
}

#endif

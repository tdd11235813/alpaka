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

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_*, __HIPCC__

#include <alpaka/dev/DevHipRt.hpp>	// DevHipRt

#include <alpaka/dev/Traits.hpp>        // dev::GetDev, dev::DevType
#include <alpaka/event/Traits.hpp>      // event::EventType
#include <alpaka/queue/Traits.hpp>      // queue::traits::Enqueue, ...
#include <alpaka/wait/Traits.hpp>       // CurrentThreadWaitFor, WaiterWaitFor

#include <alpaka/core/Hip.hpp>

#include <stdexcept>                    // std::runtime_error
#include <memory>                       // std::shared_ptr
#include <functional>                   // std::bind

namespace alpaka
{
    namespace event
    {
        class EventHipRt;
    }
}

namespace alpaka
{
    namespace queue
    {
        namespace hip
        {
            namespace detail
            {
                //#############################################################################
                //! The HIP RT queue implementation.
                //#############################################################################
                class QueueHipRtSyncImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    QueueHipRtSyncImpl(
                        dev::DevHipRt const & dev) :
                            m_dev(dev),
                            m_HipQueue()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                m_dev.m_iDevice));
                        // - hipStreamDefault: Default stream creation flag.
                        // - hipStreamNonBlocking: Specifies that work running in the created stream may run concurrently with work in stream 0 (the NULL stream),
                        //   and that the created stream should perform no implicit synchronization with stream 0.
                        // Create the stream on the current device.
                        // NOTE: hipStreamNonBlocking is required to match the semantic implemented in the alpaka CPU stream.
                        // It would be too much work to implement implicit default stream synchronization on CPU.
                        ALPAKA_HIP_RT_CHECK(
                            hipStreamCreateWithFlags(
                                &m_HipQueue,
                                hipStreamNonBlocking));
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST QueueHipRtSyncImpl(QueueHipRtSyncImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST QueueHipRtSyncImpl(QueueHipRtSyncImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueHipRtSyncImpl const &) -> QueueHipRtSyncImpl & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueHipRtSyncImpl &&) -> QueueHipRtSyncImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ~QueueHipRtSyncImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before hipStreamDestroy required?
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                m_dev.m_iDevice));
                        // In case the device is still doing work in the stream when hipStreamDestroy() is called, the function will return immediately
                        // and the resources associated with stream will be released automatically once the device has completed all work in stream.
                        // -> No need to synchronize here.
                        ALPAKA_HIP_RT_CHECK(
                            hipStreamDestroy(
                                m_HipQueue));
                    }

                public:
                    dev::DevHipRt const m_dev;   //!< The device this stream is bound to.
                    hipStream_t m_HipQueue;
                };
            }
        }

        //#############################################################################
        //! The HIP RT queue.
        //#############################################################################
        class QueueHipRtSync final
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST QueueHipRtSync(
                dev::DevHipRt const & dev) :
                m_spQueueImpl(std::make_shared<hip::detail::QueueHipRtSyncImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST QueueHipRtSync(QueueHipRtSync const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST QueueHipRtSync(QueueHipRtSync &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            auto operator=(QueueHipRtSync const &) -> QueueHipRtSync & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            auto operator=(QueueHipRtSync &&) -> QueueHipRtSync & = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            auto operator==(QueueHipRtSync const & rhs) const
            -> bool
            {
                return (m_spQueueImpl->m_HipQueue == rhs.m_spQueueImpl->m_HipQueue);
            }
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            auto operator!=(QueueHipRtSync const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~QueueHipRtSync() = default;

        public:
            std::shared_ptr<hip::detail::QueueHipRtSyncImpl> m_spQueueImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT queue device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                queue::QueueHipRtSync>
            {
                using type = dev::DevHipRt;
            };
            //#############################################################################
            //! The HIP RT queue device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                queue::QueueHipRtSync>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueHipRtSync const & queue)
                -> dev::DevHipRt
                {
                    return queue.m_spQueueImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT queue event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                queue::QueueHipRtSync>
            {
                using type = event::EventHipRt;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT queue test trait specialization.
            //#############################################################################
            template<>
            struct Empty<
                queue::QueueHipRtSync>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueHipRtSync const & queue)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for queues on non current device.
                    hipError_t ret = hipSuccess;
                    ALPAKA_HIP_RT_CHECK_IGNORE(
                        ret = hipStreamQuery(
                            queue.m_spQueueImpl->m_HipQueue),
                        hipErrorNotReady);
                    return (ret == hipSuccess);
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueHipRtSync>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueHipRtSync const & queue)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for queues on non current device.
                    ALPAKA_HIP_RT_CHECK(hipStreamSynchronize(
                        queue.m_spQueueImpl->m_HipQueue));
                }
            };
        }
    }
}

#endif

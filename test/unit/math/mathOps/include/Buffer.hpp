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
#include <alpaka/test/acc/TestAccs.hpp>
#include <ostream>

//! Provides Alpaka-style 2 dimensional Buffer
//! @brief Encapsulates initialisation and communication with Device.
//! @tparam TAcc Used accelerator, not interchangeable
//! @tparam TData The Data-type, only restricted by the alpaka-interface.
//! @tparam Idx Typically std::size_t, only used for alpaka-interface.
//! @tparam Dim Typically and tested for: alpaka::dim::DimInt...
//! @tparam extentBuf Size of each individual buffer.
//! @tparam dimBuf Number of individual Buffer, default value is one.
template<
    typename TAcc,
    typename TData,
    typename Idx,
    typename Dim,
    size_t extentBuf, // how many elements pro dimension
    size_t dimBuf = 1 // number of sub-buffer (n-dimensional)
>
struct Buffer
{
    static constexpr size_t count = dimBuf;
    static constexpr size_t extent = extentBuf;
    static constexpr size_t size = dimBuf * extentBuf;

    // Defines used for alpaka-buffer.
    using DevAcc = alpaka::dev::Dev< TAcc >;
    using DevHost = alpaka::dev::DevCpu;

    using BufHost = alpaka::mem::buf::Buf<
        DevHost,
        TData,
        Dim,
        Idx
    >;
    using BufAcc = alpaka::mem::buf::Buf<
        DevAcc,
        TData,
        Dim,
        Idx
    >;

    // Alpaka style Buffer.
    // Example memory Layout with 3 buffer of size 2:
    // [1.1,1.2, 2.1,2.2, 3.1,3.2]
    BufHost hostBuffer;
    BufAcc devBuffer;

    // Native pointer to access buffer.
    TData * const pHostBuffer;
    TData * const pDevBuffer;

    // This constructor cant be used,
    // because BufHost and BufAcc need to be initialised.
    Buffer( ) = delete;

    // Constructor needs to initialize all Buffer.
    Buffer(
        const DevHost & devHost,
        const DevAcc & devAcc
    ) :
        hostBuffer(
            alpaka::mem::buf::alloc<
                TData,
                Idx
            >(
                devHost,
                dimBuf * extentBuf
            )
        ),
        devBuffer(
            alpaka::mem::buf::alloc<
                TData,
                Idx
            >(
                devAcc,
                dimBuf * extentBuf
            )
        ),
        pHostBuffer( alpaka::mem::view::getPtrNative( hostBuffer ) ),
        pDevBuffer( alpaka::mem::view::getPtrNative( devBuffer ) )
    {}


    // Copy Host -> Acc.
    template< typename Queue >
    auto copyToDevice( Queue queue ) -> void
    {
        alpaka::mem::view::copy(
            queue,
            devBuffer,
            hostBuffer,
            dimBuf * extentBuf
        );
    }

    // Copy Acc -> Host.
    template< typename Queue >
    auto copyFromDevice( Queue queue ) -> void
    {
        alpaka::mem::view::copy(
            queue,
            hostBuffer,
            devBuffer,
            dimBuf * extentBuf
        );
    }

    // Getter:
    // TAccIn is mainly an indicator with buffer should be used.
    template<
        typename TAccIn = std::nullptr_t, size_t offset = 0, typename std::enable_if<
            std::is_same<
                TAccIn,
                std::nullptr_t
            >::value,
            int
        >::type = 0
    >
    ALPAKA_FN_HOST
    auto getArg( size_t idx ) const -> TData
    {
        return pHostBuffer[offset * extent + idx];
    }

    template<
        typename TAccIn = std::nullptr_t, size_t offset = 0, typename std::enable_if<
            std::is_same<
                TAccIn,
                TAcc
            >::value,
            int
        >::type = 0
    >
    ALPAKA_FN_ACC
    auto getArg( size_t idx ) const -> TData
    {
        return pDevBuffer[offset * extent + idx];
    }
    // Setter:
    // There is no Function for kernel, because those need to be const.
    ALPAKA_FN_HOST
    auto setArg(
        TData arg,
        size_t idx,
        size_t offset = 0
    ) -> void
    {
        pHostBuffer[offset * extent + idx] = arg;
    }

    ALPAKA_FN_HOST
    friend std::ostream & operator<<(
        std::ostream & os,
        const Buffer & buffer
    )
    {
        os << "count: " << count
           << ", extent: " << extent
           << ", size: " << size
           << "\n";
        for( size_t i = 0; i < count; ++i )
        {
            os << "buffer at offset - " << i << "\n";
            for( size_t j = 0; j < extent; ++j)
            {
                os << "elem at idx = " << j << ": "
                   << buffer.pHostBuffer[i * extent + j] << "\n";
            }
        }
        return os;
    }
};

#pragma once

#include "Defines.hpp"
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <catch2/catch.hpp>
#include <ostream>


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

    using DevAcc = alpaka::dev::Dev< TAcc >;
    using DevHost = alpaka::dev::DevCpu;

    using BufHost = alpaka::mem::buf::Buf<DevHost, TData, Dim, Idx>;
    using BufAcc = alpaka::mem::buf::Buf<DevAcc, TData, Dim, Idx>;

    // Alpaka style Buffer
    BufHost hostBuffer;
    BufAcc devBuffer;

    // Native pointer to access buffer
    TData * const pHostBuffer;
    TData * const pDevBuffer;

    Buffer() = delete;

    Buffer(
        const DevHost & devHost,
        const DevAcc & devAcc ) :
        hostBuffer(
            alpaka::mem::buf::alloc<
                TData,
                Idx
            >
            (
                devHost,
                dimBuf * extentBuf
            )
        ),
        devBuffer(
            alpaka::mem::buf::alloc<
                TData,
                Idx
            >
            (
                devAcc,
                dimBuf * extentBuf
            )
        ),
        pHostBuffer( alpaka::mem::view::getPtrNative( hostBuffer ) ),
        pDevBuffer( alpaka::mem::view::getPtrNative( devBuffer ) )
    { }


    // Copy Host -> Acc.
    template< typename Queue>
    auto copyToDevice(Queue queue) -> void
    {
        alpaka::mem::view::copy(
            queue,
            devBuffer,
            hostBuffer,
            dimBuf * extentBuf
        );
    }

    // Copy Host -> Acc.
    template< typename Queue >
    auto copyFromDevice(Queue queue) -> void
    {
        alpaka::mem::view::copy(
            queue,
            hostBuffer,
            devBuffer,
            dimBuf * extentBuf
        );
    }
   // getter
   template<
       typename TAccIn = std::nullptr_t,
       size_t offset = 0,
       typename std::enable_if<
           std::is_same<
               TAccIn,std::nullptr_t>::value, int>::type = 0>
   ALPAKA_FN_HOST
   auto getArg(size_t idx) const -> TData
   {
       return pHostBuffer[offset * extent + idx];
   }
   template<
       typename TAccIn = std::nullptr_t,
       size_t offset = 0,
       typename std::enable_if<
           std::is_same<
              TAccIn,
              TAcc>::value, int>::type = 0>
   ALPAKA_FN_ACC
   auto getArg(size_t idx) const -> TData
   {
        return pDevBuffer[offset * extent + idx];
   }
    // setter
    ALPAKA_FN_HOST
    auto setArg(TData arg, size_t idx, size_t offset=0) -> void
    {
        pHostBuffer[offset * extent + idx] = arg;
    }
    ALPAKA_FN_HOST \
 friend std::ostream & operator<<(
        std::ostream & os,
        const Buffer & buffer
    )
    {
        os << "count: " << count
            << ", extent: " << extent
            << ", size: " << size << "\n";
        os << "printing hostBuffer:\n";
        for(size_t i = 0; i < size; ++i)
        {
            os << "elem at i = " << i << ": " << buffer.pHostBuffer[i] << "\n";
        }
        return os;
    }
};

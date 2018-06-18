/**
 * \file
 * Copyright 2014-2018 Erik Zenker, Benjamin Worpitz
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
 *
 */

#include <alpaka/alpaka.hpp>
#include <lifft/libLiFFT.hpp>
#include <complex>

#include <iostream>
#include <cstdint>
#include <cassert>

template<typename TQueue>
class LiFFT_Library;

template<>
class LiFFT_Library<alpaka::queue::QueueCpuSync> {
    LiFFT_Library() { return blubb; }
};

auto main()
-> int
{
    // Define the index domain
    //using Dim = alpaka::dim::DimInt<DIMENSION>;
    using Idx = std::size_t;

    // Define the accelerator
    using Acc = alpaka::acc::AccCpuSerial<Dim, Idx>;
    using DevQueue = alpaka::queue::QueueCpuSync;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;

    using Host = alpaka::acc::AccCpuSerial<Dim, Idx>;
    using HostQueue = alpaka::queue::QueueCpuSync;
    using DevHost = alpaka::dev::Dev<Host>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;

    // Select devices
    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // Create queues
    DevQueue devQueue(devAcc);
    HostQueue hostQueue(devHost);

    // 2D single-precision FFT, 32x32 points, inplace-complex
    using Type = std::float;
    using Data = std::complex<Type>; // @todo use lifft's types?
    constexpr Idx nElementsPerDim = 32;
    using Dim = alpaka::dim::DimInt<2>;
    using Vec = alpaka::vec::Vec<Dim, Idx>;
    const Vec extents(Vec::all(static_cast<Idx>(nElementsPerDim)));

    // Allocate host memory buffers
    //
    // The `alloc` method returns a reference counted buffer handle.
    // When the last such handle is destroyed, the memory is freed automatically.
    using BufHost = alpaka::mem::buf::Buf<DevHost, Data, Dim, Idx>;
    BufHost hostInput(alpaka::mem::buf::alloc<Data, Idx>(devHost, extents));
    BufHost hostOutput(alpaka::mem::buf::alloc<Data, Idx>(devHost, extents));

    // Allocate accelerator memory buffers
    //
    // The interface to allocate a buffer is the same on the host and on the device.
    using BufAcc = alpaka::mem::buf::Buf<DevAcc, Data, Dim, Idx>;
    BufAcc input(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extents));
    BufAcc output(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extents));


    // Init host buffer
    //
    // You can not access the inner
    // elements of a buffer directly, but
    // you can get the pointer to the memory
    // (getPtrNative).
    Data * const pHostInput = alpaka::mem::view::getPtrNative(hostInput);
    Data * const pHostOutput = alpaka::mem::view::getPtrNative(hostOutput);

    std::default_random_engine gen;
    std::uniform_real_distribution<Type> dis(0.0, 1.0);

    // This pointer can be used to directly write
    // some values into the buffer memory.
    // Mind, that only a host can write on host memory.
    // The same holds true for device memory.
    for(Idx i(0); i < extents.prod(); ++i)
    {
        pHostInput[i].real = dis(gen);
        pHostInput[i].imag = dis(gen);
        pHostOutput[i].real = 0.0;
        pHostOutput[i].imag = 0.0;
    }


    // Copy host to device Buffer
    //
    // A copy operation of one buffer into
    // another buffer is enqueued into a queue
    // like it is done for kernel execution.
    // As always within alpaka, you will get a compile
    // time error if the desired copy coperation
    // (e.g. between various accelerator devices) is
    // not currently supported.
    // In this example both host buffers are copied
    // into device buffers.
    alpaka::mem::view::copy(devQueue, input, hostInput, extents);

    // Test device Buffer
    //
    // This kernel tests if the copy operations
    // were successful. In the case something
    // went wrong an assert will fail.
    Data const * const pInput = alpaka::mem::view::getPtrNative(input);
    Data const * const pOutput = alpaka::mem::view::getPtrNative(output);


    // FFT
    using FFT_TYPE = LiFFT::FFT_2D_R2C<Type>;
    constexpr const bool isDevicePtr = true;
    LiFFT::types::Vec<testNumDims> lifftExtents( extents ); // @todo
    using FFTLibrary = typename FFT_Library<Queue>::type;
    auto inWrapped = FFT_TYPE::wrapInput( LiFFT::mem::wrapPtr<isDevicePtr>(pInput, lifftExtents) );
    auto outWrapped = FFT_TYPE::wrapOutput(LiFFT::mem::wrapPtr<isDevicePtr>(pOutput, lifftExtents));
    auto fft = LiFFT::makeFFT<FFTLibrary>(inWrapped, outWrapped);
    fft(inWrapped, outWrapped);

    // copy back the results
    alpaka::mem::view::copy(devQueue, hostOutput, output, extents);

    return EXIT_SUCCESS;
}

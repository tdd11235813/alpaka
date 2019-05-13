/**
 * \file
 * Copyright 2018 Benjamin Worpitz
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

#include "mysqrt.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <boost/math/special_functions/relative_difference.hpp>

#include <catch2/catch.hpp>

#include <iostream>
#include <typeinfo>

//#############################################################################
//! A vector addition kernel.
class SqrtKernel
{
public:
    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param A The first source vector.
    //! \param B The second source vector.
    //! \param C The destination vector.
    //! \param numElements The number of elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TElem,
        typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TElem const * const A,
        TElem const * const B,
        TElem * const C,
        TIdx const & numElements) const
    -> void
    {
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 1,
            "The VectorAddKernel expects 1-dimensional indices!");

        auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        auto const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        auto const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        if(threadFirstElemIdx < numElements)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            auto const threadLastElemIdx(threadFirstElemIdx+threadElemExtent);
            auto const threadLastElemIdxClipped((numElements > threadLastElemIdx) ? threadLastElemIdx : numElements);

            for(TIdx i(threadFirstElemIdx); i<threadLastElemIdxClipped; ++i)
            {
                C[i] = mysqrt(A[i]) + mysqrt(B[i]);
            }
        }
    }
};

struct TestTemplate
{
template< typename TAcc >
void operator()()
{
    using Idx = alpaka::idx::Idx<TAcc>;

    using Val = double;

    using DevAcc = alpaka::dev::Dev<TAcc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using QueueAcc = alpaka::test::queue::DefaultQueue<alpaka::dev::Dev<TAcc>>;
    using PltfHost = alpaka::pltf::PltfCpu;
    using DevHost = alpaka::dev::Dev<PltfHost>;

    Idx const numElements(32);

    // Create the kernel function object.
    SqrtKernel kernel;

    // Get the host device.
    DevHost const devHost(
        alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // Select a device to execute on.
    DevAcc const devAcc(
        alpaka::pltf::getDevByIdx<PltfAcc>(0));

    // Get a queue on this device.
    QueueAcc queueAcc(devAcc);

    // The data extent.
    alpaka::vec::Vec<alpaka::dim::DimInt<1u>, Idx> const extent(
        numElements);

    // Let alpaka calculate good block and grid sizes given our full problem extent.
    alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<1u>, Idx> const workDiv(
        alpaka::workdiv::getValidWorkDiv<TAcc>(
            devAcc,
            extent,
            static_cast<Idx>(3u),
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

    std::cout
        << "VectorAddKernelTester("
        << " numElements:" << numElements
        << ", accelerator: " << alpaka::acc::getAccName<TAcc>()
        << ", kernel: " << typeid(kernel).name()
        << ", workDiv: " << workDiv
        << ")" << std::endl;

    // Allocate host memory buffers.
    auto memBufHostA(alpaka::mem::buf::alloc<Val, Idx>(devHost, extent));
    auto memBufHostB(alpaka::mem::buf::alloc<Val, Idx>(devHost, extent));
    auto memBufHostC(alpaka::mem::buf::alloc<Val, Idx>(devHost, extent));

    // Initialize the host input vectors
    for (Idx i(0); i < numElements; ++i)
    {
        alpaka::mem::view::getPtrNative(memBufHostA)[i] = static_cast<Val>(rand()) / static_cast<Val>(RAND_MAX);
        alpaka::mem::view::getPtrNative(memBufHostB)[i] = static_cast<Val>(rand()) / static_cast<Val>(RAND_MAX);
    }

    // Allocate the buffers on the accelerator.
    auto memBufAccA(alpaka::mem::buf::alloc<Val, Idx>(devAcc, extent));
    auto memBufAccB(alpaka::mem::buf::alloc<Val, Idx>(devAcc, extent));
    auto memBufAccC(alpaka::mem::buf::alloc<Val, Idx>(devAcc, extent));

    // Copy Host -> Acc.
    alpaka::mem::view::copy(queueAcc, memBufAccA, memBufHostA, extent);
    alpaka::mem::view::copy(queueAcc, memBufAccB, memBufHostB, extent);

    // Create the executor task.
    auto const taskKernel(alpaka::kernel::createTaskKernel<TAcc>(
        workDiv,
        kernel,
        alpaka::mem::view::getPtrNative(memBufAccA),
        alpaka::mem::view::getPtrNative(memBufAccB),
        alpaka::mem::view::getPtrNative(memBufAccC),
        numElements));

    // Profile the kernel execution.
    std::cout << "Execution time: "
        << alpaka::test::integ::measureTaskRunTimeMs(
            queueAcc,
            taskKernel)
        << " ms"
        << std::endl;

    // Copy back the result.
    alpaka::mem::view::copy(queueAcc, memBufHostC, memBufAccC, extent);

    bool resultCorrect(true);
    auto const pHostData(alpaka::mem::view::getPtrNative(memBufHostC));
    for(Idx i(0u);
        i < numElements;
        ++i)
    {
        auto const & val(pHostData[i]);
        auto const correctResult(std::sqrt(alpaka::mem::view::getPtrNative(memBufHostA)[i]) + std::sqrt(alpaka::mem::view::getPtrNative(memBufHostB)[i]));

        if( boost::math::relative_difference(val, correctResult) > 0.0001 )
        {
            std::cout << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
            resultCorrect = false;
        }
    }

    REQUIRE(true == resultCorrect);
}
};

TEST_CASE( "separableCompilation", "[separableCompilation]")
{
    using TestAccs = alpaka::test::acc::EnabledAccs<
        alpaka::dim::DimInt<1u>,
        std::size_t>;

    alpaka::meta::forEachType< TestAccs >( TestTemplate() );
}

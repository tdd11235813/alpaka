/** Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/* list of all operators
 * atan2     | Y | R^2\{(0,0)}
 * fmod      | Y | R^2\{(x,0)|x in R}
 * max       | Y | R^2
 * min       | Y | R^2
 * remainder | Y | R^2\{(x,0)|x in R}
 */

#include "../../dataGen.hpp"
#include "../include/binaryOps.hpp"
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <catch2/catch.hpp>

class TestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TData,
        typename TFunctor
    >
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        size_t const & sizeBuf,
        TData const * const argsX,
        TData const * const argsY,
        TData * results,
        TFunctor const & opFunctor) const
    -> void
    {
        TData argX;
        TData argY;
        for( size_t i = 0; i < sizeBuf; ++i )
        {
            argX = argsX[i];
            argY = argsY[i];
            // If range == Range::UNRESTRICTED all args are used.
            switch( opFunctor.range )
            {
                case Range::POSITIVE_ONLY:
                    if( i >= sizeBuf / 2 - 1 )
                        return;
                    break;
                case Range::NOT_ZERO:
                    if( i == sizeBuf / 2 || i == sizeBuf / 2 - 1 )
                        continue;
                    break;
                case Range::X_NOT_ZERO:
                    if( i == sizeBuf / 2 || i == sizeBuf / 2 - 1 )
                        argX = 1;
                    break;
                case Range::Y_NOT_ZERO:
                    if( i == sizeBuf / 2 || i == sizeBuf / 2 - 1 )
                        argY = 1;
                default:
                    break;
            }
            results[i] = opFunctor(
                acc,
                argX,
                argY
            );
        }
    }
};

template <typename TAcc, typename TData>
struct OpFunctorTemplate
{
    template<
        typename T_OpFunctor
    >
    auto operator()(
        TData * const & argXArray,
        TData * const & argYArray,
        size_t const & sizeBuf
    ) -> void
    {
        using Dim = alpaka::dim::Dim< TAcc >;
        using Idx = alpaka::idx::Idx< TAcc >;
        using DevAcc = alpaka::dev::Dev< TAcc >;
        using PltfAcc = alpaka::pltf::Pltf< DevAcc >;
        using QueueAcc = alpaka::test::queue::DefaultQueue< DevAcc >;
        using PltfHost = alpaka::pltf::PltfCpu;
        size_t const elementsPerThread = 1u;
        size_t const sizeExtent = 1u;

        T_OpFunctor opFunctor;

        TestKernel kernel;

        auto const devHost(
            alpaka::pltf::getDevByIdx< PltfHost >( 0u )
        );

        // Select a device to execute on.
        auto const devAcc(
            alpaka::pltf::getDevByIdx< PltfAcc >( 0u )
        );

        // Get a queue on this device.
        QueueAcc queue( devAcc );

        alpaka::vec::Vec<
            Dim,
            Idx
        > const extent( sizeExtent );

        // Let alpaka calculate good block and grid sizes given our full problem extent.
        alpaka::workdiv::WorkDivMembers<
            Dim,
            Idx
        > const workDiv(
            alpaka::workdiv::getValidWorkDiv< TAcc >(
                devAcc,
                extent,
                elementsPerThread,
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted
            )
        );

        // Allocate host memory buffers.
        auto memBufHostArgsX(
            alpaka::mem::buf::alloc<
                TData,
                Idx
            >(
                devHost,
                sizeBuf
            )
        );
        auto memBufHostArgsY(
            alpaka::mem::buf::alloc<
                TData,
                Idx
            >(
                devHost,
                sizeBuf
            )
        );
        auto memBufHostRes(
            alpaka::mem::buf::alloc<
                TData,
                Idx
            >(
                devHost,
                sizeBuf
            )
        );
        TData
            * const pBufHostArgsX =
            alpaka::mem::view::getPtrNative( memBufHostArgsX );
        TData
            * const pBufHostArgsY =
            alpaka::mem::view::getPtrNative( memBufHostArgsY );

        TData
            * const pBufHostRes =
            alpaka::mem::view::getPtrNative( memBufHostRes );

        // Fill Res-buffer with "-1" for better debugging.
        for( size_t i = 0; i < sizeBuf; ++i )
        {
            pBufHostRes[i] = -1;
            pBufHostArgsX[i] = argXArray[i];
            pBufHostArgsY[i] = argYArray[i];
        }

        // Allocate the buffer on the accelerator.
        auto memBufAccArgsX(
            alpaka::mem::buf::alloc<
                TData,
                Idx
            >(
                devAcc,
                sizeBuf
            )
        );
        auto memBufAccArgsY(
            alpaka::mem::buf::alloc<
                TData,
                Idx
            >(
                devAcc,
                sizeBuf
            )
        );
        auto memBufAccRes(
            alpaka::mem::buf::alloc<
                TData,
                Idx
            >(
                devAcc,
                sizeBuf
            )
        );
        // Copy Host -> Acc.
        alpaka::mem::view::copy(
            queue,
            memBufAccArgsX,
            memBufHostArgsX,
            sizeBuf
        );
        alpaka::mem::view::copy(
            queue,
            memBufAccArgsY,
            memBufHostArgsY,
            sizeBuf
        );

        auto
            pMemBufAccArgsX = alpaka::mem::view::getPtrNative( memBufAccArgsX );
        auto
            pMemBufAccArgsY = alpaka::mem::view::getPtrNative( memBufAccArgsY );
        auto pMemBufAccRes = alpaka::mem::view::getPtrNative( memBufAccRes );
        // Create the kernel execution task.
        auto const taskKernel(
            alpaka::kernel::createTaskKernel< TAcc >(
                workDiv,
                kernel,
                sizeBuf,
                pMemBufAccArgsX,
                pMemBufAccArgsY,
                pMemBufAccRes,
                opFunctor
            )
        );

        // Enqueue the kernel execution task.
        alpaka::queue::enqueue(
            queue,
            taskKernel
        );

        alpaka::mem::view::copy(
            queue,
            memBufHostRes,
            memBufAccRes,
            sizeBuf
        );

        // Wait for the queue to finish the memory operation.
        alpaka::wait::wait( queue );

        TData res;
        TData stdRes;
        TData argX;
        TData argY;
        for( size_t i = 0; i < sizeBuf; ++i )
        {
            argX = pBufHostArgsX[i];
            argY = pBufHostArgsY[i];
            // If range == Range::UNRESTRICTED all args are used.
            switch( opFunctor.range )
            {
                case Range::POSITIVE_ONLY:
                    if( i >= sizeBuf / 2 - 1 )
                        return;
                    break;
                case Range::NOT_ZERO:
                    if( i == sizeBuf / 2 || i == sizeBuf / 2 - 1 )
                        continue;
                    break;
                case Range::X_NOT_ZERO:
                    if( i == sizeBuf / 2 || i == sizeBuf / 2 - 1 )
                        argX = 1; // Hardcoded, see TestKernel
                    break;
                case Range::Y_NOT_ZERO:
                    if( i == sizeBuf / 2 || i == sizeBuf / 2 - 1 )
                        argY = 1; // Hardcoded, see TestKernel
                    break;
                default:
                    break;
            }
            stdRes = opFunctor(
                argX,
                argY
            );
            res = pBufHostRes[i];
            INFO( "Op: " << opFunctor )
            INFO( "Index: " << i )
            INFO( "Exp: " << stdRes )
            INFO( "Got: " << res )
            INFO( "Type: " << typeid( res ).name( ) )
            REQUIRE( Approx( res ) == stdRes );
        }
    }
};

template <typename TData>
struct AccTemplate
{
public:
    template< typename TAcc >
    auto operator()(
        TData * const & argXArray,
        TData * const & argYArray,
        size_t const & sizeBuf)
        -> void
    {
        // Gets all unary operators defined in unaryOps.hpp.
        alpaka::meta::forEachType< Operators >(
            OpFunctorTemplate<
                TAcc,
                TData
            >( ),
            argXArray,
            argYArray,
            sizeBuf
        );
    }
};

template < typename TData >
void runTest(
    unsigned long seedX,
    unsigned long seedY )
{
    // Gets all acc-types.
    using TestAccs = alpaka::test::acc::EnabledAccs<
        alpaka::dim::DimInt< 1u >,
        std::size_t
    >;

    // All tests are using the same arguments.
    size_t const sizeBuf = 100u;
    size_t const randomRange = 100u;
    TData argXArray[sizeBuf];
    TData argYArray[sizeBuf];

    test::fillWithRndArgs< TData>(
        argXArray,
        sizeBuf,
        randomRange,
        seedX
    );
    test::fillWithRndArgs< TData>(
        argYArray,
        sizeBuf,
        randomRange,
        seedY
    );

    alpaka::meta::forEachType< TestAccs >(
        AccTemplate< TData >( ),
        argXArray,
        argYArray,
        sizeBuf
    );
}


TEST_CASE("binaryOps", "[math] [binaryOps]")
{
    /*
     * TEST_CASE        | generates seeds    | specifies datatype
     * runTest          | generates Buffer  | specifies AccTypes
     * AccTemplate      |       -           | specifies operators (from hpp)
     *  -> OpFunctorTemplate : device, kernel, buff-copy usw.
     *      -> All operators are tested independent, one per kernel execution.
     *      -> Does the TESTS (REQUIRES).
     * TestKernel       | uses alpaka::math:: on device
     */

    std::cout.precision(10);

    // Using only one seed for all test-cases.
    auto seedX = test::generateSeed();
    auto seedY = test::generateSeed();
    std::cout << "Using seed: " << seedX << "for x-args" << std::endl;
    std::cout << "Using seed: " << seedY << "for y-args" << std::endl;

    // Testing all unary operators on all devices.
    runTest< double >(seedX, seedY);
    runTest< float >(seedX, seedY);
}

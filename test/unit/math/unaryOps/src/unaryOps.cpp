/** Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/* list of all operators
 * operator  | in std | definition |  range | notes
 * abs       | Y | R
 * acos      | Y | [-1, 1]
 * asin      | Y | [-1, 1]
 * atan      | Y | R
 * cbrt      | Y | R | third root of arg
 * ceil      | Y | R
 * cos       | Y | R
 * erf       | Y | R | error function for arg
 * exp       | Y | R | e^arg
 * floor     | Y | R
 * log       | Y | N\{0}
 * round     | Y | R
 * rsqrt     | X | N\{0} | inverse square root
 * sin       | Y | R
 * sqrt      | Y | N
 * tan       | Y | [x | x \= pi/2 + k*pi, k in Z]
 * trunc     | Y | R | round towards zero
 */

#include "../include/dataGen.hpp"
#include "../include/unaryOps.hpp"
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
            TData const * const args,
            TData * results,
            TFunctor const & opFunctor) const
    -> void
    {
        // TODO: Find a better solution.
        if( opFunctor.range == Range::ONE_NEIGHBOURHOOD )
        {
            TData argsOneNeighbourhood[5]
                {
                    -1,
                    -0.5,
                    0,
                    0.5,
                    1
                };
            for( size_t i = 0; i < 5; ++i )
            {
                results[i] =
                    opFunctor(
                        acc,
                        argsOneNeighbourhood[i]
                    );
            }
        }
        else
        {
            for( size_t i = 0; i < sizeBuf; ++i )
            {
                // If range == Range::UNRESTRICTED all args are used.
                switch( opFunctor.range )
                {
                    case Range::POSITIVE_ONLY:
                        if( i >= sizeBuf / 2 - 1 )
                            return;
                        break;
                    case Range::POSITIVE_AND_ZERO:
                        if( i >= sizeBuf / 2 )
                            return;
                        break;
                    default:
                        break;
                }
                results[i] = opFunctor(
                    acc,
                    args[i]
                );
            }
        }
    }
};

template <typename TAcc, typename TData>
struct OpFunctorTemplate
{
    template <
            typename T_OpFunctor>
    auto operator()(TData * const & argArray, size_t const & sizeBuf) -> void
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
        auto memBufHostArgs(
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
            * const pBufHostArgs =
            alpaka::mem::view::getPtrNative( memBufHostArgs );

        TData
            * const pBufHostRes =
            alpaka::mem::view::getPtrNative( memBufHostRes );

        // Fill Res-buffer with "-1" for better debugging.
        for( size_t i = 0; i < sizeBuf; ++i )
        {
            pBufHostRes[i] = -1;
            pBufHostArgs[i] = argArray[i];
        }

        // Allocate the buffer on the accelerator.
        auto memBufAccArgs(
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
            memBufAccArgs,
            memBufHostArgs,
            sizeBuf
        );
        alpaka::mem::view::copy(
            queue,
            memBufAccRes,
            memBufHostRes,
            sizeBuf
        );
        auto pMemBufAccArgs = alpaka::mem::view::getPtrNative( memBufAccArgs );
        auto pMemBufAccRes = alpaka::mem::view::getPtrNative( memBufAccRes );
        // Create the kernel execution task.
        auto const taskKernel(
            alpaka::kernel::createTaskKernel< TAcc >(
                workDiv,
                kernel,
                sizeBuf,
                pMemBufAccArgs,
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
        if( opFunctor.range == Range::ONE_NEIGHBOURHOOD )
        {
            TData argsOneNeighbourhood[5] {
                -1,
                -0.5,
                0,
                0.5,
                1
            };
            for( size_t i = 0; i < 5; ++i )
            {
                stdRes = opFunctor( argsOneNeighbourhood[i] );
                res = pBufHostRes[i];
                INFO( "op: " << opFunctor )
                INFO( "index: " << i )
                REQUIRE( stdRes == Approx( res ) );
            }
        }
        else
        {
            for( size_t i = 0; i < sizeBuf; ++i )
            {
                switch( opFunctor.range )
                {
                    case Range::POSITIVE_ONLY:
                        if( i >= sizeBuf / 2 - 1 )
                            return;
                        break;
                    case Range::POSITIVE_AND_ZERO:
                        if( i >= sizeBuf / 2 )
                            return;
                        break;
                    default:
                        break;
                }
                stdRes = opFunctor( pBufHostArgs[i] );
                res = pBufHostRes[i];
                INFO( "op: " << opFunctor )
                INFO( "index: " << i )
                REQUIRE( stdRes == Approx( res ) );
            }
        }
    }
};

template <typename TData>
struct AccTemplate
{
public:
    template< typename TAcc >
    auto operator()(TData * const & argArray, size_t const & sizeBuf) -> void
    {
        // Gets all unary operators defined in unaryOps.hpp.
        alpaka::meta::forEachType< Operators >(
            OpFunctorTemplate<
                TAcc,
                TData
            >( ),
            argArray,
            sizeBuf
        );
    }
};

template < typename TData >
void runTest( unsigned long seed)
{
    // Gets all acc-types.
    using TestAccs = alpaka::test::acc::EnabledAccs<
        alpaka::dim::DimInt< 1u >,
        std::size_t
    >;

    // All tests are using the same arguments.
    size_t const sizeBuf = 100u;
    size_t const randomRange = 100u;
    TData argArray[sizeBuf];
    test::fillWithRndArgs< TData>(
        argArray,
        sizeBuf,
        randomRange,
        seed
    );

    alpaka::meta::forEachType< TestAccs >(
        AccTemplate< TData >( ),
        argArray,
        sizeBuf
    );
}


TEST_CASE("unaryOps", "[math/unaryOps]")
{
    /*
     * TEST_CASE        | generates seed    | specifies datatype
     * runTest          | generates Buffer  | specifies AccTypes
     * AccTemplate      |       -           | specifies operators (from hpp)
     *  -> OpFunctorTemplate : device, kernel, buff-copy usw.
     *      -> All operators are tested independent, one per kernel execution.
     *      -> Does the TESTS (REQUIRES).
     * TestKernel       | uses alpaka::math:: on device
     */

    // Using only one seed for all test-cases.
    auto seed = test::generateSeed();
    std::cout << "Using seed: " << seed << std::endl;

    // Testing all unary operators on all devices.
    runTest< double >(seed);
    runTest< float  >(seed);
}

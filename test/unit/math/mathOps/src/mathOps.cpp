/** Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "../include/Helper.hpp"
#include "../include/Defines.hpp"
#include "../include/Functor.hpp"
#include "../include/DataGen.hpp"

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <catch2/catch.hpp>

//! @tparam TAcc Accelerator.
//! @tparam THelper Of type Helper.hpp, includes argBuf and resBuf.
//! @tparam TFunctor Functor defined in Functor.hpp.
//! @param acc Accelerator given from alpaka.
//! @param helper Used to encapsulate argBuf/resBuf.
//! @param functor Accesible with operator().
class TestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename THelper,
        typename TFunctor>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        THelper const & helper,
        TFunctor const & functor) const
        -> void
    {
        for( size_t i = 0; i < THelper::ResBuf::size; ++i )
        {
            helper.resBuf.pDevBuffer[i] =
                compute(
                    functor,
                    helper.argsBuf,
                    i,
                    acc
                );
        }
    }
};

//#############################################################################
// For each Arity in Defines.hpp/Arity needs to be a compute function!
// The compute function calls the correct operator() functor-function
// and collects all needed arguments from the buffer.
ALPAKA_NO_HOST_ACC_WARNING
// SFINAE: Enabled if functors needs TWO args.
template<
    typename TFunctor,
    typename TBuffer,
    typename TAcc = std::nullptr_t, typename std::enable_if<
        TFunctor::arity == Arity::BINARY,
        std::nullptr_t
    >::type = nullptr
>
ALPAKA_FN_HOST_ACC
auto compute(
    TFunctor const & opFunctor,
    TBuffer const & buffer,
    size_t const & i,
    TAcc const & acc = nullptr
) -> decltype( opFunctor(
        acc,
        buffer.template getArg<
            TAcc,
            0
        >( i ),
        buffer.template getArg<
            TAcc,
            1
        >( i )
    ) )
{
    return opFunctor(
        acc,
        buffer.template getArg<
            TAcc,
            0
        >( i ),
        buffer.template getArg<
            TAcc,
            1
        >( i )
    );
}


ALPAKA_NO_HOST_ACC_WARNING
// SFINAE: Enabled if functors needs ONE arg.
template<
    typename TFunctor,
    typename TBuffer,
    typename TAcc = std::nullptr_t, typename std::enable_if<
        TFunctor::arity == Arity::UNARY,
        std::nullptr_t
    >::type = nullptr
>
ALPAKA_FN_HOST_ACC
auto compute(
    TFunctor const & opFunctor,
    TBuffer const & buffer,
    size_t const & i,
    TAcc const & acc = nullptr
) -> decltype( opFunctor(
    acc,
    buffer.template getArg<
        TAcc,
        0
    >( i )
) )
{
    return opFunctor(
        acc,
        buffer.template getArg<
            TAcc,
            0
        >( i )
    );
}

//#############################################################################
// The TestTemplate runs the main code and the tests (Dev,Helper,Functor).
//! @tparam TAcc One of the possible accelerator types, that need to be tested.
//! @tparam TData By now either double or float.
template <
    typename TAcc,
    typename TData>
struct TestTemplate
{
    template < typename TFunctor >
    auto operator() ( unsigned long seed ) -> void
    {
        // SETUP (defines and initialising)
        // DevAcc and DevHost are defined in Buffer.hpp too.
        using DevAcc = alpaka::dev::Dev< TAcc >;
        using DevHost = alpaka::dev::DevCpu;
        using PltfAcc = alpaka::pltf::Pltf< DevAcc >;
        using PltfHost = alpaka::pltf::Pltf< DevHost >;

        using Dim = alpaka::dim::DimInt< 1u >;
        using Idx = std::size_t;
        using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
        using QueueAcc = alpaka::test::queue::DefaultQueue< DevAcc >;

        using OpHelper = Helper<
            TAcc,
            TData,
            Dim,
            Idx,
            TFunctor::arity
        >;

        // Every functor is executed individual on one kernel.
        static constexpr size_t elementsPerThread = 1u;
        static constexpr size_t sizeExtent = 1u;

        DevAcc const devAcc{ alpaka::pltf::getDevByIdx< PltfAcc >( 0u )};
        DevHost const devHost{ alpaka::pltf::getDevByIdx< PltfHost >( 0u )};

        QueueAcc queue{ devAcc };

        TestKernel kernel;
        TFunctor functor;
        OpHelper helper
            {
                devHost,
                devAcc
            };

        WorkDiv const workDiv{
            alpaka::workdiv::getValidWorkDiv< TAcc >(
                devAcc,
                sizeExtent,
                elementsPerThread,
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted
            )};

        // SETUP COMPLETED.
        // Fill the buffer with random test-numbers and copy them to the device.
        test::fillWithRndArgs<TData>
            ( helper.argsBuf, functor, seed );
        // Copy both buffer to the device
        helper.copyToDevice(queue);

        auto const taskKernel(
            alpaka::kernel::createTaskKernel< TAcc >(
                workDiv,
                kernel,
                helper,
                functor
            )
        );
        // Enqueue the kernel execution task.
        alpaka::queue::enqueue(
            queue,
            taskKernel
        );
        // Copy back the results (encapsulated in the buffer class).
        helper.resBuf.copyFromDevice( queue );

        alpaka::wait::wait( queue );

        std::cout.precision(32);

        INFO("Operator: " << functor)
        INFO("Type: " << typeid( TData ).name() ) // Compiler specific.
        for( size_t i = 0; i < OpHelper::ResBuf::size; ++i )
        {
            INFO("Idx i: " << i)
            TData std_res = compute(
                functor,
                helper.argsBuf,
                i
            );
            REQUIRE(
                helper.resBuf.getArg(i)
                == Approx(
                    std_res
                )
            );
        }
    }
};

template< typename TData >
struct ForEachFunctor
{
    template< typename TAcc >
    auto operator()( unsigned long seed ) -> void
    {
        alpaka::meta::forEachType < UnaryFunctors >(
            TestTemplate<
                TAcc,
                TData
            >( ),
            seed
        );

        alpaka::meta::forEachType< BinaryFunctors >(
            TestTemplate<
                TAcc,
                TData
            >( ),
            seed
        );
    }
};

TEST_CASE("mathOps", "[math] [operator]")
{
    /*
     * All alpaka::math:: functions are tested with an array of arguments.
     * The results are saved in a different array and are compared
     * to the result of a std:: implementation.
     * The default result is nan and should fail a test.
     *
     * BE AWARE that:
     * - ALPAKA_CUDA_FAST_MATH should be disabled
     * - not all casts between float and double can be detected.
     * - no explicit edge cases are tested, rather than 0, maximum and minimum
     *   - but it is easy to add a new Range:: enum-type with custom edge cases
     *  - some tests may fail if ALPAKA_CUDA_FAST_MATH is turned on
     * - nan typically fails every test, but could be normal defined behaviour
     * - inf/-inf typically dont fail a test
     * - for easy debugging the << operator is overloaded for Buffer objects
     * - arguments are generated between 0 and 1000
     *     and the default argument-buffer-extent is 1000
     * The arguments are generated in DataGen.hpp and can easily be modified.
     * The arguments depend on the Range:: enum-type specified for each functor.
     * ----------------------------------------------------------------------
     * TEST_CASE        | generates seeds    | specifies datatype & acc-list
     * ForEachFunctor   |       -            | specifies functors
     *                                         (from Functor.hpp)
     * TestTemplate     | helper, functor, device, host, queue, kernel, usw.
     * - main execution:
     * - each functor has one helper
     *     - each helper has two buffer
     *         - each buffer encapsulated the host/device functionality.
     *         - provides templated getter and setter
     * - uses the same code for all functors regardless the arity
     * - all operators are tested independent, one per kernel
     * - tests the results (calls compute-function + REQUIRES)
     *
     * TestKernel
     * - uses alpaka::math:: on device
     * - calls the compute-function with acc-object
     *
     * EXTENSIBILITY:
     * - Add new operators in Functor.hpp and add them to the ...Functors tuple.
     * - Add a new Range:: enum-type in Defines.hpp
     *     and specify a fill-method in DataGen.hpp
     * - Add a new Arity:: enum-type in Defines.hpp,
     *     add a matching compute-function in mathOps.cpp,
     *     add a new ...Functors tuple.
     *     call alpaka::meta::forEachType with the tuple in ForEachFunctor
     */

    using TestAccs = alpaka::test::acc::EnabledAccs<
        alpaka::dim::DimInt< 1u >,
        std::size_t
    >;
    const auto seed = test::generateSeed( );
    std::cout << "using seed: " << seed << "\n\n";
    alpaka::meta::forEachType< TestAccs >(
        ForEachFunctor< double >( ),
        seed
    );
    alpaka::meta::forEachType< TestAccs >(
        ForEachFunctor< float >( ),
        seed
    );
}

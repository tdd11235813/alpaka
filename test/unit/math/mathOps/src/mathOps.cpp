#include "../include/Helper.hpp"
#include "../include/Defines.hpp"
#include "../include/Functor.hpp"

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <catch2/catch.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

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
        for( size_t i = 0; i < helper.argsBuf.extent; ++i )
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
ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TFunctor,
    typename TBuffer,
    typename TAcc = std::nullptr_t,
    typename std::enable_if<
        TFunctor::arity == Arity::BINARY,
        std::nullptr_t >::type = nullptr>
ALPAKA_FN_HOST_ACC
auto compute(
    TFunctor const & opFunctor,
    TBuffer const & buffer,
    size_t const & i,
    TAcc const & acc = nullptr)
    -> double
{
   return
    opFunctor(
        acc,
        buffer.template getArg<TAcc,0>(i),
        buffer.template getArg<TAcc,1>(i)
        );
}

ALPAKA_NO_HOST_ACC_WARNING
template<
    typename TFunctor,
    typename TBuffer,
    typename TAcc = std::nullptr_t,
    typename std::enable_if<
        TFunctor::arity == Arity::UNARY,
        std::nullptr_t >::type = nullptr>
ALPAKA_FN_HOST_ACC
auto compute(
    TFunctor const & opFunctor,
    TBuffer const & buffer,
    size_t const & i,
    TAcc const & acc = nullptr)
    -> double
{
   return
    opFunctor(
        acc,
        buffer.template getArg<TAcc,0>(i)
        );
}

template < typename TAcc >
struct TestTemplate
{
    template < typename TFunctor >
    auto operator() ( unsigned long seed ) -> void
    {
        using TData = double;
        using DevAcc = alpaka::dev::Dev< TAcc >;
        using DevHost = alpaka::dev::DevCpu;
        using PltfAcc = alpaka::pltf::Pltf< DevAcc >;
        using PltfHost = alpaka::pltf::Pltf< DevHost >;

        using Dim = alpaka::dim::DimInt< 1u >;
        using Idx = std::size_t;
        using DevAcc = alpaka::dev::Dev< TAcc >;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
        using DevHost = alpaka::dev::DevCpu;
        using PltfHost = alpaka::pltf::Pltf<DevHost>;
        using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
        using QueueAcc = alpaka::test::queue::DefaultQueue< DevAcc >;

        static constexpr size_t elementsPerThread = 1u;
        static constexpr size_t sizeExtent = 1u;

        DevAcc const devAcc{ alpaka::pltf::getDevByIdx< PltfAcc >( 0u )};
        DevHost const devHost{ alpaka::pltf::getDevByIdx< PltfHost >( 0u )};

        QueueAcc queue( devAcc );

        TestKernel kernel;
        TFunctor functor;
        Helper<
            TAcc,
            TData,
            TFunctor::arity
        >
            helper
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
        test::fillWithRndArgs<TData>
            ( helper.argsBuf, functor, seed );
        helper.copyToDevice(queue);
        /*for( size_t i = 0; i < helper.argsBuf.size; ++i )
        {
            std::cout
                << "arg at: "
                << i
                << " = " <<
                helper.argsBuf.getArg(i)
                << "\n";
        }*/
        std::cout << "\n";
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

        helper.resBuf.copyFromDevice(queue);

        alpaka::wait::wait( queue );

        /*
        for( size_t i = 0; i < helper.resBuf.size; ++i )
        {
            std::cout
                << "res at: "
                << i
                << " = " <<
                helper.resBuf.getArg(i)
                << "\n";
        }
        */
        for( size_t i = 0; i < helper.resBuf.size; ++i )
        {
            REQUIRE(
                helper.resBuf.getArg(i)
                == Approx(
                    compute(
                        functor,
                        helper.argsBuf,
                        i
                    )
                )
            );
        }
    }
};

struct AccTemplate
{
    template< typename TAcc >
    auto operator() ( unsigned long seed ) -> void
    {
        alpaka::meta::forEachType < UnaryFunctors >(TestTemplate<TAcc>(), seed);
        alpaka::meta::forEachType < BinaryFunctors >(TestTemplate<TAcc>(), seed);
    }
};

TEST_CASE("mathOps", "[math] [operator]")
{
    using TestAccs = alpaka::test::acc::EnabledAccs<
        alpaka::dim::DimInt< 1u >,
        std::size_t
    >;
    auto seed = test::generateSeed();
    std::cout << "using seed: " << seed << "\n\n";

    alpaka::meta::forEachType< TestAccs >(AccTemplate(), seed);
}

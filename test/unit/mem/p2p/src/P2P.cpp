/**
 * \file
 * Copyright 2015-2017 Benjamin Worpitz
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

// \Hack: Boost.MPL defines BOOST_MPL_CFG_GPU_ENABLED to __host__ __device__ if nvcc is used.
// BOOST_AUTO_TEST_CASE_TEMPLATE and its internals are not GPU enabled but is using boost::mpl::for_each internally.
// For each template parameter this leads to:
// /home/travis/build/boost/boost/mpl/for_each.hpp(78): warning: calling a __host__ function from a __host__ __device__ function is not allowed
// because boost::mpl::for_each has the BOOST_MPL_CFG_GPU_ENABLED attribute but the test internals are pure host methods.
// Because we do not use MPL within GPU code here, we can disable the MPL GPU support.
#define BOOST_MPL_CFG_GPU_ENABLED

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>

#include <boost/predef.h>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

#include <type_traits>
#include <numeric>

BOOST_AUTO_TEST_SUITE(memP2P)

//#############################################################################
//! 1D: sizeof(TIdx) * (5)
//! 2D: sizeof(TIdx) * (5, 4)
//! 3D: sizeof(TIdx) * (5, 4, 3)
//! 4D: sizeof(TIdx) * (5, 4, 3, 2)
template<
    std::size_t Tidx>
struct CreateExtentBufVal
{
    //-----------------------------------------------------------------------------
    template<
        typename TIdx>
    static auto create(
        TIdx)
    -> TIdx
    {
        return sizeof(TIdx) * (5u - Tidx);
    }
};

//-----------------------------------------------------------------------------
template<
    typename TAcc>
static auto testP2P(
    alpaka::vec::Vec<alpaka::dim::Dim<TAcc>, alpaka::idx::Idx<TAcc>> const & extent)
-> void
{
    using Dev = alpaka::dev::Dev<TAcc>;
    using Pltf = alpaka::pltf::Pltf<Dev>;
    using Queue = alpaka::test::queue::DefaultQueue<Dev>;

    using Elem = std::uint32_t;
    using Idx = alpaka::idx::Idx<TAcc>;

    if(alpaka::pltf::getDevCount<Pltf>()<2) {
      BOOST_TEST_MESSAGE( "No two devices found to test peer-to-peer copy." );
      BOOST_CHECK(true);
      return;
    }

    Dev const dev0(alpaka::pltf::getDevByIdx<Pltf>(0u));
    Dev const dev1(alpaka::pltf::getDevByIdx<Pltf>(1u));
    Queue queue0(dev0);
    Queue queue1(dev1);

    //-----------------------------------------------------------------------------
    auto buf0(alpaka::mem::buf::alloc<Elem, Idx>(dev0, extent));
    auto buf1(alpaka::mem::buf::alloc<Elem, Idx>(dev1, extent));

    //-----------------------------------------------------------------------------
    std::uint8_t const byte(static_cast<uint8_t>(42u));
    alpaka::mem::view::set(queue0, buf0, byte, extent);

    //-----------------------------------------------------------------------------
    alpaka::mem::view::copy(queue0, buf1, buf0, extent);
    alpaka::wait::wait(queue0);
    alpaka::test::mem::view::verifyBytesSet<TAcc>(queue1, buf1, byte);
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    memP2PTest,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    auto const extent(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, CreateExtentBufVal>(Idx()));

    testP2P<TAcc>( extent );
}

BOOST_AUTO_TEST_SUITE_END()

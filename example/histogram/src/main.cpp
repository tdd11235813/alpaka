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

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <typeinfo>

// TODO: more general/acc-specific intrinsics/functors
// ! if no accelerator, then still double buffers => yeah, but u can implement own wrappers on-top (alpaka ~ vulkan)
// ? Is this same values for workdiv as for Acc declaration?
// TODO: - shared memory
// TODO: - specific WorkDiv for CUDA for GridStriding Loop (32*SM,<?>)

typedef unsigned uint_t;
typedef unsigned long long uintll_t;
typedef unsigned __int128 uint128_t;


//-----------------------------------------------------------------------------
//! \return The run time of the given kernel.
template<
    typename TQueue,
    typename TExec>
auto measureKernelRunTimeMs(
    TQueue & queue,
    TExec && exec)
    -> std::chrono::milliseconds::rep
{
    // Wait for the queue to finish all tasks enqueued prior to the kernel.
    alpaka::wait::wait(queue);

    // Take the time prior to the execution.
    auto const tpStart(std::chrono::high_resolution_clock::now());

    // Execute the kernel functor.
    alpaka::queue::enqueue(queue, std::forward<TExec>(exec));

    // Wait for the queue to finish the kernel execution to measure its run time.
    alpaka::wait::wait(queue);

    // Take the time after the execution.
    auto const tpEnd(std::chrono::high_resolution_clock::now());

    auto const durElapsed(tpEnd - tpStart);

    // Return the duration.
    return std::chrono::duration_cast<std::chrono::milliseconds>(durElapsed).count();
}


uint_t bitcount(uint_t n)
{
  n = ((0xaaaaaaaa & n) >> 1) + (0x55555555 & n);
  n = ((0xcccccccc & n) >> 2) + (0x33333333 & n);
  n = ((0xf0f0f0f0 & n) >> 4) + (0x0f0f0f0f & n);
  n = ((0xff00ff00 & n) >> 8) + (0x00ff00ff & n);
  n = ((0xffff0000 & n) >> 16) + (0x0000ffff & n);
  return n;
}

template<typename TAcc>
struct Bitcount {
    auto operator()(uint_t v) -> uint_t
    {
        return bitcount(v);
    }
};

template<>
struct Bitcount<alpaka::acc::AccCpuSerial<alpaka::dim::DimInt<1u>, size_t> >{

    auto operator()(uint_t v) -> uint_t
    {
        return __builtin_popcount(v);
    }
};

template<>
struct Bitcount<alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<1u>, size_t> >{
// check host compiler path
    __device__
    auto operator()(uint_t v) -> uint_t
    {
        return __popc(v);
    }
};

//#############################################################################
//!
template<uint_t TBins>
class HistogramKernelGridStriding
{
public:
    static constexpr uint_t Bins = TBins;
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
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        uintll_t * const hist,
        uint_t n,
        uint_t A,
        uint_t offset,
        uint_t end,
        uint_t Aend) const
    -> void
    {
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 1,
            "The VectorAddKernel expects 1-dimensional indices!");

        uint_t hist_local[TBins] = { 0 };
        uint_t v, w;
        uint_t num_threads = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u];

        for(uint_t i = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
                   i < end;
                   i += num_threads)
        {
            w = A*i;
            for(v=w+A; v<Aend; v+=A)
            {
                ++hist_local[ Bitcount<TAcc>()( w^v ) ];
            }

            for(uint_t c=1; c<TBins; ++c) {
                alpaka::atomic::atomicOp<alpaka::atomic::op::Add>
                    (acc, hist+c, static_cast<uintll_t>(hist_local[c]));
            }
        }
    }
};

//#############################################################################
//!
template< uint_t TBins >
class HistogramKernelMonolithic
{
public:
    static constexpr uint_t Bins = TBins;
    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \param acc The accelerator to be executed on.
    ALPAKA_NO_HOST_ACC_WARNING
    template< typename TAcc >
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        uintll_t * const hist,
        uint_t n,
        uint_t A,
        uint_t offset,
        uint_t end,
        uint_t Aend) const
    -> void
    {
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 1,
            "The VectorAddKernel expects 1-dimensional indices!");

        uint_t hist_local[TBins] = { 0 };
        uint_t v, w;
        uint_t i = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
        if(i < end)
        {
            w = A*i;
            for(v=w+A; v<Aend; v+=A)
            {
                ++hist_local[ Bitcount<TAcc>()( w^v ) ];
            }

            for(uint_t c=1; c<TBins; ++c) {
                alpaka::atomic::atomicOp<alpaka::atomic::op::Add>
                    (acc, hist+c, static_cast<uintll_t>(hist_local[c]));
            }
        }
    }
};

/**
 *
 * \param k Number of bits for the patterns (generate 2^k patterns and
 * process all pairs of them)
 * \param A Factor for each pattern (A*v, v=0,..,2^k-1)
 * \tparam TLocalInt 32-bit or 64-bit unsigned integer (affects performance)
 * \tparam TBins Number of histogram bins/elements
 */
template<typename Acc, typename QueueAcc>
struct ComputeHistogram {
public:

    template<typename TKernel>
    auto operator()(uint_t k, uint_t A) -> void {
        static constexpr uint_t Bins = TKernel::Bins;

        using Size = typename alpaka::idx::traits::IdxType<Acc>::type;
        using Vec1D = alpaka::vec::Vec<alpaka::dim::DimInt<1u>, Size>;
        using DevAcc = alpaka::dev::Dev<Acc>;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
        using PltfHost = alpaka::pltf::PltfCpu;

        // Get the host device.
        auto const devHost(
            alpaka::pltf::getDevByIdx<PltfHost>(0u));

        // Select a device to execute on.
        auto const devAcc(
            alpaka::pltf::getDevByIdx<PltfAcc>(0));

        // Get a queue on this device.
        QueueAcc queue(devAcc);
        // instantiate kernel
        TKernel kernel;


        if( ( A & (A-1) ) == 0 ) // A is power of two
            A=1; // histogram yields same result as with A==1
        uint_t n = k+floor(log(A)/log(2.0))+1; // max. bins required
        assert(n<=Bins);
        Vec1D const extent (static_cast<Size>(1u<<k));
//    Vec1D const extent_bins (Size(Bins)); // does not work
        Vec1D const extent_bins (static_cast<Size>(Bins));


        // Let alpaka calculate good block and grid sizes given our full problem extent.
        alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<1u>, Size> const workDiv(
            alpaka::workdiv::getValidWorkDiv<Acc>(
                devAcc,
                extent,
                static_cast<Size>(1u),
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

        std::cout
            << "\n" << typeid(kernel).name() <<"("
            << "n: " << n
            << ", A: " << A
            << ", extent: " << extent
            << ", bins: " << n << " [" << Bins << "]"
            << ", accelerator: " << alpaka::acc::getAccName<Acc>()
            << ", device: " << alpaka::dev::getName(devAcc)
            << ", workDiv: " << workDiv
            << ")" << std::endl;

        // Allocate host memory buffers.
        auto memBufHostHist(alpaka::mem::buf::alloc<uintll_t, Size>(devHost, extent_bins));

        // for (Size i(0); i < Bins; ++i)
        // {
        //     alpaka::mem::view::getPtrNative(memBufHostHist)[i] = 0; // memset?
        // }
        //alpaka::mem::view::set(memBufHostHist, std::uint8_t(0), Bins);
        alpaka::mem::view::set(queue, memBufHostHist, 0u, extent_bins);
        // Allocate the buffers on the accelerator.
        auto memBufAccHist(alpaka::mem::buf::alloc<uintll_t, Size>(devAcc, extent_bins));
        // Copy Host -> Acc.
        alpaka::mem::view::copy(queue, memBufAccHist, memBufHostHist, extent_bins);

        // Create the executor task.
        auto const exec(alpaka::kernel::createTaskExec<Acc>(
                            workDiv,
                            kernel,
                            alpaka::mem::view::getPtrNative(memBufAccHist), // devPtr hist[]
                            n,  //
                            A,  //
                            0u, // offset
                            static_cast<uint_t>(extent[0u]),   // end
                            static_cast<uint_t>(A<<k)          // Aend
                            ));

        // Profile the kernel execution.
        //alpaka::queue::enqueue(queue, exec);
        auto elapsed = measureKernelRunTimeMs(queue, exec);

        // Copy back the result.
        alpaka::mem::view::copy(queue, memBufHostHist, memBufAccHist, extent_bins);

        auto const pHostData(alpaka::mem::view::getPtrNative(memBufHostHist));
        std::cout << "Result: ";
        for(uint_t i = 0u; i < n; ++i)
        {
            auto const & val(pHostData[i]);
            if(val)
                std::cout << "(" << i << ", " << 2*val << ")";
        }
        std::cout << "\nElapsed Time: " << elapsed << " ms\n";
    }
};

auto main()
-> int
{
    using CPU_Acc = alpaka::acc::AccCpuSerial<alpaka::dim::DimInt<1u>, std::size_t>;
    using CPU_QueueAcc = alpaka::queue::QueueCpuSync;
    using CUDA_Acc = alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<1u>, std::size_t>;
    using CUDA_QueueAcc = alpaka::queue::QueueCudaRtSync;

    using ComputeHistogramCPU  = ComputeHistogram<CPU_Acc, CPU_QueueAcc>;
    using ComputeHistogramCUDA = ComputeHistogram<CUDA_Acc, CUDA_QueueAcc>;

    using Kernels = std::tuple<HistogramKernelGridStriding<32u>,
                               HistogramKernelMonolithic<32u>>;

    alpaka::meta::forEachType<Kernels>(ComputeHistogramCPU(), 16u, 61u);
    alpaka::meta::forEachType<Kernels>(ComputeHistogramCUDA(), 16u, 61u);
    return EXIT_SUCCESS;
}

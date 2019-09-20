#pragma once

#include "Buffer.hpp"
#include "Defines.hpp"
#include "DataGen.hpp"
#include <iostream>
#include <alpaka/pltf/PltfCpu.hpp>
#include <alpaka/test/queue/Queue.hpp>

template<
    typename TAcc,
    typename TData,
    Arity arity>
struct Helper
{
    static constexpr size_t extentBuf = 5;
    using Dim = alpaka::dim::DimInt< 1u >;
    using Idx = std::size_t;

        using ArgsBuf = Buffer<
            TAcc,
            TData,
            Idx,
            Dim,
            extentBuf,
            static_cast<size_t>(arity)
        >;
        using ResBuf = Buffer<
            TAcc,
            TData,
            Idx,
            Dim,
            extentBuf
        >;

    ArgsBuf argsBuf;
    ResBuf resBuf;
    template<
        typename DevHost,
        typename DevAcc>
    Helper( DevHost const & devHost, DevAcc const & devAcc ) :
        argsBuf( devHost, devAcc ), resBuf( devHost, devAcc )
        {
            for(size_t i = 0; i < ResBuf::size; ++i)
                resBuf.setArg(std::nan(""),i);
        }

    template< typename Queue >
    auto copyToDevice( Queue queue ) -> void
    {
        argsBuf.copyToDevice( queue );
        resBuf.copyToDevice( queue );
    }
};

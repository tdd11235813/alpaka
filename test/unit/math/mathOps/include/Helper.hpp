/** Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "Buffer.hpp"
#include "Defines.hpp"


template<
    typename TAcc,
    typename TData,
    typename Dim,
    typename Idx,
    Arity arity>
struct Helper
{
    static constexpr size_t extentBuf = 1000;

    // Theses types can be accessed from outside,
    // to access the static variables like ArgsBuf::size.
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

    // The argsBuf can have as many sub-buffer as needed (2D).
    // Example: [1.1, 1.2, 2.1, 2.2] than is 1.1 = x and 2.1 = y in abs( x, y ).
    ArgsBuf argsBuf;
    // The resBuf is 1D and saves the computed results regarding the argsBuf.
    ResBuf resBuf;


    template<
        typename DevHost,
        typename DevAcc
    >
    Helper(
        DevHost const & devHost,
        DevAcc const & devAcc
    ) :
        argsBuf(
            devHost,
            devAcc
        ),
        resBuf(
            devHost,
            devAcc
        )
    {
        // Fill the resBuf with nan,
        // so that a test fails if an argument wasn't used properly.
        for( size_t i = 0; i < ResBuf::size; ++i )
            resBuf.setArg(
                std::nan( "" ),
                i
            );
    }


    template< typename Queue >
    auto copyToDevice( Queue queue ) -> void
    {
        argsBuf.copyToDevice( queue );
        resBuf.copyToDevice( queue );
    }
};

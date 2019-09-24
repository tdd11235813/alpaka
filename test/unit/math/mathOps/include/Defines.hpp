/** Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

// New types need to be added to the switch-case in DataGen.hpp
enum class Range
{
    ONE_NEIGHBOURHOOD,
    POSITIVE_ONLY,
    POSITIVE_AND_ZERO,
    NOT_ZERO,
    UNRESTRICTED
};

// New types need to be added to the compute function in mathOps.cpp
enum class Arity
{
    UNARY = 1,
    BINARY = 2
};

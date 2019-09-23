#pragma once
#include <array>

enum class Range
{
    ONE_NEIGHBOURHOOD,
    POSITIVE_ONLY,
    POSITIVE_AND_ZERO,
    NOT_ZERO,
    UNRESTRICTED
};

enum class Arity
{
    UNARY = 1,
    BINARY = 2
};

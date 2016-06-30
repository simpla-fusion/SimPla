//
// Created by salmon on 16-6-27.
//

#ifndef SIMPLA_BOXUTILITY_H
#define SIMPLA_BOXUTILITY_H

#include "../sp_def.h"

namespace simpla { namespace gtl
{

box_type box_overlap(box_type const &first, box_type const &second)
{
    point_type lower, upper;

    for (int i = 0; i < 3; ++i)
    {
        lower[i] = std::max(std::get<0>(first)[i], std::get<0>(second)[i]);
        upper[i] = std::min(std::get<1>(first)[i], std::get<1>(second)[i]);
    }

    return std::make_tuple(lower, upper);
}

}}//namespace simpla{namespace gtl{

#endif //SIMPLA_BOXUTILITY_H

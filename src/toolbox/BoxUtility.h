//
// Created by salmon on 16-6-27.
//

#ifndef SIMPLA_BOXUTILITY_H
#define SIMPLA_BOXUTILITY_H

#include "SIMPLA_config.h"
#include <tuple>


namespace simpla { namespace toolbox
{

template<typename BoxT>
BoxT box3_intersection(BoxT const &left, BoxT const &right)
{
    BoxT res;
    for (int i = 0; i < 3; ++i)
    {
        std::get<0>(res)[i] = std::max(std::get<0>(left)[i], std::get<0>(right)[i]);
        std::get<1>(res)[i] = std::min(std::get<1>(left)[i], std::get<1>(right)[i]);
    }

    return res;


}

}}//namespace simpla{namespace toolbox{

#endif //SIMPLA_BOXUTILITY_H

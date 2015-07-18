/**
 *  @file boundary.h
 *  Created by salmon on 7/1/15.
 */
#ifndef SIMPLA_BOUNDARY_H
#define SIMPLA_BOUNDARY_H

#include "mesh_traits.h"
#include "../gtl/macro.h"
#include <map>
#include <bits/unordered_map.h>


namespace simpla {

template<typename ...> struct Boundary;

template<typename CS, typename TGeoObject>
struct Boundary<CS, TGeoObject>
{
    typedef traits::point_type_t<CS> point_type;
    typedef traits::vector_type_t<CS> normal_vector_type;


    std::unordered_multimap<id_type, TGeoObject const *> m_data_;

public:


    template<typename ...Args>
    auto insert(Args const &...args)
    DECL_RET_TYPE(m_data_.insert(std::forward<Args>(args)...))

    template<typename ...Args>
    auto erase(Args const &...args)
    DECL_RET_TYPE(m_data_.erase(std::forward<Args>(args)...))
};
}//namespace simpla

#endif //SIMPLA_BONDARY_H

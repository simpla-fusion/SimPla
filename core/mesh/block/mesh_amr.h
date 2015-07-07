//
// Created by salmon on 7/7/15.
//

#ifndef SIMPLA_MESH_AMR_H
#define SIMPLA_MESH_AMR_H

#include <list>
#include "../mesh_traits.h"
#include "../../geometry/coordinate_system.h"
#include "block.h"

namespace simpla {
namespace tags {
template<int LEVEL> struct amr : public std::integral_constant<int, LEVEL>
{
};
}


template<typename CS, int LEVEL>
struct Mesh<CS, tags::amr<LEVEL>> : public Block<geometry::traits::dimension<CS>::value, LEVEL>
{
    typedef Mesh<CS, tags::amr<LEVEL - 1> > finer_mesh;
    typedef Mesh<CS, tags::amr<LEVEL + 1> > coarser_mesh;

    typedef traits::point_type_t<CS> point_type;

    std::list<finer_mesh> m_finer_mesh_list_;


    void re_mesh();

    void make_finer_mesh();

};

template<typename CS>
using finest_mesh = Mesh<CS, tags::amr<0> >;


}//namespace simpla
#endif //SIMPLA_MESH_AMR_H

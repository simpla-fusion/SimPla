//
// Created by salmon on 7/7/15.
//

#ifndef SIMPLA_MESH_AMR_H
#define SIMPLA_MESH_AMR_H

#include <map>

namespace simpla
{
namespace tags
{
template<int LEVEL> struct amr;
}

typedef std::uint64_t mesh_id_type;

template<typename CS, int LEVEL>
struct Mesh<CS, tags::amr<LEVEL>> : public Block<traits::dimension<CS>::value, LEVEL>
{
	typedef Mesh<CS, tags::amr<LEVEL - 1>> fine_mesh;
	typedef Mesh<CS, tags::amr<LEVEL + 1>> coarse_mesh;

	typedef traits::point_type_t <CS> point_type;

	std::map<mesh_id_type, fine_mesh> m_fine_meshs_;


};
}//namespace simpla
#endif //SIMPLA_MESH_AMR_H

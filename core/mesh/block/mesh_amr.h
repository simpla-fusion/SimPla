//
// Created by salmon on 7/7/15.
//

#ifndef SIMPLA_MESH_AMR_H
#define SIMPLA_MESH_AMR_H

#include <map>
#include <list>
#include <utility>

#include "../mesh_traits.h"
#include "../../geometry/coordinate_system.h"
#include "block.h"
#include "block_id.h"

namespace simpla
{
namespace tags
{
template<int LEVEL> struct amr : public std::integral_constant<int, LEVEL>
{
};
}
template<typename ...> struct DOFHolder;


template<typename CS, int LEVEL>
struct Mesh<CS, tags::amr<LEVEL>> : public Block<geometry::traits::dimension<CS>::value, LEVEL>
{

	typedef Mesh<CS, tags::amr<LEVEL>> this_type;

	typedef Mesh<CS, tags::amr<LEVEL - 1> > finer_mesh;

	typedef Mesh<CS, tags::amr<LEVEL + 1> > coarser_mesh;

	typedef Block<geometry::traits::dimension<CS>::value, LEVEL> topology_type;

	using topology_type::index_tuple;

	typedef traits::point_type_t<CS> point_type;

	struct Observer
	{


		virtual void initialize() = 0;

		virtual void destroy() = 0;

		virtual void sync() = 0;

	};

private:

	typedef std::map<BlockID::value_type, finer_mesh> mesh_map;

	mesh_map m_finer_mesh_list_;

	point_type m_coord_min_, m_coord_max_;

	std::list<std::shared_ptr<Observer>> m_observers_;
public:

	Mesh(point_type const &xmin, point_type const &xmax, index_tuple const &d) :
			topology_type(d), m_coord_min_(xmin), m_coord_max_(xmax)
	{

	}

	~Mesh()
	{

	}

	BlockID::value_type add_finer_mesh(index_tuple const &imin, index_tuple const &imax);

	void remove_finer_mesh(BlockID::value_type);

	void remove_finer_mesh();

	std::tuple<point_type, point_type> box() const { return std::make_tuple(m_coord_min_, m_coord_max_); }

	using topology_type::dimensions;
	using topology_type::hash;

	void re_mesh();


};

template<typename CS>
using finest_mesh = Mesh<CS, tags::amr<0> >;


template<typename CS, int LEVEL>
Mesh<CS, tags::amr<LEVEL>>::Mesh(point_type const &xmin, point_type const &xmax, index_tuple const &d) :
		topology_type(d), m_coord_min_(xmin), m_coord_max_(xmax)
{

}

template<typename CS, int LEVEL>
Mesh<CS, tags::amr<LEVEL>>::~Mesh()
{

}

template<typename CS, int LEVEL>
BlockID::value_type Mesh<CS, tags::amr<LEVEL>>::add_finer_mesh(index_tuple const &imin, index_tuple const &imax)
{

	/**
	 * TODO:
	 *  1. check overlap
	 *  2.
	 */
	point_type xmin, xmax;

	auto res = m_finer_mesh_list_.emplace(BlockIDFactory::generator(LEVEL), finer_mesh(xmin, xmax, imax - imin));

	return res.second->first;
}

template<typename CS, int LEVEL>
void Mesh<CS, tags::amr<LEVEL>>::remove_finer_mesh(BlockID::value_type id)
{
	if (m_finer_mesh_list_.find(id) != m_finer_mesh_list_.end())
	{
		m_finer_mesh_list_[id].remove_finer_mesh();
		m_finer_mesh_list_.erase(id);
	}
}

void Mesh<CS, tags::amr<LEVEL>>::remove_finer_mesh()
{
	for (auto &item:m_finer_mesh_list_)
	{
		item.second.remove_finer_mesh();
	}
	m_finer_mesh_list_.clear();
}
}//namespace simpla
#endif //SIMPLA_MESH_AMR_H

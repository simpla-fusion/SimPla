/*
 * surface.h
 *
 *  Created on: 2014年4月21日
 *      Author: salmon
 */

#ifndef SURFACE_H_
#define SURFACE_H_

#include <stddef.h>
#include <utility>
#include <vector>

#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"
#include "../utilities/sp_type_traits.h"

namespace simpla
{

/**
 *  @ingroup Model Geometry
 *
 *
 */

/**
 *  @brief surface
 */

template<typename TM>
class Surface
{
public:
	typedef TM mesh_type;

	typedef typename mesh_type::iterator mesh_iterator;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef nTuple<3, nTuple<3, Real>> plane_type;

	typedef std::vector<std::pair<mesh_iterator, plane_type> > cell_type;

	typedef typename cell_type::iterator iterator;

	typedef typename cell_type::const_iterator const_iterator;

	mesh_type const &mesh;

	cell_type cell_list_;

	Surface(mesh_type const & pmesh) :
			mesh(pmesh)
	{
	}
	~Surface()
	{
	}

	void insert(mesh_iterator s, plane_type const & v)
	{
		cell_list_.push_back(std::make_pair(s, v));
	}
	iterator begin()
	{
		return cell_list_.begin();
	}
	iterator end()
	{
		return cell_list_.end();
	}
	const_iterator begin() const
	{
		return cell_list_.begin();
	}
	const_iterator end() const
	{
		return cell_list_.end();
	}

	Range<iterator> Split(int t_num, int t_id)
	{
		size_t s = cell_list_.size();
		return Range<iterator>(begin() + s * t_id / t_num, begin() + s * (t_id + 1) / t_num);
	}
	Range<const_iterator> Split(int t_num, int t_id) const
	{
		size_t s = cell_list_.size();
		return Range<const_iterator>(begin() + s * t_id / t_num, begin() + s * (t_id + 1) / t_num);
	}
};

template<typename TM>
auto Split(Surface<TM> & s, int t_num, int t_id)-> decltype(s.Split(t_num,t_id))
{
	return s.Split(t_num, t_id);
}
}
// namespace simpla

#endif /* SURFACE_H_ */

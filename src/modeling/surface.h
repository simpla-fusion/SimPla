/*
 * surface.h
 *
 *  Created on: 2014年4月21日
 *      Author: salmon
 */

#ifndef SURFACE_H_
#define SURFACE_H_

#include <map>

#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"

namespace simpla
{

template<typename TM>
class Surface
{
public:
	typedef TM mesh_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef nTuple<3, Real> Vec3;

	std::map<index_type, Vec3> cell_list_;
	typedef typename std::map<index_type, Vec3>::iterator iterator;
	typedef typename std::map<index_type, Vec3>::const_iterator const_iterator;
	mesh_type const &mesh;
	Surface(mesh_type const & pmesh)
			: mesh(pmesh)
	{
	}
	~Surface()
	{
	}

	void insert(index_type s, Vec3 const & v)
	{
		cell_list_.emplace(s, v);
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

};
}  // namespace simpla

#endif /* SURFACE_H_ */

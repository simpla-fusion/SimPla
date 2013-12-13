/*
 * domain.h
 *
 *  Created on: 2013年12月13日
 *      Author: salmon
 */

#ifndef DOMAIN_H_
#define DOMAIN_H_

#include <vector>

#include "../fetl/primitives.h"

namespace simpla
{
template<typename TM>
class Domain
{

public:
	typedef TM mesh_type;
	mesh_type const &mesh;
	typedef typename mesh_type::tag_type tag_type;
private:
	typename mesh_type::template Container<tag_type> tags_;

public:

	Domain(mesh_type const & m)
			: mesh(m), tags_(mesh.MakeContainer<0, tag_type>(0))
	{
	}
	~Domain()
	{
	}

	template<typename TFUN>
	void AssignMediaTag(TFUN const & fun, int tag)
	{
		mesh.AssignDomainMedia(fun, &tags_, tag);
	}

};

}  // namespace simpla

#endif /* DOMAIN_H_ */

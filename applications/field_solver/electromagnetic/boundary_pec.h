/*
 * boundary_pec.h
 *
 *  Created on: 2013年12月4日
 *      Author: salmon
 */

#ifndef BOUNDARY_PEC_H_
#define BOUNDARY_PEC_H_
#include "fetl/fetl.h"
#include "numeric/pointinpolygen.h"
namespace simpla
{
template<typename TM>
class PEC
{

public:
	typedef TM mesh_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::tag_type tag_type;
private:

	mesh_type const & mesh_;
	std::vector<index_type> bc_;

public:

	PEC(mesh_type const &m, tag_type in, tag_type out) :
			mesh_(m)
	{
		mesh_.GetElementOnInterface<1, mesh_type::PARALLEL>(in, out, &bc_);
	}

	template<typename TF>
	inline void SetElectricField(Field<Geometry<mesh_type, 1>, TF> * E) const
	{
		for (auto const &p : bc_)
		{
			(*E)(p) = 0;
		}
	}

}
;
}  // namespace simpla

#endif /* BOUNDARY_PEC_H_ */

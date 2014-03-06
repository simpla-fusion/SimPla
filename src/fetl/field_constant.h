/*
 * field_constant.h
 *
 *  Created on: 2012-3-15
 *      Author: salmon
 */

#ifndef FIELD_CONSTANT_H_
#define FIELD_CONSTANT_H_

#include "constant_ops.h"

namespace simpla
{
template<typename TG, int IFORM, typename TValue> struct Field;

template<typename TM, int IFORM, typename TV>
struct Field<TM, IFORM, Constant<TV> >
{

	typedef TM mesh_type;

	typedef TV value_type;

	static const int IForm = IFORM;

	static const int NDIMS = mesh_type::NDIMS;

	typedef typename mesh_type::index_type index_type;

	typedef typename Geometry<mesh_type, IForm>::template field_value_type<value_type> field_value_type;

	mesh_type const &mesh;

	const value_type v_;

	Field(mesh_type const &pmesh, value_type const & v)
			: mesh(pmesh), v_(v)
	{
	}
	~Field()
	{
	}
	inline const value_type & get(index_type s) const
	{
		return v_;
	}

	inline const value_type & operator[](index_type s) const
	{
		return get(s);
	}
};

}  // namespace simpla

#endif /* FIELD_CONSTANT_H_ */

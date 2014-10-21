/*
 * field_constant.h
 *
 *  created on: 2012-3-15
 *      Author: salmon
 */

#ifndef FIELD_CONSTANT_H_
#define FIELD_CONSTANT_H_

#include "../utilities/constant_ops.h"

namespace simpla
{
template<typename, unsigned int> struct Domain;
template<typename ... > struct _Field;

template<typename TM, unsigned int IFORM, typename TV>
struct _Field<Domain<TM, IFORM>, Constant<TV> >
{

	typedef TM mesh_type;

	typedef TV value_type;

	static const unsigned int IForm = IFORM;

	static const unsigned int NDIMS = mesh_type::NDIMS;

	typedef typename mesh_type::iterator iterator;

	typedef typename Geometry<mesh_type, IForm>::template field_value_type<
			value_type> field_value_type;

	mesh_type const &mesh;

	const value_type v_;

	_Field(mesh_type const &pmesh, value_type const & v) :
			mesh(pmesh), v_(v)
	{
	}
	~_Field()
	{
	}
	inline const value_type & get(iterator s) const
	{
		return v_;
	}

	inline const value_type & operator[](iterator s) const
	{
		return get(s);
	}
};

}  // namespace simpla

#endif /* FIELD_CONSTANT_H_ */

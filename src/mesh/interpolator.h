/*
 * interpolator.h
 *
 *  Created on: 2014年4月17日
 *      Author: salmon
 */

#ifndef INTERPOLATOR_H_
#define INTERPOLATOR_H_

#include <cstddef>

#include "../fetl/field.h"
#include "../utilities/primitives.h"

namespace simpla
{
template<typename, int, typename > class Field;

template<typename TM, typename Policy = std::nullptr_t>
class Interpolator
{

public:

	typedef TM mesh_type;
	typedef typename mesh_type::topology_type topology_type;
	typedef typename mesh_type::geometry_type geometry_type;
	typedef typename topology_type::coordinates_type coordinates_type;
	typedef typename topology_type::iterator iterator;
	typedef typename topology_type::compact_index_type compact_index_type;

	template<typename TF>
	static inline typename TF::value_type Gather_(TF const &f, coordinates_type r, compact_index_type shift)
	{
		mesh_type const & mesh = f.mesh;
		mesh.geometry_type::CoordinatesGlobalToLocal(&r);

		auto X = topology_type::DeltaIndex(0, shift);
		auto Y = topology_type::DeltaIndex(1, shift);
		auto Z = topology_type::DeltaIndex(2, shift);

		auto s = mesh.topology_type::CoordinatesGlobalToLocal(&r, shift) + topology_type::DeltaIndex(shift);

		return

		f[((s + X) + Y) + Z] * (r[0]) * (r[1]) * (r[2])

		+ f[((s + X) + Y) - Z] * (r[0]) * (r[1]) * (1.0 - r[2])

		+ f[((s + X) - Y) + Z] * (r[0]) * (1.0 - r[1]) * (r[2])

		+ f[((s + X) - Y) - Z] * (r[0]) * (1.0 - r[1]) * (1.0 - r[2])

		+ f[((s - X) + Y) + Z] * (1.0 - r[0]) * (r[1]) * (r[2])

		+ f[((s - X) + Y) - Z] * (1.0 - r[0]) * (r[1]) * (1.0 - r[2])

		+ f[((s - X) - Y) + Z] * (1.0 - r[0]) * (1.0 - r[1]) * (r[2])

		+ f[((s - X) - Y) - Z] * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
	}

	template<typename TExpr>
	static inline typename Field<mesh_type, VERTEX, TExpr>::field_value_type Gather(
	        Field<mesh_type, VERTEX, TExpr> const &f, coordinates_type x, unsigned long h = 0)
	{
		return Gather_(f, x, (h << (topology_type::INDEX_DIGITS * 3)));
	}

	template<typename TExpr>
	static inline typename Field<mesh_type, EDGE, TExpr>::field_value_type Gather(
	        Field<mesh_type, EDGE, TExpr> const &f, coordinates_type x, unsigned long h = 0)
	{

		return typename Field<mesh_type, EDGE, TExpr>::field_value_type( {

		Gather_(f, x, (topology_type::_DI >> (1 + h))),

		Gather_(f, x, (topology_type::_DJ >> (1 + h))),

		Gather_(f, x, (topology_type::_DK >> (1 + h)))

		});

	}
	template<typename TExpr>
	static inline typename Field<mesh_type, FACE, TExpr>::field_value_type Gather(
	        Field<mesh_type, FACE, TExpr> const &f, coordinates_type x, unsigned long h = 0)
	{
		return typename Field<mesh_type, EDGE, TExpr>::field_value_type( {

		Gather_(f, x, ((topology_type::_DJ | topology_type::_DK) >> (1 + h))),

		Gather_(f, x, ((topology_type::_DK | topology_type::_DI) >> (1 + h))),

		Gather_(f, x, ((topology_type::_DI | topology_type::_DJ) >> (1 + h))) });

	}
	template<typename TExpr>
	static inline typename Field<mesh_type, VOLUME, TExpr>::field_value_type Gather(
	        Field<mesh_type, VOLUME, TExpr> const &f, coordinates_type x, unsigned long h = 0)
	{
		return Gather_(f, x, (topology_type::_DA >> (1 + h)));

	}

	template<typename TF>
	static inline void Scatter_(coordinates_type r, typename TF::value_type const & v, compact_index_type shift, TF *f)
	{

		mesh_type const & mesh = f->mesh;
		auto X = topology_type::DeltaIndex(0, shift);
		auto Y = topology_type::DeltaIndex(1, shift);
		auto Z = topology_type::DeltaIndex(2, shift);

		mesh.geometry_type::CoordinatesGlobalToLocal(&r);

		auto s = mesh.topology_type::CoordinatesGlobalToLocal(&r, shift) + topology_type::DeltaIndex(shift);

		f->get(((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);

		f->get(((s + X) + Y) - Z) += v * (r[0]) * (r[1]) * (1.0 - r[2]);

		f->get(((s + X) - Y) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);

		f->get(((s + X) - Y) - Z) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);

		f->get(((s - X) + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);

		f->get(((s - X) + Y) - Z) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);

		f->get(((s - X) - Y) + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);

		f->get(((s - X) - Y) - Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
	}

	template<typename TExpr>
	static inline void Scatter(coordinates_type x, typename Field<mesh_type, VERTEX, TExpr>::field_value_type const & v,
	        Field<mesh_type, VERTEX, TExpr> *f, unsigned long h = 0)
	{
		Scatter_(x, v, 0UL, f);
	}

	template<typename TExpr>
	static inline void Scatter(coordinates_type x, typename Field<mesh_type, EDGE, TExpr>::field_value_type const & v,
	        Field<mesh_type, EDGE, TExpr> *f, unsigned long h = 0)
	{

		Scatter_(x, v[0], (topology_type::_DI >> (1 + h)), f);

		Scatter_(x, v[1], (topology_type::_DJ >> (1 + h)), f);

		Scatter_(x, v[2], (topology_type::_DK >> (1 + h)), f);
	}

	template<typename TExpr>
	static inline void Scatter(coordinates_type x, typename Field<mesh_type, FACE, TExpr>::field_value_type const & v,
	        Field<mesh_type, FACE, TExpr> *f, unsigned long h = 0)
	{

		Scatter_(x, v[0], ((topology_type::_DJ | topology_type::_DK) >> (1 + h)), f);

		Scatter_(x, v[1], ((topology_type::_DK | topology_type::_DI) >> (1 + h)), f);

		Scatter_(x, v[2], ((topology_type::_DI | topology_type::_DJ) >> (1 + h)), f);
	}

	template<typename TExpr>
	static inline void Scatter(coordinates_type x, typename Field<mesh_type, VOLUME, TExpr>::field_value_type const & v,
	        Field<mesh_type, VOLUME, TExpr> *f, unsigned long h = 0)
	{
		Scatter_(x, v, (topology_type::_DA >> (1 + h)), f);
	}

};

}  // namespace simpla

#endif /* INTERPOLATOR_H_ */

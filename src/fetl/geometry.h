/*
 * geometry.h
 *
 *  Created on: 2013-7-20
 *      Author: salmon
 */

#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include <fetl/primitives.h>
#include <cstddef>
#include <vector>

namespace simpla
{
template<typename, typename > struct Field;

template<typename TM, int IFORM> class Geometry;

template<typename > class GeometryTraits;

template<typename TM>
struct GeometryTraits<Geometry<TM, 0> >
{

	template<typename T> using field_value_type = T;
	typedef Real weight_type;
	typedef Real scatter_weight_type;
	typedef Real gather_weight_type;
};
template<typename TM>
struct GeometryTraits<Geometry<TM, 1> >
{

	template<typename T> using field_value_type = nTuple<3,T>;
	typedef nTuple<3, Real> weight_type;
	typedef nTuple<3, Real> scatter_weight_type;
	typedef nTuple<3, Real> gather_weight_type;

};
template<typename TM>
struct GeometryTraits<Geometry<TM, 2> >
{

	template<typename T> using field_value_type = nTuple<3,T>;
	typedef nTuple<3, Real> weight_type;
	typedef nTuple<3, Real> scatter_weight_type;
	typedef nTuple<3, Real> gather_weight_type;

};
template<typename TM>
struct GeometryTraits<Geometry<TM, 3> >
{

	template<typename T> using field_value_type = T;
	typedef Real weight_type;
	typedef Real scatter_weight_type;
	typedef Real gather_weight_type;

};

/**
 * @brief Geometry
 *
 * @ingroup Field Expression
 * */
template<typename TM, int IFORM>
class Geometry
{
public:

	static const int IForm = IFORM;

	static const int NUM_OF_DIMS = TM::NUM_OF_DIMS;

	typedef TM mesh_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	template<typename Element> using Container=typename mesh_type::template Container<Element>;

	typedef typename mesh_type::index_type index_type;

	typedef Geometry<mesh_type, IFORM> this_type;

	template<typename T> using field_value_type=
	typename GeometryTraits<this_type>::template field_value_type<T>;

	mesh_type const* mesh;

	Geometry() :
			mesh(NULL)
	{

	}
	Geometry(mesh_type const & g) :
			mesh(&g)
	{
	}
	Geometry(mesh_type const * g) :
			mesh(g)
	{
	}
	template<int IF>
	Geometry(Geometry<mesh_type, IF> const & g) :
			mesh(g.mesh)
	{
	}

	template<int IL, typename TL, int IR, typename TR>
	Geometry(Field<Geometry<mesh_type, IL>, TL> const & l,
			Field<Geometry<mesh_type, IR>, TR> const & r) :
			mesh(l.mesh)
	{
	}
	template<typename TL, typename TR>
	Geometry(TL const & l, TR const & r) :
			mesh(get_mesh(l, r))
	{
	}
	Geometry(this_type const & r) :
			mesh(r.mesh)
	{
	}

	~Geometry()
	{
	}

	template<typename E> inline Container<E> MakeContainer(E const & d =
			E()) const
	{
		return std::move(mesh->MakeContainer(IFORM, d));
	}

	template<typename TExpr, typename Fun>
	inline void ForEach(Field<this_type, TExpr> & v, Fun const &fun) const
	{
		mesh->ForEach(IFORM, v, fun);
	}

	template<typename Fun>
	inline void ForEach(Fun const &fun) const
	{
		mesh->ForEach(IFORM, fun);
	}
//	template<typename TE>
//	inline typename Field<this_type, TE>::Value IntepolateFrom(
//			Field<this_type, TE> const & f, Coordinates const & s,
//			Real effect_radius) const
//	{
//		return (f[0]);
//	}
//
//	template<typename TE>
//	inline void IntepolateTo(Field<this_type, TE> const & f,
//			typename Field<this_type, TE>::Value const & v,
//			Coordinates const & s, Real effect_radius) const
//	{
//	}

private:
	template<int IL, typename TR> static typename std::enable_if<
			!std::is_same<Geometry<mesh_type, IL>, TR>::value, mesh_type const *>::type get_mesh(
			Geometry<mesh_type, IL> const & l, TR const & r)
	{
		return (l.mesh);
	}

	template<int IR, typename TL> static typename std::enable_if<
			!std::is_same<Geometry<mesh_type, IR>, TL>::value, mesh_type const *>::type get_mesh(
			TL const & l, Geometry<mesh_type, IR> const & r)
	{
		return (r.mesh);
	}

//	template<int IL, int IR> static Mesh const * //
//	get_mesh(Geometry<Mesh, IL> const & l, Geometry<Mesh, IR> const & r)
//	{
//		return (l.mesh);
//	}
};

}  // namespace simpla

#endif /* GEOMETRY_H_ */

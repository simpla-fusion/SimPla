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

	typedef TM Mesh;

	typedef typename Mesh::coordinates_type Coordinates;

	template<typename Element> using Container=typename Mesh::template Container<Element>;

	typedef typename Mesh::index_type index_type;

	typedef Geometry<Mesh, IFORM> this_type;

	Mesh const* mesh;

	Geometry() :
			mesh(NULL)
	{

	}
	Geometry(Mesh const & g) :
			mesh(&g)
	{
	}
	Geometry(Mesh const * g) :
			mesh(g)
	{
	}
	template<int IF>
	Geometry(Geometry<Mesh, IF> const & g) :
			mesh(g.mesh)
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
	template<int IL, typename TR> static Mesh const * //
	get_mesh(Geometry<Mesh, IL> const & l, TR const & r)
	{
		return (l.mesh);
	}

//	template<typename TL, int IR> static Mesh const * //
//	get_mesh(TL const & l, Geometry<Mesh, IR> const & r)
//	{
//		return (r.mesh);
//	}

//	template<int IL, int IR> static Mesh const * //
//	get_mesh(Geometry<Mesh, IL> const & l, Geometry<Mesh, IR> const & r)
//	{
//		return (l.mesh);
//	}
};

}  // namespace simpla

#endif /* GEOMETRY_H_ */

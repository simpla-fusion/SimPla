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

template<typename TM, int IFORM>
class Geometry
{
public:

	static const int NUM_OF_DIMS = TM::NUM_OF_DIMS;

	typedef TM Mesh;

	typedef typename Mesh::Coordinates Coordinates;

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

	template<typename E> inline Container<E> makeContainer(E const & d =
			E()) const
	{
		return std::move(Container<E>(get_num_of_elements(), d));
	}

	template<typename TE>
	inline typename Field<this_type, TE>::Value IntepolateFrom(
			Field<this_type, TE> const & f, Coordinates const & s,
			Real effect_radius) const
	{
		return (f[0]);
	}

	template<typename TE>
	inline void IntepolateTo(Field<this_type, TE> const & f,
			typename Field<this_type, TE>::Value const & v,
			Coordinates const & s, Real effect_radius) const
	{
	}
	inline size_t get_num_of_elements() const
	{
		return (mesh->get_num_of_elements(IFORM));
	}

	inline typename std::vector<size_t>::const_iterator get_center_elements_begin() const
	{
		return (mesh->center_ele_[IFORM].begin());
	}
	inline typename std::vector<size_t>::const_iterator get_center_elements_end() const
	{
		return (mesh->center_ele_[IFORM].end());
	}

	inline size_t get_num_of_comp() const
	{
		return (mesh->get_num_of_comp(IFORM));
	}
	inline size_t get_num_of_center_elements() const
	{
		return (mesh->get_num_of_center_elements(IFORM));
	}

	template<typename T>
	inline size_t get_cell_num(T const & p) const
	{
		return (mesh->get_cell_num(p));
	}
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

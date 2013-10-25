/*
 * geometry.h
 *
 *  Created on: 2013-7-20
 *      Author: salmon
 */

#ifndef GEOMETRY_H_
#define GEOMETRY_H_
#include <type_traits>
#include <utility>
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
	Geometry(this_type const & r) :
			mesh(r.mesh)
	{
	}

	~Geometry()
	{
	}

	template<typename Element> using Container=std::vector<Element>;

	template<typename E> inline Container<E>   makeContainer(E const & d) const
	{
		return std::move(Container<E>(get_num_of_elements(), d));
	}

	template<typename E> inline Container<E> makeContainer() const
	{
		return std::move(Container<E>(get_num_of_elements()));
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

};

}  // namespace simpla

#endif /* GEOMETRY_H_ */

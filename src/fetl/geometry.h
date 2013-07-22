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
namespace simpla
{
template<typename, typename > struct Field;

template<typename TG, int IFORM>
class Geometry
{
public:
	typedef TG Grid;

	typedef Geometry<Grid, IFORM> ThisType;

	typedef typename Grid::CoordinatesType CoordinatesType;

	typedef typename Grid::IndexType IndexType;

	Grid const & grid;

	Geometry(Grid const & g) :
			grid(g)
	{
	}

	Geometry(ThisType const &)=default;

	~Geometry()
	{
	}
	template<typename TE>
	inline typename Field<ThisType,TE>::Value
	IntepolateFrom(Field<ThisType,TE> const & f,CoordinatesType const & s ,Real effect_radius) const
	{
		return (f[0]);
	}

	template<typename TE>
	inline void IntepolateTo( Field<ThisType,TE> const & f, typename Field<ThisType,TE>::Value const & v,
			CoordinatesType const & s ,Real effect_radius)const
	{
	}
	inline size_t get_num_of_elements() const
	{
		return (grid.get_num_of_elements(IFORM));
	}

	inline typename std::vector<IndexType>::const_iterator get_center_elements_begin( ) const
	{
		return (grid.center_ele_[IFORM].begin());
	}
	inline typename std::vector<IndexType>::const_iterator get_center_elements_end( ) const
	{
		return (grid.center_ele_[IFORM].end());
	}

	inline size_t get_num_of_comp()const
	{
		return (grid.get_num_of_comp(IFORM));
	}

	template<typename TEXPR,typename INDEX> inline auto
	eval(TEXPR const & expr,INDEX const &idx) const
	DECL_RET_TYPE((grid.eval(expr,idx)))
//	{
//		return (grid.eval(expr,idx));
//	}
};

}  // namespace simpla

#endif /* GEOMETRY_H_ */

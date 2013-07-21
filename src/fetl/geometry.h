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
template<typename TG, int IFORM>
class Geometry
{
	typedef TG Grid;

	typedef Geometry<Grid, IFORM> ThisType;

	typedef typename Grid::Coordinates Coordinates;

	Grid const & grid;

	Geometry(Grid const & g) :
			grid(g)
	{
	}

	Geometry(ThisType const &)=default;

	~Geometry()
	{
	}
	template<typename TF>
	auto IntepolateFrom(TF const & f,Coordinates const & s ,Real effect_radius)->decltype(f[0])
	{
		return (f[0]);
	}

	template<typename TF>
	void IntepolateTo(TF const & f,typename TF::Value const & v,Coordinates const & s ,Real effect_radius)
	{
	}
	size_t get_num_of_elements()
	{
		return (0);
	}
};

}  // namespace simpla

#endif /* GEOMETRY_H_ */

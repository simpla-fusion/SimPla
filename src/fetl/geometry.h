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

template<typename TG>
class ZeroForm
{
	typedef TG Grid;

	typedef ZeroForm<TG> ThisType;

	typedef typename Grid::Coordinates Coordinates;

	Grid const & grid;

	ZeroForm(Grid const & g) :
			grid(g)
	{
	}

	ZeroForm(ThisType const &)=default;

	~ZeroForm()
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

template<typename TG>
class OneForm
{
	typedef TG Grid;

	typedef OneForm<TG> ThisType;

	typedef typename Grid::Coordinates Coordinates;

	Grid const & grid;

	OneForm(Grid const & g) :
			grid(g)
	{
	}

	OneForm(ThisType const &)=default;

	~OneForm()
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

template<typename TG>
class TwoForm
{
	typedef TG Grid;

	typedef TwoForm<TG> ThisType;

	typedef typename Grid::Coordinates Coordinates;

	Grid const & grid;

	TwoForm(Grid const & g) :
			grid(g)
	{
	}

	TwoForm(ThisType const &)=default;

	~TwoForm()
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

template<typename TG>
class ThreeForm
{
	typedef TG Grid;

	typedef ThreeForm<TG> ThisType;

	typedef typename Grid::Coordinates Coordinates;

	Grid const & grid;

	ThreeForm(Grid const & g) :
			grid(g)
	{
	}

	ThreeForm(ThisType const &)=default;

	~ThreeForm()
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

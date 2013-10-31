/*
 * field.h
 *
 *  Created on: 2013-7-19
 *      Author: salmon
 */

#ifndef FIELD_H_
#define FIELD_H_

#include <fetl/primitives.h>

//#include "utilities/log.h"

namespace simpla
{

template<typename > class GeometryTraits;
template<typename TF> class ReadCache;
template<typename TF> class WriteCache;

/***
 *
 * @brief Field
 *
 * @ingroup Field Expression
 *
 */

template<typename TGeometry, typename TValue>
struct Field: public TGeometry, public TGeometry::template Container<TValue>
{
public:

	typedef TGeometry geometry_type;

	typedef typename TGeometry::mesh_type mesh_type;

	typedef typename TGeometry::template Container<TValue> container_type;

	typedef typename TGeometry::index_type index_type;

	typedef typename container_type::value_type value_type;

	typedef typename geometry_type::template field_value_type<value_type> field_value_type;

	typedef Field<geometry_type, value_type> this_type;

	friend class ReadCache<this_type> ;

	friend class WriteCache<this_type> ;

	template<typename TG>
	Field(TG const &g) :
			geometry_type(g), container_type(
					std::move(
							geometry_type::template MakeContainer<value_type>()))
	{
	}

	Field() = default;

	Field(this_type const & f) = delete;

	Field(this_type &&rhs) = delete;

	virtual ~Field()
	{
	}

	void swap(this_type & rhs)
	{
		geometry_type::swap(rhs.geometry_);
		container_type::swap(rhs);
	}

#ifdef DEBUG  // check boundary
	inline value_type & operator[](typename geometry_type::index_type const &s)
	{

		return container_type::at(s);

	}
	inline value_type const & operator[](
			typename geometry_type::index_type const &s) const
	{
		return container_type::at(s);
	}
#endif

	template<typename Fun>
	inline void ForEach(Fun const &fun)
	{
		geometry_type::ForEach(

		[this,&fun](typename geometry_type::index_type s)
		{
			fun((*this)[s]);
		}

		);
	}

	inline this_type & operator=(this_type const & rhs)
	{
		geometry_type::ForEach(

		[this, &rhs](typename geometry_type::index_type s)
		{
			(*this)[s]=rhs[s];
		}

		);
		return (*this);
	}

#define DECL_SELF_ASSIGN( _OP_ )                                                  \
	template<typename TR> inline this_type &                                      \
	operator _OP_(Field<TGeometry, TR> const & rhs)                               \
	{                                                                             \
		geometry_type::ForEach(												      \
		[this, &rhs](typename geometry_type::index_type const &s)                 \
		{	(*this)[s] _OP_ rhs[s];});                                            \
		return (*this);                                                           \
	}                                                                             \
	template<typename TR> inline this_type &                                      \
	operator _OP_(TR const & rhs)                                                 \
	{                                                                             \
		geometry_type::ForEach(									                  \
		[this, &rhs](typename geometry_type::index_type const &s)                 \
		{	(*this)[s] _OP_ rhs ;});                                              \
		return (*this);                                                           \
	}

	DECL_SELF_ASSIGN(=)

DECL_SELF_ASSIGN	(+=)

	DECL_SELF_ASSIGN(-=)

	DECL_SELF_ASSIGN(*=)

	DECL_SELF_ASSIGN(/=)
#undef DECL_SELF_ASSIGN

	inline field_value_type operator()( coordinates_type const &x) const
	{
		return std::move(Gather(x));
	}

	inline field_value_type Gather( coordinates_type const &x) const
	{

		coordinates_type pcoords;

		index_type s = mesh->SearchCell(x, &pcoords);

		return std::move(Gather(s, pcoords));

	}

	inline field_value_type Gather( index_type const & s,coordinates_type const &pcoord) const
	{

		int num_of_vertices = mesh->GetCellNumOfVertices(Int2Type<IFORM>(), s);

		std::vector<index_type> points(num_of_vertices);

		std::vector<typename GeometryTraits<geometry_type>::gather_weight_type> weights(num_of_vertices);

		mesh->GetCellVertices(Int2Type<0>(), s, points);

		mesh->CalcuateWeight(Int2Type<0>(), pcoords, weights);

		field_value_type res;

		res *= 0;

		for (auto it1=points.begin(), it2=weights.begin();
				it1!=points.end()&& it2!=weights.end();++it1,++it2 )
		{
			res += this->operator[](*it1) * (*it2);
		}

		return std::move(res);

	}

	template<typename TV >
	inline void Scatter(TV const & v, coordinates_type const &x )
	{
		coordinates_type pcoords;

		index_type s = mesh->SearchCell(x, &pcoords);

		Scatter(v,s,pcoords);

	}
	template<typename TV >
	inline void Scatter(TV const & v,index_type const & s,coordinates_type const &pcoord )
	{

		int num_of_vertices = mesh->GetCellNumOfVertices(Int2Type<IFORM>(), s);

		std::vector<index_type> points(num_of_vertices);

		std::vector<typename GeometryTraits<geometry_type>::scatter_weight_type> weights(num_of_vertices);

		mesh->GetCellVertices(Int2Type<0>(), s, points);

		mesh->CalcuateWeight(Int2Type<0>(), pcoords, weights);

		std::vector<value_type> cache(num_of_vertices);

		for (auto it1=cache.begin(), it2=weights.begin();
				it1!=cache.end() && it2!=weights.end();++it1,++it2 )
		{
			// FIXME: this incorrect for vector field interpolation
			*it1 += Dot(v ,*it2);
		}

		Scatter(points,cache);
	}

	template<typename TV >
	inline void Scatter(std::vector<index_type> const & points,std::vector<value_type> & cache)
	{
		//FIXME: this is not thread safe, need a mutex lock

		for (auto it1=points.begin(), it2=cache.begin();
				it!=points.end()&&it2!=cache.end();++it1,++it2 )
		{
			(*this)[*it1] += *it2;
		}

	}

	//	inline auto Get(CoordinatesType const &x,Real effect_radius=0)const
//	DECL_RET_TYPE( (geometry.IntepolateFrom(*this,x,effect_radius)))
//
//	inline auto Put(ValueType const & v,CoordinatesType const &x,Real effect_radius=0)
//	DECL_RET_TYPE(( geometry.IntepolateTo(*this,v,x,effect_radius)))

};

}
// namespace simpla

#endif /* FIELD_H_ */

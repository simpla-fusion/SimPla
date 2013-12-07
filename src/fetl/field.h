/*
 * field.h
 *
 *  Created on: 2013-7-19
 *      Author: salmon
 */

#ifndef FIELD_H_
#define FIELD_H_

#include "primitives.h"
#include "../utilities/container.h"
#include "../utilities/log.h"
#include "../mesh/mesh.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>
namespace simpla
{
template<typename TG, typename TValue> struct Field;

/***
 *
 * @brief Field
 *
 * @ingroup Field Expression
 *
 */

template<typename TG, typename TValue>
struct Field: public Container<TValue>::type
{
public:

	typedef TG geometry_type;

	typedef typename geometry_type::mesh_type mesh_type;

	enum
	{
		IForm = geometry_type::IForm
	};

	typedef TValue value_type;

	typedef Field<geometry_type, value_type> this_type;

	static const int NUM_OF_DIMS = mesh_type::NUM_OF_DIMS;

	typedef typename Container<value_type>::type base_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename geometry_type::template field_value_type<value_type> field_value_type;

	mesh_type const &mesh;

	Field(mesh_type const &pmesh) :
			base_type(
					std::move(
							pmesh.template MakeContainer<IForm, value_type>())), mesh(
					pmesh)
	{
	}

	Field(this_type const & f) = delete;

	Field(this_type &&rhs) = delete;

	virtual ~Field()
	{
	}

	void swap(this_type & rhs)
	{
		base_type::swap(rhs);
	}

	inline value_type & operator[](size_t s)
	{
		return base_type::operator[](s);
	}
	inline value_type const & operator[](size_t s) const
	{
		return base_type::operator[](s);
	}

	template<typename ... TI>
	inline value_type & get(TI ...s)
	{
		return base_type::operator[](mesh.template Component<IForm>(s...));
	}
	template<typename ...TI>
	inline value_type const & get(TI ...s) const
	{
		return base_type::operator[](mesh.template Component<IForm>(s...));
	}

	inline this_type & operator=(this_type const & rhs)
	{
		mesh.ForA;;(

		[](value_type &l, value_type const &r)
		{	l = r;},

		this, rhs);

		return (*this);
	}

#define DECL_SELF_ASSIGN( _OP_ )                                                                   \
	                                                                                               \
	template<typename TR> inline this_type &                                                       \
	operator _OP_(Field<geometry_type, TR> const & rhs)                                            \
	{                                                                                              \
	typedef typename Field<geometry_type, TR>::value_type r_value_type;                            \
		mesh.ForAll( [](value_type &l, r_value_type const& r) { l _OP_ r;},this,rhs);             \
		return (*this);                                                                            \
	}                                                                                              \
	template<typename TR> inline this_type &                                                       \
	operator _OP_(TR const & rhs)                                                                  \
	{                                                                                              \
		mesh.ForAll( [](value_type &l, TR const & r){	l _OP_ r;}  ,this, rhs);                   \
		return (*this);                                                                            \
	}                                                                                              \


	DECL_SELF_ASSIGN(=)

DECL_SELF_ASSIGN	(+=)

	DECL_SELF_ASSIGN (-=)

	DECL_SELF_ASSIGN (*=)

	DECL_SELF_ASSIGN(/=)
#undef DECL_SELF_ASSIGN

	inline field_value_type operator()(coordinates_type const &x) const
	{
		return std::move(Gather(x));
	}

//	inline field_value_type Gather(coordinates_type const &x) const
//	{
//
//		coordinates_type pcoords;
//
//		index_type s = mesh.SearchCell(x, &pcoords);
//
//		return std::move(Gather(s, pcoords));
//
//	}
//
//	inline field_value_type Gather(index_type const & s,
//			coordinates_type const &pcoords) const
//	{
//
//		std::vector<index_type> points;
//
//		std::vector<typename geometry_type::gather_weight_type> weights;
//
//		mesh.GetAffectedPoints(Int2Type<IForm>(), s, points);
//
//		mesh.CalcuateWeights(Int2Type<IForm>(), pcoords, weights);
//
//		field_value_type res;
//
//		res *= 0;
//
//		auto it1 = points.begin();
//		auto it2 = weights.begin();
//		for (; it1 != points.end() && it2 != weights.end(); ++it1, ++it2)
//		{
//
//			try
//			{
//				res += this->at(*it1) * (*it2);
//
//			} catch (std::out_of_range const &e)
//			{
//#ifndef NDEBUG
//				WARNING
//#else
//						VERBOSE
//#endif
//<<				e.what() <<"[ idx="<< *it1<<"]";
//			}
//
//		}
//
//		return std::move(res);
//
//	}
//
//	template<typename TV>
//	inline void Scatter(TV const & v, coordinates_type const &x)
//	{
//		coordinates_type pcoords;
//
//		index_type s = mesh.SearchCell(x, &pcoords);
//
//		Scatter(v, s, pcoords);
//
//	}
//	template<typename TV>
//	inline void Scatter(TV const & v, index_type const & s,
//			coordinates_type const &pcoords, int affected_region = 1)
//	{
//
//		std::vector<index_type> points;
//
//		std::vector<typename geometry_type::scatter_weight_type> weights;
//
//		mesh.GetAffectedPoints(Int2Type<IForm>(), s, points);
//
//		mesh.CalcuateWeights(Int2Type<IForm>(), pcoords, weights);
//
//		auto it1 = points.begin();
//		auto it2 = weights.begin();
//		for (; it1 != points.end() && it2 != weights.end(); ++it1, ++it2)
//		{
//			// FIXME: this incorrect for vector field interpolation
//
//			try
//			{
//
//				this->at(*it1) += Dot(v, *it2);
//
//			} catch (std::out_of_range const &e)
//			{
//#ifndef NDEBUG
//				WARNING
//#else
//						VERBOSE
//#endif
//<<				e.what() <<"[ idx="<< *it1<<"]";
//			}
//		}
//
//	}
//
//	inline void Scatter(std::vector<index_type> const & points,std::vector<value_type> & cache)
//	{
//		//FIXME: this is not thread safe, need a mutex lock
//
//		auto it2=cache.begin();
//		auto it1=points.begin();
//		for(;it2!=cache.end() && it1!=points.end(); ++it1,++it2 )
//		{
//			try
//			{
//
//				this->at(*it1) += *it2;
//
//			}
//			catch(std::out_of_range const &e)
//			{
//#ifndef NDEBUG
//				WARNING
//#else
//				VERBOSE
//#endif
//				<< e.what() <<"[ idx="<< *it1<<"]";
//
//			}
//		}
//
//	}
};

}
// namespace simpla

#endif /* FIELD_H_ */

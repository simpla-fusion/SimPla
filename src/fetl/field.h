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
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>
namespace simpla
{
template<typename TG, typename TValue> struct Field;
template<typename, int> struct Geometry;

template<typename T>
struct FieldTraits
{
	enum
	{
		is_field = false
	};

	enum
	{
		IForm = 0
	}
	;
	typedef T value_type;
};

template<typename TM, int IFORM, typename TExpr>
struct FieldTraits<Field<Geometry<TM, IFORM>, TExpr> >
{
	typedef Field<Geometry<TM, IFORM>, TExpr> this_type;
	enum
	{
		is_field = true
	};

	enum
	{
		IForm = IFORM
	}
	;
	typedef typename this_type::value_type value_type;
};

template<typename TL>
struct is_field
{
	static const bool value = FieldTraits<TL>::is_field;
};
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
	inline value_type & index(TI ...s)
	{
		return mesh.index(*this, s...);
	}
	template<typename ...TI>
	inline value_type index(TI ...s) const
	{
		return mesh.index(*this, s...);
	}

	inline this_type & operator=(this_type const & rhs)
	{
		mesh.ForEach(

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
		mesh.ForEach( [](value_type &l, r_value_type const& r) { l _OP_ r;},this,rhs);             \
		return (*this);                                                                            \
	}                                                                                              \
	template<typename TR> inline this_type &                                                       \
	operator _OP_(TR const & rhs)                                                                  \
	{                                                                                              \
		mesh.ForEach( [](value_type &l, TR const & r){	l _OP_ r;}  ,this, rhs);                   \
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

template<typename TM, int IL, int TOP, typename TL>
struct Field<Geometry<TM, IL>, UniOp<TOP, TL> >
{

private:

	typename ConstReferenceTraits<TL>::type l_;

public:

	typedef Field<Geometry<TM, IL>, UniOp<TOP, TL> > this_type;
	TM const & mesh;

	enum
	{
		IForm = IL
	};

	Field(TL const & l) :
			mesh(l.mesh), l_(l)
	{
	}
	template<typename ... TI> inline auto index(TI ... s) const
	DECL_RET_TYPE((_FieldOpEval(Int2Type<TOP>(), l_, s...)))
//
//	template<typename ... TI> inline auto index(TI ... s) const
//	DECL_RET_TYPE((_FieldOpEval(Int2Type<TOP>(), l_, s...)))

	typedef decltype(_FieldOpEval(Int2Type<TOP>(),std::declval<TL>(),0 )) value_type;

};

template<typename TM, int IFORM, int TOP, typename TL, typename TR>
struct Field<Geometry<TM, IFORM>, BiOp<TOP, TL, TR> >
{

private:
	typename ConstReferenceTraits<TL>::type l_;
	typename ConstReferenceTraits<TR>::type r_;

public:
	TM const & mesh;
	typedef Field<Geometry<TM, IFORM>, BiOp<TOP, TL, TR> > this_type;
	enum
	{
		IForm = IFORM
	};

	Field(TL const & l, TR const & r) :
			mesh(get_mesh(l, r)), l_(l), r_(r)
	{
	}

	template<typename ... TI> inline auto index(TI ... s) const
	DECL_RET_TYPE((_FieldOpEval(Int2Type<TOP>(), l_, r_, s...)))

	typedef decltype(_FieldOpEval(Int2Type<TOP>(),std::declval<TL>(),std::declval<TR>(),0 )) value_type;

private:

	template<int IL, typename VL, typename VR> static inline TM const & get_mesh(
			Field<Geometry<TM, IL>, VL> const & l, VR const & r)
	{
		return (l.mesh);
	}
	template<typename VL, int IR, typename VR> static inline TM const & get_mesh(
			VL const & l, Field<Geometry<TM, IR>, VR> const & r)
	{
		return (r.mesh);
	}

	template<int IL, typename VL, int IR, typename VR> static inline TM const & get_mesh(
			Field<Geometry<TM, IL>, VL> const & l,
			Field<Geometry<TM, IR>, VR> const & r)
	{
		return (l.mesh);
	}

}
;

}
// namespace simpla

#endif /* FIELD_H_ */

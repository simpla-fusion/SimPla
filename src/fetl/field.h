/*
 * field.h
 *
 *  Created on: 2013-7-19
 *      Author: salmon
 */

#ifndef FIELD_H_
#define FIELD_H_

#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../utilities/log.h"
#include "primitives.h"

namespace simpla
{

/***
 *
 * @brief Field
 *
 * @ingroup Field Expression
 *
 */

template<typename TG, typename TValue>
struct Field: public TG::mesh_type::template Container<TValue>::type
{
public:

	typedef TG geometry_type;

	typedef typename geometry_type::mesh_type mesh_type;

	static const int IForm = geometry_type::IForm;

	typedef TValue value_type;

	typedef Field<geometry_type, value_type> this_type;

	static const int NUM_OF_DIMS = mesh_type::NUM_OF_DIMS;

	typedef typename mesh_type::template Container<value_type>::type base_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename geometry_type::template field_value_type<value_type> field_value_type;

	mesh_type const &mesh;

	Field(mesh_type const &pmesh, value_type default_value = value_type()) :
			base_type(
					std::move(
							pmesh.template MakeContainer<value_type>(IForm,
									default_value))), mesh(pmesh)
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

#ifdef DEBUG  // check boundary
	inline value_type & operator[](index_type const &s)
	{

		return base_type::at(s);

	}
	inline value_type const & operator[](index_type const &s) const
	{
		return base_type::at(s);
	}
#endif

	template<typename Fun>
	inline void ForEach(Fun const &fun)
	{
		mesh.ForEach(IForm,

		[this,&fun](index_type s)
		{
			fun((*this)[s]);
		}

		);
	}

	template<typename Fun>
	inline void ForEach(Fun const &fun) const
	{
		mesh.ForEach(IForm,

		[this,&fun](index_type s)
		{
			fun((*this)[s]);
		}

		);
	}

	inline this_type & operator=(this_type const & rhs)
	{
		mesh.ForEach(IForm,

		[this, &rhs](index_type s)
		{
			(*this)[s]=rhs[s];
		}

		);
		return (*this);
	}

#define DECL_SELF_ASSIGN( _OP_ )                                                  \
	template<typename TR> inline this_type &                                      \
	operator _OP_(TR const & rhs)                               \
	{                                                                             \
		mesh.ForEach(IForm,												      \
		[this, &rhs](index_type const &s)                 \
		{	(*this)[s] _OP_ index(rhs,s);});                                            \
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

		index_type s = mesh.SearchCell(x, &pcoords);

		return std::move(Gather(s, pcoords));

	}

	inline field_value_type Gather( index_type const & s,coordinates_type const &pcoords) const
	{

		std::vector<index_type> points;

		std::vector<typename geometry_type::gather_weight_type> weights;

		mesh.GetAffectedPoints(Int2Type<IForm>(), s, points);

		mesh.CalcuateWeights(Int2Type<IForm>(), pcoords, weights);

		field_value_type res;

		res *= 0;

		auto it1=points.begin();
		auto it2=weights.begin();
		for(;it1!=points.end() && it2!=weights.end(); ++it1,++it2 )
		{

			try
			{
				res += this->at(*it1) * (*it2);

			}
			catch(std::out_of_range const &e)
			{
#ifndef NDEBUG
				WARNING
#else
				VERBOSE
#endif
				<< e.what() <<"[ idx="<< *it1<<"]";
			}

		}

		return std::move(res);

	}

	template<typename TV >
	inline void Scatter(TV const & v, coordinates_type const &x )
	{
		coordinates_type pcoords;

		index_type s = mesh.SearchCell(x, &pcoords);

		Scatter(v,s,pcoords);

	}
	template<typename TV >
	inline void Scatter(TV const & v,index_type const & s,coordinates_type const &pcoords ,int affected_region=1)
	{

		std::vector<index_type> points;

		std::vector<typename geometry_type::scatter_weight_type> weights;

		mesh.GetAffectedPoints(Int2Type<IForm>(), s, points);

		mesh.CalcuateWeights(Int2Type<IForm>(), pcoords, weights);

		auto it1=points.begin();
		auto it2=weights.begin();
		for(;it1!=points.end() && it2!=weights.end(); ++it1,++it2 )
		{
			// FIXME: this incorrect for vector field interpolation

			try
			{

				this->at(*it1) += Dot(v ,*it2);

			}
			catch(std::out_of_range const &e)
			{
#ifndef NDEBUG
				WARNING
#else
				VERBOSE
#endif
				<< e.what() <<"[ idx="<< *it1<<"]";
			}
		}

	}

	inline void Scatter(std::vector<index_type> const & points,std::vector<value_type> & cache)
	{
		//FIXME: this is not thread safe, need a mutex lock

		auto it2=cache.begin();
		auto it1=points.begin();
		for(;it2!=cache.end() && it1!=points.end(); ++it1,++it2 )
		{
			try
			{

				this->at(*it1) += *it2;

			}
			catch(std::out_of_range const &e)
			{
#ifndef NDEBUG
				WARNING
#else
				VERBOSE
#endif
				<< e.what() <<"[ idx="<< *it1<<"]";

			}
		}

	}
};

template<typename TM, int IL, int TOP, typename TL>
struct Field<Geometry<TM, IL>, UniOp<TOP, TL> >
{

private:

	typename ConstReferenceTraits<TL>::type l_;

public:

	TM const & mesh;

	typedef decltype(
			_OpEval(Int2Type<TOP>(),
					std::declval<typename std::remove_reference<TL>::type const&>()
					,std::declval<typename TM::index_type>())

	) value_type;

	typedef Geometry<TM, IL> geometry_type;

	typedef typename geometry_type::template field_value_type<value_type> field_value_type;

	typedef Field<Geometry<TM, IL>, UniOp<TOP, TL> > this_type;

	Field(TL const & l) :
			mesh(l.mesh), l_(l)
	{
	}

	inline value_type operator[](typename TM::index_type s) const
	{
		return (_OpEval(Int2Type<TOP>(), l_, s));
	}
};

template<typename TM, int IFORM, int TOP, typename TL, typename TR>
struct Field<Geometry<TM, IFORM>, BiOp<TOP, TL, TR> >
{

private:
	typename ConstReferenceTraits<TL>::type l_;
	typename ConstReferenceTraits<TR>::type r_;
	typedef Field<Geometry<TM, IFORM>, BiOp<TOP, TL, TR> > this_type;

public:
	TM const & mesh;
	Field(TL const & l, TR const & r) :
			mesh(get_mesh(l, r)), l_(l), r_(r)
	{
	}

	typedef decltype(
			_OpEval(Int2Type<TOP>(),
					std::declval<typename std::remove_reference<TL>::type const&>(),
					std::declval<typename std::remove_reference<TR>::type const&>(),
					std::declval<typename TM::index_type>()
			)

	) value_type;

	typedef Geometry<TM, IFORM> geometry_type;

	typedef typename geometry_type::template field_value_type<value_type> field_value_type;

	inline value_type operator[](typename TM::index_type s) const
	{
		return (_OpEval(Int2Type<TOP>(), l_, r_, s));
	}
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

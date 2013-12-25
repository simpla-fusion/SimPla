/*
 * field.h
 *
 *  Created on: 2013-7-19
 *      Author: salmon
 */

#ifndef FIELD_H_
#define FIELD_H_

#include "primitives.h"
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

/***
 *
 * @brief Field
 *
 * @ingroup Field Expression
 *
 */

template<typename TG, typename TValue>
struct Field
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

	typedef typename mesh_type::template Container<value_type> base_type;

	static const int NUM_OF_DIMS = mesh_type::NUM_OF_DIMS;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename geometry_type::template field_value_type<value_type> field_value_type;

	typedef typename mesh_type::template Container<value_type> container_type;

private:
	container_type data_;
	size_t num_of_eles_;
public:

	mesh_type const &mesh;

	Field(mesh_type const &pmesh)
			: mesh(pmesh), data_(nullptr), num_of_eles_(0)
	{
	}

	Field(mesh_type const &pmesh, value_type d_value)
			: mesh(pmesh), data_(nullptr), num_of_eles_(0)
	{
		*this = d_value;
	}

	/**
	 *  Copy/clone Construct only copy mesh reference, but do not copy/move data, which is designed to
	 *  initializie stl containers, such as std::vector
	 *    \code
	 *       Field<...> default_value(mesh);
	 *       std::vector<Field<...> > v(4,default_value);
	 *    \endcode
	 *  the element in v have same mesh reference.
	 *
	 * @param rhs
	 */

	Field(this_type const & rhs)
			: mesh(rhs.mesh), data_(nullptr), num_of_eles_(rhs.num_of_eles_)
	{
	}

	/// Move Construct copy mesh, and move data,
	Field(this_type &&rhs)
			: mesh(rhs.mesh), data_(rhs.data_), num_of_eles_(rhs.num_of_eles_)
	{
	}

	virtual ~Field()
	{
	}

	void swap(this_type & rhs)
	{
		base_type::swap(rhs);
		std::swap(rhs.num_of_eles_, num_of_eles_);
	}

	container_type & data()
	{
		return data_;
	}

	const container_type & data() const
	{
		return data_;
	}
	size_t size() const
	{
		return (data_ == nullptr) ? 0 : mesh.GetNumOfElements(IForm);
	}
	bool empty() const
	{
		return size() <= 0;
	}

	typedef value_type* iterator;

	iterator begin()
	{
		return &mesh.get_value(data_, 0);
	}
	iterator end()
	{
		return &mesh.get_value(data_, size());
	}

	const iterator begin() const
	{
		return &mesh.get_value(data_, 0);
	}
	const iterator end() const
	{
		return &mesh.get_value(data_, size());
	}

	inline std::vector<size_t> GetShape() const
	{
		return std::move(mesh.GetShape(IForm));
	}

	inline value_type & operator[](size_t s)
	{
		ASSERT(s < num_of_eles_)
		return mesh.get_value(data_, s);
	}
	inline value_type const & operator[](size_t s) const
	{
		return mesh.get_value(data_, s);
	}

	template<typename ... TI>
	inline value_type & get(TI ...s)
	{
		size_t ts = mesh.GetComponentIndex(IForm, s...);
		ASSERT(ts < num_of_eles_)
		return mesh.get_value(data_, ts);
	}
	template<typename ...TI>
	inline value_type const & get(TI ...s) const
	{
		return mesh.get_value(data_, mesh.GetComponentIndex(IForm, s...));
	}

	void Init()
	{
		if (data_ == nullptr)
		{
			data_ = mesh.template MakeContainer<IForm, value_type>();
			num_of_eles_ = mesh.GetNumOfElements(IForm);
		}
	}

	template<typename TD>
	void Fill(TD default_value)
	{
		Init();
		for (size_t s = 0; s < num_of_eles_; ++s)
		{
			mesh.get_value(data_, s) = default_value;
		}

	}

	inline this_type &
	operator =(this_type const & rhs)
	{
		Init();
		mesh.AssignContainer(IForm, this, rhs);
		return (*this);
	}

	template<typename TR> inline this_type &
	operator =(TR const & rhs)
	{
		Init();
		mesh.AssignContainer(IForm, this, rhs);
		return (*this);
	}

#define DECL_SELF_ASSIGN( _OP_ )                                                                   \
	template<typename TR> inline this_type &                                                       \
	operator _OP_(TR const & rhs)                                                                  \
	{   Init();                                                                                    \
		mesh.ForEach( [](value_type &l,typename FieldTraits<TR>::value_type const & r)             \
	            {	l _OP_ r;},	 this,rhs);     return (*this);}

	DECL_SELF_ASSIGN (+=)

DECL_SELF_ASSIGN	(-=)

	DECL_SELF_ASSIGN (*=)

	DECL_SELF_ASSIGN (/=)
#undef DECL_SELF_ASSIGN

	inline field_value_type operator()(coordinates_type const &x) const
	{
		return Gather(x);
	}

	inline field_value_type operator()(index_type s,coordinates_type const &pcoords) const
	{
		return Gather(s,pcoords);
	}

	inline field_value_type Gather(coordinates_type const &x) const
	{

		coordinates_type pcoords;

		index_type s = mesh.SearchCell(x, &pcoords[0]);

		return Gather(s, pcoords);

	}

	inline field_value_type Gather(index_type const & s,
			coordinates_type const &pcoords) const
	{

		field_value_type res;

		std::vector<index_type> points;

		std::vector<typename geometry_type::gather_weight_type> weights;

//		mesh.GetAffectedPoints(Int2Type<IForm>(), s, points);
//
//		mesh.CalcuateWeights(Int2Type<IForm>(), pcoords, weights);
//
//		res *= 0;

//		auto it1 = points.begin();
//		auto it2 = weights.begin();
//		for (; it1 != points.end() && it2 != weights.end(); ++it1, ++it2)
//		{
//
//			try
//			{
//				res += this->at(*it1) * (*it2);
//
//			}
//			catch (std::out_of_range const &e)
//			{
//#ifndef NDEBUG
//				WARNING
//#else
//				VERBOSE
//#endif
//				<< e.what() <<"[ idx="<< *it1<<"]";
//			}
//
//		}

		return res;

	}

	template<typename TV>
	inline void Collect(TV const & v, coordinates_type const &x)
	{
		coordinates_type pcoords;

		index_type s = mesh.SearchCell(x, &pcoords[0]);

		Collect(v, s, pcoords);

	}
	template<typename TV>
	inline void Collect(TV const & v, index_type const & s,
			coordinates_type const &pcoords, int affected_region = 1)
	{

		std::vector<index_type> points;

		std::vector<typename geometry_type::scatter_weight_type> weights;

		mesh.GetAffectedPoints(Int2Type<IForm>(), s, points);

		mesh.CalcuateWeights(Int2Type<IForm>(), pcoords, weights);

		auto it1 = points.begin();
		auto it2 = weights.begin();
		for (; it1 != points.end() && it2 != weights.end(); ++it1, ++it2)
		{
			// FIXME: this incorrect for vector field interpolation

//			try
//			{
//				this->get(*it1) += Dot(v, *it2);
//			}
//			catch (std::out_of_range const &e)
//			{
//#ifndef NDEBUG
//				WARNING
//#else
//				VERBOSE
//#endif
//				<< e.what() <<"[ idx="<< *it1<<"]";
//			}
		}

	}

	inline void Collect(std::vector<index_type> const & points,std::vector<value_type> & cache)
	{
		//FIXME: this is not thread safe, need a mutex lock

		auto it2=cache.begin();
		auto it1=points.begin();
		for(;it2!=cache.end() && it1!=points.end(); ++it1,++it2 )
		{
			this->operator[](*it1)+= *it2;
		}

	}
};

}
// namespace simpla

#endif /* FIELD_H_ */

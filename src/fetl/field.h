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
#include <utility>

#include "../utilities/log.h"
#include "../utilities/primitives.h"

#include "../parallel/parallel.h"
#include "../utilities/iterator_mapped.h"
#include "field_update_ghosts.h"

namespace simpla
{
template<typename TM, int IFORM, typename > struct Field;

/***
 *
 * @brief Field
 *
 * @ingroup Field Expression
 *
 */
template<typename TM, int IFORM, typename TContainer>
struct Field: public TContainer
{

public:

	typedef TM mesh_type;

	static constexpr unsigned int IForm = IFORM;

	static constexpr unsigned int NDIMS = mesh_type::NDIMS;

	typedef TContainer container_type;

	typedef Field<mesh_type, IForm, container_type> this_type;

	typedef typename container_type::value_type value_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::compact_index_type compact_index_type;

	typedef typename mesh_type::iterator mesh_iterator;

	typedef typename mesh_type::range_type mesh_range_type;

	typedef typename std::conditional<(IForm == VERTEX || IForm == VOLUME),  //
	        value_type, nTuple<NDIMS, value_type> >::type field_value_type;

	typedef typename mesh_type::interpolator_type interpolator_type;
	friend mesh_type;

	mesh_type const &mesh;

private:

	mesh_range_type range_;

	/***
	 *  Field
	 * @param pmesh
	 * @param args
	 */
	template<typename ...Args>
	Field(mesh_type const &pmesh, mesh_range_type const & range, Args && ... args) :
			container_type(range, std::forward<Args>(args)...), mesh(pmesh), range_(range)
	{
	}
public:

	Field(mesh_type const & mesh, value_type d = value_type()) :
			container_type(d), mesh(mesh)
	{
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
	Field(this_type const & rhs) :
			container_type(rhs), mesh(rhs.mesh), range_(rhs.range_)
	{
	}
	/// Move Construct copy mesh, and move data,
	Field(this_type &&rhs) :
			container_type(std::forward<this_type>(rhs)), mesh(rhs.mesh), range_(
			        std::forward<typename mesh_type::range_type>(rhs.range_))
	{
	}

	~Field()
	{
	}

	void swap(this_type & rhs)
	{
		ASSERT( mesh==rhs.mesh);

		container_type::swap(rhs);

		std::swap(range_, rhs.range_);

	}

	void allocate()
	{
		if (container_type::empty())
		{
			mesh.template make_field<this_type>().swap(*this);
		}
		container_type::allocate();

	}

	void initialize()
	{
		allocate();

		container_type::clear();
	}

	void clear()
	{
		initialize();
	}

	template<typename TVistor>
	void Accept(TVistor const & visitor)
	{
		visitor.Visit(this);
	}

	const mesh_range_type& GetRange() const
	{
		return range_;
	}

	void SetRange(const mesh_range_type& range)
	{
		range_ = range;
	}

	template<typename ...Args>
	int GetDataSetShape(Args &&...others) const
	{
		return mesh.GetDataSetShape(range_, std::forward<Args>(others)...);
	}

	inline value_type & at(compact_index_type s)
	{
		if (!mesh.CheckLocalMemoryBounds(s))
		OUT_RANGE_ERROR(mesh.Decompact(s));
		return get(s);
	}

	inline value_type const & at(compact_index_type s) const
	{
		if (!mesh.CheckLocalMemoryBounds(s))
		OUT_RANGE_ERROR(mesh.Decompact(s));
		return get(s);
	}

	inline value_type & operator[](compact_index_type s)
	{
		return get(s);
	}

	inline value_type const & operator[](compact_index_type s) const
	{
		return get(s);
	}

	inline value_type & get(compact_index_type s)
	{
		return container_type::get(s);
	}

	inline value_type const & get(compact_index_type s) const
	{
		return container_type::get(s);
	}

public:

	auto Select()
	DECL_RET_TYPE((make_mapped_range( *this, range_)))
	auto Select() const
	DECL_RET_TYPE((make_mapped_range( *this, range_)))

	template<typename ... Args>
	auto Select(Args &&... args)
	DECL_RET_TYPE((make_mapped_range( *this, mesh.Select( range_,std::forward<Args>(args)...))))
	template<typename ... Args>
	auto Select(Args &&... args) const
	DECL_RET_TYPE((make_mapped_range( *this, mesh.Select( range_,std::forward<Args>(args)...))))

	auto begin() DECL_RET_TYPE(simpla::begin(this->Select()))
	auto begin() const DECL_RET_TYPE(simpla::begin(this->Select()))

	auto end() DECL_RET_TYPE(simpla::end(this->Select()))
	auto end() const DECL_RET_TYPE(simpla::end(this->Select()))

	template<typename T>
	void Fill(T v)
	{
		container_type::allocate();

		ParallelForEach(range_,

				[this,v](compact_index_type s)
				{
					get_value(*this, s) = v;
				}

		);
	}
	template<typename TR>
	void assign(TR const & rhs)
	{
		container_type::allocate();

		ParallelForEach(range_,

				[this,&rhs](compact_index_type s)
				{
					get_value(*this, s) = get_value( rhs, s);
				}

		);

		UpdateGhosts(this);

	}

	this_type & operator =(this_type const & rhs)
	{
		assign(rhs);
		return *this;
	}

	this_type & operator =(value_type rhs)
	{
		Fill(rhs);
		return (*this);
	}

	template<typename TR>
	this_type & operator =(TR const & rhs)
	{
		assign(rhs);

		return (*this);
	}

	template<typename TR> inline this_type &
	operator +=(TR const & rhs)
	{
		*this = *this + rhs;
		return (*this);
	}
	template<typename TR> inline this_type &
	operator -=(TR const & rhs)
	{
		*this = *this - rhs;
		return (*this);
	}
	template<typename TR> inline this_type &
	operator *=(TR const & rhs)
	{
		*this = *this * rhs;
		return (*this);
	}
	template<typename TR> inline this_type &
	operator /=(TR const & rhs)
	{
		*this = *this / rhs;
		return (*this);
	}

	inline field_value_type operator()(coordinates_type const &x) const
	{
		return interpolator_type::Gather( *this, x);
	}
	template<typename TZ,typename TV>
	inline void Add(coordinates_type const &x, TZ const & z,TV const &v)
	{
		return interpolator_type::Scatter( this, z,v);
	}

}
;

template<typename TL>
struct is_field
{
	static const bool value = false;
};

template<typename TG, int IF, typename TL>
struct is_field<Field<TG, IF, TL>>
{
	static const bool value = true;
};

template<typename T>
struct is_field_expression
{
	static constexpr bool value = false;
};

template<typename TG, int IF, int TOP, typename TL, typename TR>
struct is_field_expression<Field<TG, IF, BiOp<TOP, TL, TR> > >
{
	static constexpr bool value = true;
};

template<typename TG, int IF, int TOP, typename TL>
struct is_field_expression<Field<TG, IF, UniOp<TOP, TL> > >
{
	static constexpr bool value = true;
};

template<typename TG, int IF, int TOP, typename TL, typename TR>
struct is_expression<Field<TG, IF, BiOp<TOP, TL, TR> > >
{
	static constexpr bool value = true;
};

template<typename TG, int IF, int TOP, typename TL>
struct is_expression<Field<TG, IF, UniOp<TOP, TL> > >
{
	static constexpr bool value = true;
};

}
// namespace simpla

#endif /* FIELD_H_ */

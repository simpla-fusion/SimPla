/*
 * @file field.h
 *
 * @date  2013-7-19
 * @author  salmon
 */

#ifndef FIELD_H_
#define FIELD_H_

#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <utility> //for move

#include "../utilities/log.h"
#include "../utilities/primitives.h"

#include "../parallel/parallel.h"
#include "../utilities/sp_iterator_mapped.h"
#include "field_update_ghosts.h"
#include "../mesh/interpolator.h"
namespace simpla
{
template<typename TM, typename > struct Field;

/**
 * \ingroup FETL
 * @class Field
 * \brief Field object
 *
 */
template<typename TDomain, typename TV>
struct Field
{

public:

	typedef TDomain domain_type;

	typedef TV value_type;

	typedef Field<domain_type, value_type> this_type;

	typedef typename domain_type::coordinates_type coordinates_type;

	typedef typename domain_type::index_type index_type;

	/**
	 *  create constructer
	 * @param pmesh
	 * @param args
	 */
	Field(domain_type const &d) :
			domain_(d)
	{
	}

	/**
	 *
	 *  \brief Copy Constructer
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
			domain_(rhs.domain_)
	{
	}

	~Field()
	{
	}

	void allocate()
	{
		domain_.template make_field<this_type>().swap(*this);
	}
	void deallocate()
	{
	}

	void erase()
	{
		deallocate();
	}

	const domain_type & domain() const
	{
		return domain_;
	}
	void domain(const domain_type & d) const
	{
		domain_=d;
		deallocate();
	}

	inline value_type & at(index_type s)
	{
		if (!domain_.check_local_memory_bounds(s))
		OUT_RANGE_ERROR(domain_.decompact(s));
		return get(domain_.hash(s));
	}

	inline value_type const & at(index_type s) const
	{
		if (!domain_.check_local_memory_bounds(s))
		OUT_RANGE_ERROR(domain_.decompact(s));

		return get(domain_.hash(s));
	}

	inline value_type & operator[](index_type s)
	{
		return get(domain_.hash(s));
	}

	inline value_type const & operator[](index_type s) const
	{
		return get(domain_.hash(s));
	}

	template<typename TR,typename TFun>
	void self_assign(TR const & rhs, TFun const & fun)
	{
		if(empty())allocate();

		parallel_for(domain_,

				[this,&rhs,&fun](domain_type const &r)
				{
					for(auto const & s:r)
					{
						(*this)[s] =fun((*this)[s], r.get_value( rhs, s));
					}
				}

		);

//		parallel_for_each(range_,
//
//				[this,&rhs](compact_index_type s)
//				{
//					get_value(*this, s) = get_value( rhs, s);
//				}
//
//		);

		update_ghosts(this);

	}

	template<typename TR>
	void assign(TR const & rhs)
	{
		allocate();

		parallel_for(domain_,

				[this,&rhs ](domain_type const &r)
				{
					for(auto const & s:r)
					{
						(*this)[s] = get_value( rhs,r.hash( s));
					}
				}

		);

//		parallel_for_each(range_,
//
//				[this,&rhs](compact_index_type s)
//				{
//					get_value(*this, s) = get_value( rhs, s);
//				}
//
//		);

		update_ghosts(this);

	}
	template<typename TR>
	void assign(Field<domain_type,TR> const & rhs)
	{
		if(empty()) allocate();

		parallel_for( domain_ & rhs.domain(),

				[this,&rhs ](domain_type const &r)
				{
					for(auto const & s:r)
					{
						(*this)[s] = rhs[s];
					}
				}

		);

//		parallel_for_each(range_,
//
//				[this,&rhs](compact_index_type s)
//				{
//					get_value(*this, s) = get_value( rhs, s);
//				}
//
//		);

		update_ghosts(this);

	}

	template<typename TR>
	this_type & operator =(TR const & rhs)
	{
		assign(rhs);
		return *this;
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

	template<typename TZ,typename TF>
	inline void scatter( TZ const & z ,TF const & f )
	{
		domain_.scatter( this,z,f);
	}
	inline auto gather(coordinates_type const &x) const
	DECL_RET_TYPE( (domain_.gather( *this, x)))

	inline auto operator()(coordinates_type const &x) const
	->decltype(((gather( x) )))
	{	return std::move((gather( x) ));}

private:
	domain_type const &domain_;

};

template<typename TL> struct is_field
{
	static const bool value = false;
};

template<typename TG, unsigned int IF, typename TL> struct is_field<
		Field<TG, IF, TL>>
{
	static const bool value = true;
};

template<typename T> struct is_field_expression
{
	static constexpr bool value = false;
};

template<typename TG, unsigned int IF, unsigned int TOP, typename TL,
		typename TR> struct is_field_expression<
		Field<TG, IF, BiOp<TOP, TL, TR> > >
{
	static constexpr bool value = true;
};

template<typename TG, unsigned int IF, unsigned int TOP, typename TL> struct is_field_expression<
		Field<TG, IF, UniOp<TOP, TL> > >
{
	static constexpr bool value = true;
};

template<typename TG, unsigned int IF, unsigned int TOP, typename TL,
		typename TR> struct is_expression<Field<TG, IF, BiOp<TOP, TL, TR> > >
{
	static constexpr bool value = true;
};

template<typename TG, unsigned int IF, unsigned int TOP, typename TL> struct is_expression<
		Field<TG, IF, UniOp<TOP, TL> > >
{
	static constexpr bool value = true;
};

}
// namespace simpla

#endif /* FIELD_H_ */

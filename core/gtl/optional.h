/**
 * @file  optional.h
 *
 * @date  2014-4-1
 *      @author  salmon
 */

#ifndef CORE_GTL_OPTIONAL_H_
#define CORE_GTL_OPTIONAL_H_

#include <boost/optional.hpp>

namespace simpla {
namespace gtl {
#ifndef NO_BOOST

template<typename T> using optional=boost::optional<T>;

#else
template<typename T> class optional
{
	bool cond_;

public:
	typedef optional<T> this_type;
	typedef T value_type;

	T value;
	optional() :
	cond_(false)
	{

	}
	optional(bool cond, value_type v) :
	value(v), cond_(cond)
	{
	}
	~optional()
	{
	}

	void SetValue(value_type && v)
	{
		value = v;
	}
	void SetTrue()
	{
		cond_ = true;
	}
	void SetFalse()
	{
		cond_ = false;
	}

	operator bool() const
	{
		return cond_;
	}
	bool operator!() const
	{
		return !cond_;
	}

	value_type & operator*()
	{
		return value;
	}
	value_type const & operator*() const
	{
		return value;
	}
	value_type * operator ->()
	{
		return &value;
	}
	value_type const* operator ->() const
	{
		return &value;
	}

};
#endif
}
}//  namespace simpla::gtl

#endif /* CORE_GTL_OPTIONAL_H_ */

/**
 * @file  optional.h
 *
 * @date  2014-4-1
 *      @author  salmon
 */

#ifndef CORE_toolbox_OPTIONAL_H_
#define CORE_toolbox_OPTIONAL_H_
#include <boost/optional.hpp>

namespace simpla
{
#ifndef NO_BOOST

template<typename T> using optional=boost::optional<T>;

#else
template<typename T> class optional
{
	bool cond_;

public:
	typedef optional<T> this_type;
	typedef T value_type;

	T m_value_;
	optional() :
	cond_(false)
	{

	}
	optional(bool cond, value_type v) :
	m_value_(v), cond_(cond)
	{
	}
	~optional()
	{
	}

	void SetValue(value_type && v)
	{
		m_value_ = v;
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
		return m_value_;
	}
	value_type const & operator*() const
	{
		return m_value_;
	}
	value_type * operator ->()
	{
		return &m_value_;
	}
	value_type const* operator ->() const
	{
		return &m_value_;
	}

};
#endif
}
 // namespace simpla

#endif /* CORE_toolbox_OPTIONAL_H_ */

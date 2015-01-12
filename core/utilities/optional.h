/*
 * optional.h
 *
 * \date  2014-4-1
 *      \author  salmon
 */

#ifndef OPTIONAL_H_
#define OPTIONAL_H_

namespace simpla
{

/** @ingroup utilities*/
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
}  // namespace simpla

#endif /* OPTIONAL_H_ */

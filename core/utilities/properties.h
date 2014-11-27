/**
 * \file properties.h
 *
 * \date    2014年7月13日  上午7:27:37 
 * \author salmon
 */

#ifndef PROPERTIES_H_
#define PROPERTIES_H_
#include <map>
#include <string>
#include "any.h"
namespace simpla
{
/**
 *   Properties Tree
 *  @todo using shared_ptr storage data
 */
class Properties: public std::map<std::string, Properties>
{

public:
	typedef std::string key_type;
	typedef Properties this_type;
	typedef std::map<key_type, this_type> base_type;
private:
	Any value_;
	static const Properties fail_safe_;
public:
	Properties()
	{
	}
	Properties(Any const &v) :
			value_(v)
	{
	}
	~Properties()
	{
	}

	template<typename T>
	this_type & operator=(T const& v)
	{
		value_ = v;
		return *this;
	}

	template<typename T>
	T & as()
	{
		return value_.template as<T>();
	}

	template<typename T>
	T as(T const & default_v) const
	{
		if (value_.empty())
		{
			return default_v;
		}
		else
		{
			return value_.template as<T>();
		}
	}

	template<typename T>
	T const & as() const
	{
		return value_.template as<T>();
	}
	inline bool empty() const // STL style
	{
		return value_.empty() && base_type::empty();
	}
	inline bool IsNull() const
	{
		return empty();
	}
	operator bool() const
	{
		return !empty();
	}

	Properties & get(std::string const & key)
	{
		if (key == "")
		{
			return *this;
		}
		else
		{
			return base_type::operator[](key);
		}
	}

	Properties const &get(std::string const & key) const
	{
		auto it = base_type::find(key);
		if (it == base_type::end())
		{
			return *this;
		}
		else
		{
			return it->second;
		}
	}

	template<typename T>
	T get(std::string const & key, T const & v) const
	{
		auto it = base_type::find(key);
		if (it != base_type::end() && it->second.value_.is<T>())
		{
			return it->second.value_.as<T>();
		}
		else
		{
			return v;
		}
	}

	template<typename T>
	void set(std::string const & key, T const & v)
	{
		get(key) = v;
	}

	inline Properties & operator[](key_type const & key)
	{
		return get(key);
	}
	inline Properties & operator[](const char key[])
	{
		return get(key);
	}

	Properties const & operator()(std::string const & key) const
	{
		return get(key);
	}
	Properties & operator()(std::string const & key)
	{
		return get(key);
	}
	template<typename T>
	void operator()(std::string const & key, T && v)
	{
		set(key, std::forward<T>(v));
	}

	Any & value()
	{
		return value_;
	}
	Any const& value() const
	{
		return value_;
	}
	inline Properties const& operator[](key_type const & key) const
	{
		return get(key);
	}
	inline Properties const& operator[](const char key[]) const
	{
		return get(key);
	}

	std::ostream & print(std::ostream & os) const;
//	template<typename T>
//	bool set(std::string const & key, T && v)
//	{
//		auto it = base_type::find(key);
//		if (it == base_type::end())
//		{
//			base_type::emplace(key, Any(v));
//		}
//		else if (it->second.value_.Is<T>())
//		{
//			it->second.value_.AnyCast<T>() = v;
//		}
//		else
//		{
//			ERROR("try to assign a value with incompatible type ");
//		}
//	}
};

std::ostream & operator<<(std::ostream & os, Properties const & prop);
}  // namespace simpla

#endif /* PROPERTIES_H_ */

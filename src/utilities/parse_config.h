/*
 * parse_config.h
 *
 *  Created on: 2012-3-6
 *      Author: salmon
 */

#ifndef PARSE_CONFIG_H_
#define PARSE_CONFIG_H_
#include <string>
namespace simpla
{

template<typename T1, typename T2 = std::nullptr_t>
class ParseConfig: public T1, public T2
{
public:
	void ParseFile(std::string const & filename)
	{
		try
		{
			T1::template ParseFile(filename);
		} catch (...)
		{
			T2::template ParseFile(filename);
		}
	}

	void ParseString(std::string const & str)
	{
		try
		{
			T1::template ParseString(str);
		} catch (...)
		{
			T2::template ParseString(str);
		}
	}

	ParseConfig operator[](std::string const & key) const
	{
		return (ParseConfig(T1(T1::template operator[](key)),
				T2(T2::template operator[](key))));
	}

	template<typename T>
	inline T Get(std::string const & key)
	{
		T res;
		Get(key, res);
		return res;
	}

	template<typename T>
	inline void Get(std::string const & key, T & res)
	{
		try
		{
			T1::Get(key, res);
		} catch (...)
		{
			T2::Get(key, res);
		}
	}

	template<typename T, typename ... Args>
	void Function(T* res, Args const & ... args) const
	{

		try
		{
			T1::template Function(res, args...);
		} catch (...)
		{
			T2::template Function(res, args...);
		}
	}
};

template<typename T1>
class ParseConfig<T1, std::nullptr_t> : public T1
{
};
}  // namespace simpla
#endif /* PARSE_CONFIG_H_ */

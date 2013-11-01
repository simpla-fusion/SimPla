/*
 * test.cpp
 *
 *  Created on: 2013年11月1日
 *      Author: salmon
 */

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>

template<typename TF> struct RWCache
{
public:
	RWCache()
	{

	}
	~RWCache()
	{

	}
};

template<typename TF> struct RWCache<TF&> : public RWCache<TF>
{
	RWCache(TF & f)
	{
		std::cout << "write cache" << std::endl;
	}
};

template<typename TF> struct RWCache<TF const&> : public RWCache<TF>
{
	RWCache(TF const& f)
	{
		std::cout << "read cache" << std::endl;
	}
};

template<typename TF> typename std::enable_if<std::is_const<TF>::value,
		RWCache<TF const &> >::type MakeCache(const TF&f)
{
	return RWCache<TF const&>(f);
}

template<typename TF> RWCache<TF &> MakeCache(TF & f)
{
	return RWCache<TF &>(f);
}

int main()
{

	double a = 5;
	double &b = a;
	double const &c = a;

	MakeCache(a);
	MakeCache(const_cast<double const&>(b));

}


/*
 * any.cpp
 *
 *  Created on: 2014年7月14日
 *      Author: salmon
 */
#include "any.h"
#include  <complex>
#include  <string>
namespace simpla
{

inline std::ostream & Any::print(std::ostream & os) const
{

	if (t_index_ == std::type_index(typeid(std::string)))
	{
		os << as<std::string>();
	}
	else if (t_index_ == std::type_index(typeid(bool)))
	{
		os << std::boolalpha << as<bool>() << std::noboolalpha;
	}
	else if (t_index_ == std::type_index(typeid(int)))
	{
		os << as<int>();
	}
	else if (t_index_ == std::type_index(typeid(long)))
	{
		os << as<long>();
	}
	else if (t_index_ == std::type_index(typeid(float)))
	{
		os << as<float>();
	}
	else if (t_index_ == std::type_index(typeid(double)))
	{
		os << as<double>();
	}
	else if (t_index_ == std::type_index(typeid(long double)))
	{
		os << as<long double>();
	}
	else if (t_index_ == std::type_index(typeid(std::complex<double>)))
	{
		os << as<std::complex<double>>();
	}
	else if (t_index_ == std::type_index(typeid(std::complex<float>)))
	{
		os << as<std::complex<float>>();
	}
	return os;
}
inline std::ostream & operator<<(std::ostream & os, Any const& v)
{
	return v.print(os);

}

}  // namespace simpla


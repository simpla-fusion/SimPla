/*
 * properties.h
 *
 *  Created on: 2012-3-6
 *      Author: salmon
 */

#ifndef PROPERTIES_H_
#define PROPERTIES_H_
#include <string>
#include <sstream>
#include <map>
#include <complex>
#include <boost/property_tree/ptree.hpp>
namespace simpla
{
typedef boost::property_tree::ptree ptree;

template<int N, typename T> struct nTuple;

template<class Ext, class Int = std::string> struct pt_trans;

void read_file(std::string const & fname, ptree & pt);

void write_file(std::string const & fname, ptree const & pt);

template<class T>
struct pt_trans<T, std::string>
{
	typedef T external_type;
	typedef std::string internal_type;

	external_type get_value(const internal_type &value) const
	{
		std::istringstream is(value);
		external_type tv;
		is >> tv;
		return tv;
	}

	internal_type put_value(const external_type &value) const
	{
		std::ostringstream os;

		os << " " << value;

		return os.str();
	}

};

template<>
struct pt_trans<std::string, std::string>
{
	typedef std::string external_type;
	typedef std::string internal_type;

	external_type get_value(const internal_type &value) const
	{
		return value;
	}

	internal_type put_value(const external_type &value) const
	{
		return value;
	}

};

template<class T>
struct pt_trans<std::complex<T>, std::string>
{
	typedef std::complex<T> external_type;
	typedef std::string internal_type;

	external_type get_value(const internal_type &value) const
	{
		std::istringstream is(value);

		T r, i;
		is >> r >> i;

		return external_type(r, i);
	}

	internal_type put_value(const external_type &value) const
	{
		std::ostringstream os;

		os << " " << value;

		return os.str();
	}

};
template<int N, class T>
struct pt_trans<nTuple<N, T>, std::string>
{
	typedef nTuple<N, T> external_type;
	typedef std::string internal_type;

	external_type get_value(const internal_type &value) const
	{
		std::istringstream is(value);
		nTuple<N, T> tv;
		for (int i = 0; i < N && is; ++i)
		{
			is >> tv[i];
		}
		return tv;
	}

	internal_type put_value(const external_type &value) const
	{
		std::ostringstream os;

		for (int i = 0; i < N; ++i)
		{
			os << " " << value[i];
		}
		return os.str();
	}

};

template<int M, int N, class T>
struct pt_trans<nTuple<M, nTuple<N, T> >, std::string>
{
	typedef nTuple<M, nTuple<N, T> > external_type;
	typedef std::string internal_type;

	external_type get_value(const internal_type &value) const
	{
		std::istringstream is(value);
		external_type tv;

		for (int i = 0; i < M; ++i)
			for (int j = 0; j < N; ++j)
			{
				is >> tv[i][j];
			}

		return tv;
	}

	internal_type put_value(const external_type &value) const
	{
		std::ostringstream os;

		for (int i = 0; i < M; ++i)
			for (int j = 0; j < N; ++j)
			{
				os << " " << value[i][j];
			}
		return os.str();
	}

};
//class ptree: public boost::property_tree::ptree
//{
//public:
//	typedef boost::property_tree::ptree BaseType;
//	typedef ptree ThisType;
//	template<typename T> T get_value() const
//	{
//		return BaseType::get_value<T>(
//				pt_trans<T, typename BaseType::data_type>());
//	}
//	template<typename T> T get(std::string const & path) const
//	{
//		return BaseType::get<T>(path,
//				pt_trans<T, typename BaseType::data_type>());
//	}
//
//	template<typename T> T get(std::string const & path, T def) const
//	{
//		return BaseType::get(path, def,
//				pt_trans<T, typename BaseType::data_type>());
//	}
//
//	std::string get(std::string const & path, const char* def) const
//	{
//		return BaseType::get(path, def);
//	}
//	const ptree &get_child(const std::string &path) const
//	{
//		return reinterpret_cast<ptree const &>(BaseType::get_child(path));
//	}
////	boost::optional<const ptree> get_child_optional(
////			const std::string &path) const
////	{
////		boost::optional<const boost::property_tree::ptree&> res =
////				BaseType::get_child_optional(path);
////
////		return reinterpret_cast<ptree const&>(*res);
////	}
//};
} // namespace simpla
#endif /* PROPERTIES_H_ */

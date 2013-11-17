/*
 * properties_ptree.h
 *
 *  Created on: 2013年11月17日
 *      Author: salmon
 */

#ifndef PROPERTIES_PTREE_H_
#define PROPERTIES_PTREE_H_

#include "properties.h"
#include <map>
#include <complex>
#include <boost/property_tree/ptree.hpp>
namespace simpla
{

template<int N, typename T> struct nTuple;

template<typename ... TOthers>
class Properties<boost::property_tree::ptree, TOthers...> : public Properties<
		TOthers...>
{
public:

	using boost::property_tree::ptree;
	typedef Properties<TOthers...> base_type;

	typedef Properties<ptree, TOthers...> this_type;

	template<typename ...Args>
	Properties(ptree const &t1, Args &... args) :
			ptree(t1), base_type(std::forward<Args>(args)...)
	{
	}

	~Properties() = default;

	void ParseFile(std::string const & filename)
	{
		int npos = filename.find_last_of('.');

		if (filename.substr(npos) == ".xml")
		{
			read_file(*this, filename);
		}

		base_type::template ParseFile(filename);
	}

	void ParseString(std::string const & str)
	{
		base_type::template ParseString(str);
	}

	this_type GetChild(std::string const & key) const
	{
		return std::move(
				this_type(ptree::get_child(key), base_type::get_child(key)));
	}

	template<typename T>
	inline T GetValue(std::string const & key, T const & default_value = T())
	{
		T res = default_value;
		try
		{
			res = ptree::get<T>(key, pt_trans<T, std::string>());
		} catch (...)
		{
			res = base_type::template GetValue<T>(key, default_value);
		}
		return std::move(res);
	}

	template<typename T>
	inline void SetValue(std::string const & key, T const & default_value)
	{

	}

	template<typename T, typename ... Args>
	void Function(T* res, Args const & ... args) const
	{
		base_type::template Function(res, args...);
	}
private:
	template<class Ext, class Int = std::string> struct pt_trans;

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

}
;

}  // namespace simpla

#endif /* PROPERTIES_PTREE_H_ */

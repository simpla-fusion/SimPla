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
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace simpla
{
template<int N, typename T> struct nTuple;
class ptree
{
public:
	ptree();
	virtual ~ptree();

	class iterator;

	class const_iterator;

	iterator begin();
	const_iterator begin() const;
	iterator end();
	const_iterator end() const;

	virtual TR1::shared_ptr<const ptree> get_child(
			std::string const & name) const;

	virtual inline void get_value(std::string const & name, bool &val) const;

	virtual inline void get_value(std::string const & name, short &val) const;

	virtual inline void get_value(std::string const & name,
			unsigned short &val) const;

	virtual inline void get_value(std::string const & name, int &val) const;

	virtual inline void get_value(std::string const & name,
			unsigned int &val) const;

	virtual inline void get_value(std::string const & name, long &val) const;

	virtual inline void get_value(std::string const & name,
			unsigned long &val) const;
	virtual inline void get_value(std::string const & name, float &val) const;

	virtual inline void get_value(std::string const & name, double &val) const;

	virtual inline void get_value(std::string const & name,
			long double &val) const;

	virtual inline void get_value(std::string const & name,
			std::string &val) const;

	virtual inline void get_value(std::string const & name,
			nTuple<THREE, short> &val) const;

	virtual inline void get_value(std::string const & name,
			nTuple<THREE, unsigned short> &val) const;

	virtual inline void get_value(std::string const & name,
			nTuple<THREE, int> &val) const;

	virtual inline void get_value(std::string const & name,
			nTuple<THREE, unsigned int> &val) const;

	virtual inline void get_value(std::string const & name,
			nTuple<THREE, long> &val) const;

	virtual inline void get_value(std::string const & name,
			nTuple<THREE, unsigned long> &val) const;

	virtual inline void get_value(std::string const & name,
			nTuple<THREE, float> &val) const;

	virtual inline void get_value(std::string const & name,
			nTuple<THREE, double> &val) const;

	virtual inline void get_value(std::string const & name,
			nTuple<THREE, long double> &val) const;

	template<typename T>
	inline T get(std::string const &name, T const & value) const
	{
		T res;
		try
		{
			get_value(name, res);
		} catch (...)
		{
			res = value;
		}
		return res;
	}

	template<typename T>
	inline T get(std::string const &name)
	{
		T res;
		get_value(name, res);
		return res;
	}
};

#define DEF_GET_VALUE                                                            \
	                                                                             \
virtual inline void get_value(std::string const & name, bool &val) const         \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name, short &val) const        \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name,                          \
		unsigned short &val) const                                               \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name, int &val) const          \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name,                          \
		unsigned int &val) const                                                 \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name, long &val) const         \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name,                          \
		unsigned long &val) const                                                \
{	val = get(name, val); }                                                      \
virtual inline void get_value(std::string const & name, float &val) const        \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name, double &val) const       \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name,                          \
		long double &val) const                                                  \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name,                          \
		std::string &val) const                                                  \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name,                          \
		nTuple<THREE, short> &val) const                                         \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name,                          \
		nTuple<THREE, unsigned short> &val) const                                \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name,                          \
		nTuple<THREE, int> &val) const                                           \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name,                          \
		nTuple<THREE, unsigned int> &val) const                                  \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name,                          \
		nTuple<THREE, long> &val) const                                          \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name,                          \
		nTuple<THREE, unsigned long> &val) const                                 \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name,                          \
		nTuple<THREE, float> &val) const                                         \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name,                          \
		nTuple<THREE, double> &val) const                                        \
{	val = get(name, val); }                                                      \
                                                                                 \
virtual inline void get_value(std::string const & name,                          \
		nTuple<THREE, long double> &val) const                                   \
{	val = get(name, val); }                                                      \


class PropertyTree: public ptree
{
public:

	PropertyTree()
	{
	}
	PropertyTree(boost::property_tree::ptree _pt) :
			pt(_pt)
	{

	}
	~PropertyTree()
	{

	}
	void read_file(std::string const & fname);

	void write_file(std::string const & fname);

	template<typename T>
	void put(std::string const &name, T const &v)
	{
		pt.put(name, v);
	}
	const PropertyTree get_child(std::string const & name) const
	{
		return PropertyTree(pt.get_child(name));
	}

	DEF_GET_VALUE

	template<typename T>
	inline T get(std::string const &name, T const & value) const
	{
		T res;
		try
		{
			res = pt.get_value(name, value);
		} catch (...)
		{
			res = value;
		}
		return res;
	}

	template<int N, typename T>
	inline nTuple<N, T> get(std::string const &name,
			nTuple<N, T> const & value) const
	{
		T res;
		try
		{
			res = pt.get<nTuple<N, T> >(name,
					pt_trans<nTuple<N, T>, std::string>());
		} catch (...)
		{
			res = value;
		}
		return res;
	}

private:
	boost::property_tree::ptree pt;

	template<class Ext, class Int = std::string> struct pt_trans;

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
};

} // namespace simpla

#include "detail/properties_impl.h"
#endif /* PROPERTIES_H_ */

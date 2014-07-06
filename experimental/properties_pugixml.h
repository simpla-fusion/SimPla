/*
 * properties_pugixml.h
 *
 * \date  2013-11-18
 *      \author  salmon
 */

#ifndef PROPERTIES_PUGIXML_H_
#define PROPERTIES_PUGIXML_H_
#include "properties.h"
#include "third_part/pugixml/src/pugixml.hpp"
namespace simpla
{

template<int N, typename T> struct nTuple;

template<typename ... TOthers>
class Properties<pugi::xml_node, TOthers...> : public Properties<TOthers...>
{
public:

	pugi::xml_node node_;

	typedef Properties<TOthers...> base_type;

	typedef Properties<pugi::xml_node, TOthers...> this_type;

	template<typename ...Args>
	Properties(pugi::xml_node const &n, Args &... args) :
			nide_(n), base_type(std::forward<Args>(args)...)
	{
	}

	~Properties() = default;

	this_type Child(std::string const & key)
	{
		return std::move(this_type(node_.child(key.c_str())));
	}



	template<typename T>
	inline T Get(std::string const & key, T const & default_value = T())
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
	inline void Set(std::string const & key, T const & default_value)
	{

	}

	template<typename T, typename ... Args>
	void Function(T* res, Args const & ... args) const
	{
		base_type::template Function(res, args...);
	}
};

}  // namespace simpla

#endif /* PROPERTIES_PUGIXML_H_ */

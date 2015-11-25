/**
 * @file parser_configure.h
 *
 * @date    2014-9-18  AM8:20:15
 * @author salmon
 */

#ifndef PARSER_CONFIGURE_H_
#define PARSER_CONFIGURE_H_

#include <functional>
#include <ostream> //for print
#include "log.h"
namespace simpla
{
/**
 * @ingroup configuration
 */
template<typename TDict>
class ParserConfigure
{
public:
	typedef TDict dict_type;

	typedef std::function<void(dict_type const &)> callback_type;

	ParserConfigure();

	~ParserConfigure();

	std::ostream & print(std::ostream & os);

	template<typename TFun>
	void register_callback(std::string const & key, TFun const & fun, std::string const& desc = "");

	void register_module(std::string const & key, std::string const& description = "");

	void parser(dict_type const &dict, std::string const & prefix = "");
private:

	std::map<std::string, callback_type> callbacks_;
	std::map<std::string, std::string> description_;
};

template<typename TDict>
ParserConfigure<TDict>::ParserConfigure()
{
}

template<typename TDict>
ParserConfigure<TDict>::~ParserConfigure()
{
}
template<typename TDict>
std::ostream &
ParserConfigure<TDict>::print(std::ostream & os)
{
	for (auto const & item : description_)
	{
		os << "\t" << item.first << "\t" << item.second << std::endl;
	}
}
template<typename TDict>
template<typename TFun>
void ParserConfigure<TDict>::register_callback(std::string const & key, TFun const & fun, std::string const& desc)
{
	auto it = callbacks_.find(key);

	if (it != callbacks_.end())
	{
		THROW_EXCEPTION_RUNTIME_ERROR(key + "is registered!");
	}
	else
	{
		callbacks_.emplace(key, fun);

		if (desc != "")
		{
			description_.emplace(key, desc);
		}
	}
}
template<typename TDict>
void ParserConfigure<TDict>::register_module(std::string const & prefix, std::string const& desc)
{
	register_callback(prefix,

	[prefix,this](dict_type const& dict)
	{
		this->parser(dict,prefix)
	}, desc);

}

template<typename TDict>
void ParserConfigure<TDict>::parser(dict_type const &dict, std::string const & prefix)
{

	for (auto const & item : dict)
	{
		auto it = callbacks_.find(prefix + "." + item.first.as<std::string>(""));
		if (it != callbacks_.end())
		{
			it->second(item.second);
		}

	}
}
}  // namespace simpla

#endif /* PARSER_CONFIGURE_H_ */

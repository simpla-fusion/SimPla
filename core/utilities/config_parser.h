/*
 * configure.h
 *
 *  Created on: 2014年11月24日
 *      Author: salmon
 */

#ifndef CORE_UTILITIES_CONFIG_PARSER_H_
#define CORE_UTILITIES_CONFIG_PARSER_H_

namespace simpla
{

struct ConfigParser: public LuaObject
{
	typedef LuaObject dict_type;

	ConfigParser();
	ConfigParser(int, char**);
	~ConfigParser();

	void init(int argc, char** argv);

	template<typename T, typename ... Others>
	void convert_cmdline_to_option(std::string const & key,
			Others const & ... alias)
	{

		simpla::parse_cmd_line(argc_, argv_,

		[&](std::string const & opt,
				std::string const & value)->int
		{
			if(find_same(opt,alias...) )
			{
				dict_type::set(key,ToValue<T>(value));
				return TERMINATE;
			}

			return CONTINUE;

		}

		);

	}

private:
	int argc_;
	char** argv_;
};

}  // namespace simpla

#endif /* CORE_UTILITIES_CONFIG_PARSER_H_ */

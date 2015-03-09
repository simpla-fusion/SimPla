/**
 * @file  config_parser.h
 *
 *  Created on: 2014年11月24日
 *      Author: salmon
 */

#ifndef CORE_UTILITIES_CONFIG_PARSER_H_
#define CORE_UTILITIES_CONFIG_PARSER_H_

#include <map>
#include <string>

#include "lua_object.h"
#include "misc_utilities.h"
namespace simpla
{

/**
 * @ingroup utilities
 */
struct ConfigParser
{

	ConfigParser();

	~ConfigParser();

	void init(int argc, char** argv);

	struct DictObject //: public LuaObject
	{
		DictObject()
				: m_value_("")
		{

		}
//		DictObject(DictObject const & other)
//				: LuaObject(other), m_value_(other.m_value_)
//		{
//		}
//		DictObject(LuaObject const & lua_obj)
//				: LuaObject(lua_obj), m_value_("")
//		{
//			CHECK("LLL");
//
//		}
		DictObject(std::string const & value)
				: /*LuaObject(),*/m_value_(value)
		{
			CHECK(value);
		}
		~DictObject()
		{
			CHECK(m_value_);
		}
		void swap(DictObject & other)
		{
//			CHECK("swap");
////			LuaObject::swap(other);
//			std::swap(m_value_, other.m_value_);
		}

		DictObject & operator=(DictObject const & other)
		{
//			CHECK("===");
//			DictObject(other).swap(*this);
			return *this;
		}

//		template<typename T>
//		T as() const
//		{
//			CHECK("as");
////			if (m_value_ != "")
////			{
//			return std::move(string_to_value<T>(m_value_));
////			}
////			else
////			{
//////				if (LuaObject::IsNull())
//////				{
//////					RUNTIME_ERROR("undefined lua object!");
//////				}
////				return std::move(LuaObject::template as<T>());
////			}
//		}
		template<typename T>
		T as(T const & default_value) const
		{
			if (m_value_ != "")
			{
				return std::move(string_to_value<T>(m_value_));
			}
//			else
//			{
//				if (LuaObject::IsNull())
//				{
//					RUNTIME_ERROR("undefined lua object!");
//				}
//				return std::move(LuaObject::template as<T>(default_value));
//			}
			return default_value;
		}

		template<typename T>
		void as(T* v) const
		{
//			CHECK(*v);
			*v = as<T>(*v);
		}
	private:
		std::string m_value_;
	};

	DictObject operator[](std::string const & key) const
	{

		auto it = m_kv_map_.find(key);
		if (it != m_kv_map_.end())
		{
			return std::move(DictObject(it->second));
		}
//		else if (!m_lua_object_.IsNull())
//		{
//			return std::move(DictObject(m_lua_object_[key]));
//		}
		else
		{
			return std::move(DictObject());
		}
	}

//	void parse_cmd_line(int argc, char** argv);

private:

//	LuaObject m_lua_object_;
	std::map<std::string, std::string> m_kv_map_;
}
;

}  // namespace simpla

#endif /* CORE_UTILITIES_CONFIG_PARSER_H_ */

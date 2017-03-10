/**
 * @file config_parser.cpp
 *
 *  Created on: 2014-11-24
 *      Author: salmon
 */

#include "ConfigParser.h"
#include "Log.h"

namespace simpla {
namespace toolbox {

ConfigParser::ConfigParser() {}

ConfigParser::~ConfigParser() {}

void ConfigParser::parse(std::string const &lua_file, std::string const &lua_prologue,
                         std::string const &lua_epilogue) {
    m_lua_object_.init();
    m_lua_object_.parse_string(lua_prologue);
    m_lua_object_.parse_file(lua_file, "");
    m_lua_object_.parse_string(lua_epilogue);
}

void ConfigParser::add(std::string const &k, std::string const &v) { m_kv_map_[k] = v; }
}
}  // namespace simpla

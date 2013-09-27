/*
 * lua_script.h
 *
 *  Created on: 2012-11-6
 *      Author: salmon
 */

#ifndef LUA_SCRIPT_H_
#define LUA_SCRIPT_H_

#include "include/simpla_defs.h"

#include <lua.hpp>

#include "utilities/properties.h"

namespace simpla
{
namespace field_fun
{
template<typename TV>
class LuaScript
{
public:
	lua_State * lua_s;
	std::string funname;

	LuaScript(PTree const &pt) :
			lua_s(luaL_newstate()), funname(pt.get("Script.<xmlattr>.Name", "Fun"))
	{
		luaL_openlibs(lua_s);
		luaL_dostring(lua_s, pt.get<std::string>("Script").c_str());
	}
	~LuaScript()
	{

	}
	template<typename TE>
	TV operator()(nTuple<THREE, TE> x, Real t)
	{
		lua_getfield(lua_s, LUA_GLOBALSINDEX, funname.c_str());

		lua_pushnumber(lua_s, x[0]);
		lua_pushnumber(lua_s, x[1]);
		lua_pushnumber(lua_s, x[2]);
		lua_pushnumber(lua_s, t);
		lua_call(lua_s, 4, 1);

		TV res;
		toValue_(lua_gettop(lua_s), &res);
		lua_pop(lua_s, 1);
		return res;
	}
private:

	inline void toValue_(int idx, double *res)
	{
		*res = lua_tonumber(lua_s, idx);
	}
	inline void toValue_(int idx, int *res)
	{
		*res = lua_tointeger(lua_s, idx);
	}
	inline void toValue_(int idx, bool *res)
	{
		*res = lua_toboolean(lua_s, idx);
	}
	inline void toValue_(int idx, std::string &res)
	{
		res = lua_tostring(lua_s, idx);
	}
	template<typename T>
	inline void toValue_(int idx, std::complex<T> *res)
	{
		nTuple<TWO, T> r;
		toValue_(idx, &r);
		(*res) = std::complex<T>(r[0], r[1]);
	}
	template<typename T, int N> inline
	void toValue_(int idx, nTuple<N, T> * res)
	{
		if (lua_istable(lua_s, idx))
		{
			size_t fnum = lua_objlen(lua_s, idx);
			if (fnum > 0)
			{
				for (size_t s = 0; s < N; ++s)
				{
					lua_rawgeti(lua_s, idx, s % fnum + 1);
					toValue_(-1, &(*res)[s]);
					lua_pop(lua_s, 1);
				}
			}
		}
	}

}
;
} // namespace field_op
}  // namespace simpla

#endif /* LUA_SCRIPT_H_ */

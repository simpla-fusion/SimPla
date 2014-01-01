/*
 * fieldsolver.h
 *
 *  Created on: 2013年12月18日
 *      Author: salmon
 */

#ifndef FIELDSOLVER_H_
#define FIELDSOLVER_H_

#include <iostream>
#include <string>

#include "../utilities/log.h"
#include "../fetl/fetl.h"

namespace simpla
{
class LuaObject;

template<typename TM>
class FieldSolver
{
public:

	typedef TM mesh_type;

	DEFINE_FIELDS(mesh_type)

	FieldSolver()
	{

	}
	virtual ~FieldSolver()
	{

	}

	virtual bool empty()
	{
		return true;
	}

	virtual void Deserialize(LuaObject const&cfg);
	virtual std::ostream & Serialize(std::ostream & os) const;
	virtual void DumpData() const;

	virtual std::string GetTypeAsString() const
	{
		return "Unknown";
	}

	void NextTimeStepE(double dt, Form<1> const&E, Form<2> const&B, Form<1> *dE)
	{
		NOTHING_TODO;
	}
	void NextTimeStepB(double dt, Form<1> const&E, Form<2> const&B, Form<2> *dB)
	{
		NOTHING_TODO;
	}

	void NextTimeStepE(double dt, VectorForm<0> const&E, VectorForm<0> const&B, VectorForm<0> *dE)
	{
		NOTHING_TODO;
	}
	void NextTimeStepB(double dt, VectorForm<0> const&E, VectorForm<0> const&B, VectorForm<0> *dB)
	{
		NOTHING_TODO;
	}

};

template<typename TM>
inline std::ostream & operator<<(std::ostream & os, FieldSolver<TM> const &self)
{
	return self.Serialize(os);
}

}  // namespace simpla

#endif /* FIELDSOLVER_H_ */

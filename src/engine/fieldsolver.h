/*
 * fieldsolver.h
 *
 *  Created on: 2013年12月18日
 *      Author: salmon
 */

#ifndef FIELDSOLVER_H_
#define FIELDSOLVER_H_

namespace simpla
{

class SolverBase
{
public:
	SolverBase()
	{

	}
	virtual ~SolverBase()
	{

	}

	virtual bool IsEmpty()
	{
		return true;
	}

	virtual void Deserialize(LuaObject const&cfg);
	virtual std::ostream & Serialize(std::ostream & os) const;
	virtual void DumpData() const;

	virtual std::string GetTypeAsString() const
	{
		return "Unknonwn";
	}

	template<typename ...Args>
	void NextTimeStep(double dt, Args ...B);

};

}  // namespace simpla

#endif /* FIELDSOLVER_H_ */

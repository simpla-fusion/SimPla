/*
 * Domain.h
 *
 *  Created on: 2012-10-9
 *      Author: salmon
 */

#ifndef DOMAIN_H_
#define DOMAIN_H_

#include "include/simpla_defs.h"
#include "basecontext.h"


namespace simpla
{

template<typename TG>
class Context: public BaseContext
{
public:
	typedef Context<TG> ThisType;

	typedef typename TG::ValueType ValueType;

	typedef TG Grid;

	typedef TR1::shared_ptr<ThisType> Holder;

	Grid grid;

	template<typename PT>
	Context(const PT & pt) :
			BaseContext(pt), grid(pt.get_child("Grid"))
	{

		LoadModules(pt);
	}

	~Context()
	{
	}

	template<typename PT>
	static TR1::shared_ptr<ThisType> Create(PT const & pt);

	template<typename PT> void LoadModules(PT const & pt);

	virtual std::string Summary() const;

private:
	Grid const * getGridPtr() const
	{
		return &grid;
	}
	Context(ThisType const &);
	Context & operator=(ThisType const &);

}
;

}
// namespace simpla

#endif /* DOMAIN_H_ */

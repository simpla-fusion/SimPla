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
#include "fetl/fetl.h"
#include "utilities/properties.h"
namespace simpla
{

template<typename TG>
class Context: public BaseContext
{
public:
	DEFINE_FIELDS(typename TG::ValueType, TG)

	typedef Context<TG> ThisType;

	typedef typename TG::ValueType ValueType;

	typedef TR1::shared_ptr<ThisType> Holder;

	std::map<std::string, TR1::function<TR1::shared_ptr<Object>(void)> > objFactory;

	Grid grid;

	Context(const ptree & pt);

	virtual ~Context();

	virtual std::string Summary() const;

	template<typename TOBJ> TR1::shared_ptr<TOBJ> CreateObject();

	template<typename TOBJ>
	TR1::shared_ptr<TOBJ> GetObject(std::string const & name = "");

private:

	Context(ThisType const &);

	Context & operator=(ThisType const &);

}
;

}
// namespace simpla

#endif /* DOMAIN_H_ */

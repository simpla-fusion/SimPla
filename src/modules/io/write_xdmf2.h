/*
 * write_xdmf2.h
 *
 *  Created on: 2012-10-27
 *      Author: salmon
 */

#ifndef WRITE_XDMF2_H_
#define WRITE_XDMF2_H_

#include "include/simpla_defs.h"
#include <Xdmf.h>

#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include "engine/basecontext.h"
#include "engine/object.h"
#include "engine/modules.h"

namespace simpla
{
namespace io
{
template<typename TG>
class WriteXDMF2: public Module
{
	enum
	{
		MAX_XDMF_NDIMS = 10
	};
public:

	typedef TG Grid;
	typedef WriteXDMF2<TG> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	template<typename PT> WriteXDMF2(BaseContext const & d, const PT & pt);

	virtual ~WriteXDMF2();

	virtual void Initialize();

	virtual void Eval();
private:
	BaseContext const & ctx;
	Grid const & grid;

	size_t step;

	std::list<std::string> obj_list_;

	std::string path_;

};
template<typename TG>
template<typename PT>
WriteXDMF2<TG>::WriteXDMF2(BaseContext const & d, const PT & pt) :
		ctx(d),

		grid(d.Grid<TG>()),

		step(pt.get("step", 1)),

		path_(pt.template get("<xmlattr>.path", "Untitled"))
{

	BOOST_FOREACH(const typename PT::value_type &v, pt)
	{
		std::string id = v.second.template get_value<std::string>();
		boost::algorithm::trim(id);
		obj_list_.push_back(id);
	}

	Initialize();
}
}  // namespace io

}  // namespace simpla

#endif /* WRITE_XDMF2_H_ */

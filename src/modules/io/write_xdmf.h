/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id$*/
#ifndef SRC_IO_WRITE_XDMF_H_
#define SRC_IO_WRITE_XDMF_H_

#include "include/simpla_defs.h"

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
class WriteXDMF: public Module
{
public:

	typedef TG Grid;
	typedef WriteXDMF<TG> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	template<typename PT> WriteXDMF(BaseContext const & d, const PT & pt);

	virtual ~WriteXDMF();

	virtual void Initialize();

	virtual void Eval();
private:
	BaseContext const & ctx;
	Grid const & grid;

	size_t step;

	std::list<std::string> obj_list_;

	std::string file_template;
	std::string attrPlaceHolder;

	std::string path_;
};

template<typename TG>
template<typename PT>
WriteXDMF<TG>::WriteXDMF(BaseContext const & d, const PT & pt) :
		ctx(d),

		grid(d.Grid<TG>()),

		step(pt.get("step", 1)),

		file_template(""),

		attrPlaceHolder("<!-- Add Attribute Here -->"),

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
} // namespace io
} // namespace simpla
#endif  // SRC_IO_WRITE_XDMF_H_

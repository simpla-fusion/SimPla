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
#include "utilities/properties.h"

namespace simpla
{

namespace io
{
template<typename TG>
class WriteXDMF
{
	enum
	{
		MAX_XDMF_NDIMS = 10
	};
public:

	typedef TG Grid;
	typedef WriteXDMF<TG> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	WriteXDMF(BaseContext const & d, const ptree & pt);

	virtual ~WriteXDMF();

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

} // namespace io
} // namespace simpla
#endif  // SRC_IO_WRITE_XDMF_H_

/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id$*/
#ifndef SRC_IO_WRITE_XDMF_H_
#define SRC_IO_WRITE_XDMF_H_

#include "include/simpla_defs.h"
#include "engine/object.h"
#include "engine/context.h"
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
	typedef WriteXDMF<Grid> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	WriteXDMF(Context<TG> const & d, const ptree & properties);

	virtual ~WriteXDMF()
	{
	}

	virtual void Eval();
private:
	Context<Grid> const & ctx;
	Grid const & grid;

	size_t step;

	std::list<std::string> obj_list_;

	std::string file_template;
	std::string attrPlaceHolder;

	std::string dir_path_;
};
} // namespace io
} // namespace simpla
#endif  // SRC_IO_WRITE_XDMF_H_

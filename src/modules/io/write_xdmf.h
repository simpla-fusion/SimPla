/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id$*/
#ifndef SRC_IO_WRITE_XDMF_H_
#define SRC_IO_WRITE_XDMF_H_

#include "include/simpla_defs.h"

#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include "engine/context.h"
#include "engine/arrayobject.h"
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

	WriteXDMF(Context<TG> const & d, const ptree & pt);

	virtual ~WriteXDMF();

	static TR1::function<void()> Create(Context<TG> * d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(*d, pt)));
	}
	virtual void Eval();
private:
	Context<TG> const & ctx;
	Grid const & grid;

	size_t stride_;

	std::list<std::string> obj_list_;

	std::string file_template;
	std::string attrPlaceHolder;

};

} // namespace io
} // namespace simpla
#endif  // SRC_IO_WRITE_XDMF_H_

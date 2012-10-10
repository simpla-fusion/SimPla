/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id$*/
#ifndef SRC_IO_WRITE_XDMF_H_
#define SRC_IO_WRITE_XDMF_H_

#include "include/simpla_defs.h"
#include "engine/object.h"
#include "engine/context.h"
#include "grid/uniform_rect.h"
namespace simpla
{
namespace io
{
void WriteXDMF(std::list<Object::Holder> const & objs, std::string const & dir_path, fetl::UniformRectGrid const & grid,
		 TR1::function<Real()> counter);
} // namespace io
} // namespace simpla
#include "io/write_xdmf.impl.h"
#endif  // SRC_IO_WRITE_XDMF_H_

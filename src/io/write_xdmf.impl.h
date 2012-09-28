/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * IO/WriterXDMF.cpp
 *
 *  Created on: 2010-12-3
 *      Author: salmon
 */
#include "write_xdmf.h"
#include <algorithm>
#include <string>
#include <list>
#include <H5Cpp.h>
#include <hdf5_hl.h>
#include <fstream>
#include <sstream>

#include "grid/uniform_rect.h"
#include "io/write_hdf5.h"

namespace simpla
{
namespace io
{
} // namespace IO
} // namespace simpla

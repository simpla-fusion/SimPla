/*
 * write_silo.cpp
 *
 *  Created on: 2011-3-30
 *      Author: salmon
 */

#include "write_silo.h"
#include "log.h"
namespace IO {
WriteSilo::WriteSilo(Context * ctx, SizeType step, const std::string & path,
		std::list<std::string> const & rec_list) :
	ctx_(ctx), step_(step), records_(rec_list) {
	silo_ = DBCreate((path+".silo").c_str(),
			0, DB_LOCAL, "", DB_HDF5);
}

WriteSilo::~WriteSilo() {
	DBClose(silo_);
}

void WriteSilo::wirteField() {
	ERROR("UNIMPLEMENT!");
}

} //namespace IO

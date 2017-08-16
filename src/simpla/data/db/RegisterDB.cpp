//
// Created by salmon on 17-8-15.
//
#include "DataBaseHDF5.h"
#include "DataBaseLua.h"
#include "DataBaseMemory.h"
#include "DataBaseStdIO.h"

namespace simpla {
namespace data {
int DataBase::s_num_of_pre_registered_ = DataBaseMemory::_is_registered +  //
                                         DataBaseHDF5::_is_registered +    //
                                         DataBaseLua::_is_registered +     //
                                         DataBaseStdIO::_is_registered;
;
}  // namespace data
}  // namespace simpla
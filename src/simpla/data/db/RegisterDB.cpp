//
// Created by salmon on 17-8-15.
//
#include "DataBaseHDF5.h"
#include "DataBaseLua.h"
#include "DataBaseMemory.h"
namespace simpla {
namespace data {
int DataBase::s_num_of_pre_registered_ = DataBaseMemory::_is_registered +  //
                                         DataBaseHDF5::_is_registered +    //
                                         DataBaseLua::_is_registered;
}  // namespace data
}  // namespace simpla
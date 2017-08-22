//
// Created by salmon on 17-8-15.
//
#include "../../../../experiment/DataBase.h"
//#include "DataBaseHDF5.h"
//#include "DataBaseLua.h"
//#include "DataBaseMDS.h"
//#include "DataBaseXDMF.h"
#include "DataNodeMemory.h"
namespace simpla {
namespace data {
int DataNode::s_num_of_pre_registered_ = DataNodeMemory::_is_registered;  //
//                                         DataBaseHDF5::_is_registered +    //
//                                         DataBaseLua::_is_registered +     //
//                                         DataBaseHDF5::_is_registered;
;
}  // namespace data
}  // namespace simpla
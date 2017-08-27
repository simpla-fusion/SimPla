//
// Created by salmon on 17-8-15.
//
#include "../DataNode.h"
#include "DataNodeHDF5.h"
#include "DataNodeIMAS.h"
#include "DataNodeLua.h"
#include "DataNodeMemory.h"
//#include "DataBaseMDS.h"
//#include "DataNodeXDMF.h"
namespace simpla {
namespace data {
int DataNode::s_num_of_pre_registered_ = DataNodeMemory::_is_registered +  //
                                         DataNodeLua::_is_registered +     //
                                         DataNodeHDF5::_is_registered +    //
                                         DataNodeIMAS::_is_registered      //
                                                                           // ;
    ;
}  // namespace data
}  // namespace simpla
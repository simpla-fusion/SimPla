//
// Created by salmon on 17-3-10.
//
#include "../DataBackendFactory.h"
#include "DataBackendHDF5.h"
#include "DataBackendLua.h"
#include "DataBackendSAMRAI.h"

namespace simpla {
namespace data {
void DataBackendFactory::RegisterDefault() {
    Register<DataBackendHDF5>(DataBackendHDF5::ext);
    Register<DataBackendLua>(DataBackendLua::ext);
    Register<DataBackendSAMRAI>(DataBackendSAMRAI::ext);
};
}  // namespace data {
}  // namespace simpla {
//
// Created by salmon on 17-3-10.
//
#include "../DataBackend.h"
#include "../DataBackendMemory.h"
#include "DataBackendHDF5.h"
#include "DataBackendLua.h"
#include "DataBackendSAMRAI.h"

namespace simpla {
namespace data {
void DataBackendFactory::RegisterDefault() {
    Register<DataBackendSAMRAI>((DataBackendSAMRAI::scheme_tag));
    Register<DataBackendLua>((DataBackendLua::scheme_tag));
    Register<DataBackendHDF5>((DataBackendHDF5::scheme_tag));
};
}  // namespace data {
}  // namespace simpla {
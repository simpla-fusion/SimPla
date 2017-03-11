//
// Created by salmon on 17-3-9.
//

#include "DataBackend.h"
#include <iomanip>
#include <string>

#include <simpla/toolbox/ParserURI.h>
#include "DataBackendMemory.h"
#include "DataEntity.h"
#include "DataTable.h"

namespace simpla {
namespace data {
DataBackendFactory::DataBackendFactory() : base_type() { RegisterDefault(); };
DataBackendFactory::~DataBackendFactory(){};
DataBackend *DataBackendFactory::Create(std::string const &scheme) { return base_type::Create(scheme); }
std::ostream &DataBackendFactory::Print(std::ostream &os, int indent) const {
    os << std::setw(indent) << " "
       << "{";
    for (auto const &item : *this) { os << item.first << ", "; }
    os << "}";
    return os;
};

DataBackend *DataBackend::Create(std::string const &scheme) {
    DataBackend *res = GLOBAL_DATA_BACKEND_FACTORY.Create(scheme);
    if (res == nullptr) { res = new DataBackendMemory; }
}

}  // namespace data {
}  // namespace simpla {
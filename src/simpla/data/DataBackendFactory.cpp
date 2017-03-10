//
// Created by salmon on 17-3-6.
//
#include "DataBackendFactory.h"
#include <iomanip>
#include "DataBackend.h"
#include "DataBackendMemory.h"


namespace simpla {
namespace data {
DataBackendFactory::DataBackendFactory() : base_type() { RegisterDefault(); };
DataBackendFactory::~DataBackendFactory(){};

std::ostream &DataBackendFactory::Print(std::ostream &os, int indent) const {
    os << std::setw(indent) << " "
       << "{";
    for (auto const &item : *this) { os << item.first << ", "; }
    os << "}";
    return os;
};
DataBackend *DataBackendFactory::Create(std::string const &uri, std::string const &args) {
    std::string ext = "";
    DataBackend *res = nullptr;
    size_type pos = uri.find_last_of('.');
    if (pos != std::string::npos) { ext = uri.substr(pos + 1); }
    res = base_type::Create(ext, uri, args);
    if (res == nullptr) { res = new DataBackendMemory(); }
    return res;
}

}  // namespace data{
}  // namespace simpla{
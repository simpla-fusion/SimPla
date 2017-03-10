//
// Created by salmon on 17-3-6.
//
#include "DataBackendFactory.h"
#include <iomanip>
#include "../DataBackend.h"

#include "DataBackendMemory.h"

namespace simpla {
namespace data {

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
//
// bool DataBackendFactory::Unregister(std::string const &ext) { return m_factory_.erase(ext) > 0; }
// DataBackendFactory::create_function_type DataBackendFactory::Find(std::string const &ext) const {
//    auto it = m_factory_.find(ext);
//    return it == m_factory_.end() ? create_function_type() : it->second;
//}

// std::unique_ptr<DataBackend> CreateDataBackend(std::string const &url, std::string const &param) {
//    if (url == "") { return std::make_unique<DataBackendMemory>(); }
//
//    DataBackend *res = nullptr;
//    std::string ext = "";
//    size_type pos = url.find_last_of('.');
//    if (pos != std::string::npos) { ext = url.substr(pos + 1); }
//    if (ext == "lua") {
//        return std::make_unique<DataBackendLua>(url, param);
//    } else if (ext == "h5") {
//        return std::make_unique<DataBackendHDF5>(url, param);
//    } else if (ext == "samrai") {
//        return std::make_unique<DataBackendSAMRAI>(url, param);
//    } else {
//        return std::make_unique<DataBackendMemory>(url, param);
//    }
//};
}  // namespace data{
}  // namespace simpla{
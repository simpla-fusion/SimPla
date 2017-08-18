//
// Created by salmon on 16-6-6.
//
#include "DataEntity.h"
#include "DataBlock.h"
#include "DataTable.h"
#include "db/DataBaseStdIO.h"

namespace simpla {
namespace data {

// std::ostream& DataEntity::Serialize(std::ostream& os, int indent) const {
//    if (dynamic_cast<DataBlock const*>(this) != nullptr) {
//        os << "<Block>";
//    } else if (dynamic_cast<DataTable const*>(this) != nullptr) {
//        os << "<Table>";
//    } else if (dynamic_cast<DataArray const*>(this) != nullptr) {
//        os << "<Array>";
//    } else {
//        os << "<LightData>";
//    }
//    return os;
//};
std::ostream &operator<<(std::ostream &os, DataEntity const &v) {
//    auto out = DataBaseStdIO::New();
//    out->SetStream(os);
//    out->Set("", const_cast<DataEntity &>(v).shared_from_this());
    return os;
}

std::istream &operator<<(std::istream &is, DataEntity &v) {
//    auto in = DataBaseStdIO::New();
//    in->SetStream(is);
//    auto db = DataTable::New(in);
//    dynamic_cast<DataTable &>(v).Set(db);
    return is;
}

// DataEntity DataEntity::operator[](std::string const& url) {
//    if (m_holder_ == nullptr) {
//        m_holder_ = new DataTable;
//    } else if (!m_holder_->isA(typeid(DataTable))) {
//        RUNTIME_ERROR << "Data entity is not indexable!" << std::endl;
//    }
//    return static_cast<DataTable*>(m_holder_)->Insert(url).first;
//};
// DataEntity DataEntity::operator[](std::string const& url) const {
//    if (!m_holder_->isA(typeid(DataTable))) { RUNTIME_ERROR << "Data entity is not indexable!" << std::endl; }
//    return (*static_cast<DataTable const*>(m_holder_))[url];
//}
}  // namespace get_mesh{
}  // namespace simpla{
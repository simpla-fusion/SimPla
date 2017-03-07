//
// Created by salmon on 16-6-6.
//
#include "DataEntity.h"
#include "DataTable.h"
namespace simpla {
namespace data {
DataEntity::DataEntity(DataHolderBase* p) : m_data_(p) {}
DataEntity::DataEntity(DataEntity const& other) : m_data_(other.m_data_ == nullptr ? nullptr : other.m_data_->Copy()) {}
DataEntity::DataEntity(DataEntity&& other) : m_data_(other.m_data_) { other.m_data_ = nullptr; }
DataEntity::~DataEntity() {
    if (m_data_ != nullptr) { delete m_data_; }
}
std::type_info const& DataEntity::type() const { return m_data_ == nullptr ? typeid(void) : m_data_->type(); };
void DataEntity::swap(DataEntity& other) { std::swap(m_data_, other.m_data_); }
bool DataEntity::empty() const { return m_data_ == nullptr; }
std::ostream& DataEntity::Print(std::ostream& os, int indent) const {
    return m_data_ == nullptr ? os : m_data_->Print(os, indent);
}
DataEntity & DataEntity::operator=(DataEntity const& other) {
    DataEntity(other).swap(*this);
    return *this;
}

//DataEntity DataEntity::operator[](std::string const& url) {
//    if (m_data_ == nullptr) {
//        m_data_ = new DataTable;
//    } else if (!m_data_->isA(typeid(DataTable))) {
//        RUNTIME_ERROR << "Data entity is not indexable!" << std::endl;
//    }
//    return static_cast<DataTable*>(m_data_)->Insert(url).first;
//};
//DataEntity DataEntity::operator[](std::string const& url) const {
//    if (!m_data_->isA(typeid(DataTable))) { RUNTIME_ERROR << "Data entity is not indexable!" << std::endl; }
//    return (*static_cast<DataTable const*>(m_data_))[url];
//}
}  // namespace get_mesh{
}  // namespace simpla{
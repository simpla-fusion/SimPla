//
// Created by salmon on 17-7-19.
//

#include "simpla/SIMPLA_config.h"

#include "DataIOPort.h"
#include "simpla/engine/SPObject.h"
namespace simpla {
namespace data {
struct DataIOPort::pimpl_s {
    std::shared_ptr<DataTable> m_data_ = nullptr;
};

DataIOPort::DataIOPort(std::string uri) : m_pimpl_(new pimpl_s) {
    if (!uri.empty()) { m_pimpl_->m_data_ = std::make_shared<DataTable>(uri); }
}
DataIOPort::~DataIOPort(){};

void DataIOPort::Flush() {}

void DataIOPort::Serialize(data::DataTable &cfg) const {};
void DataIOPort::Deserialize(const DataTable &cfg) {
    EnableCreateFromDataTable<DataIOPort>::Deserialize(cfg);
    if (m_pimpl_->m_data_ == nullptr) {
        m_pimpl_->m_data_ = std::make_shared<DataTable>(cfg.GetValue<std::string>("URI", "mem://"));
    }
    m_pimpl_->m_data_->Set(cfg);
};

std::shared_ptr<DataEntity> DataIOPort::Get(std::string const &uri) const { return m_pimpl_->m_data_->Get(uri); }
void DataIOPort::Set(std::string const &uri, std::shared_ptr<DataEntity> const &d) { m_pimpl_->m_data_->Set(uri, d); }
void DataIOPort::Add(std::string const &uri, std::shared_ptr<DataEntity> const &d) { m_pimpl_->m_data_->Add(uri, d); }
int DataIOPort::Delete(std::string const &uri) { return m_pimpl_->m_data_->Delete(uri); }

void DataIOPort::Set(const DataTable &) { UNIMPLEMENTED; }
void DataIOPort::Add(const DataTable &) { UNIMPLEMENTED; };

id_type DataIOPort::TryGet(std::string const &uri, std::shared_ptr<DataEntity> *d) const {
    // FIXME:  Just workaround!
    *d = Get(uri);
    return NULL_ID;
}

id_type DataIOPort::TrySet(std::string const &uri, std::shared_ptr<DataEntity> const &d) {
    // FIXME:  Just workaround!
    Set(uri, d);
    return NULL_ID;
}
id_type DataIOPort::TryAdd(std::string const &uri, std::shared_ptr<DataEntity> const &d) {
    // FIXME:  Just workaround!
    Add(uri, d);
    return NULL_ID;
};
id_type DataIOPort::TryDelete(std::string const &uri) {
    // FIXME:  Just workaround!
    Delete(uri);
    return NULL_ID;
}
id_type DataIOPort::Cancel(id_type) {
    // FIXME:  Just workaround!
    return NULL_ID;
}
bool DataIOPort::Check(id_type) const {
    // FIXME:  Just workaround!
    return true;
}
}  // namespace data
}  // namespace simpla

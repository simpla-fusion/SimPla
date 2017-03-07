//
// Created by salmon on 17-3-6.
//
#include "DataBackendMemory.h"
#include <map>
namespace simpla {
namespace data {
struct DataBackendMemory::pimpl_s {
    static constexpr char split_char = '.';
    std::map<id_type, KeyValue> m_table_;

    id_type Hash(std::string const& s) const { return std::hash<std::string>()(s); }
    std::pair<KeyValue*, std::string> Traversal(std::string const& url, bool create_if_not_exist = false);
};
std::pair<KeyValue*, std::string> DataBackendMemory::pimpl_s::Traversal(std::string const& url,
                                                                        bool create_if_not_exist) {
    {
        auto it = m_table_.find(Hash(url));
        if (it != m_table_.end()) { return std::make_pair(&it->second, url); }
    }
    size_type start_pos = 0;
    KeyValue* p = nullptr;
    std::map<id_type, KeyValue>* t = &m_table_;

    //    while (start_pos < url.size()) {
    //        size_type pos = url.find(split_char, start_pos);
    //        std::string sub_k = url.substr(start_pos, pos == std::string::npos ? std::string::npos : pos - start_pos);
    //        id_type sub_id = Hash(sub_k);
    //        if (pos == std::string::npos) {
    //            p = &t->emplace(Hash(sub_k), KeyValue{sub_k, nullptr}).first->second;
    //            start_pos = std::string::npos;
    //            break;
    //        } else {
    //            auto it = t->find(sub_id);
    //            if (it == t->end() && create_if_not_exist) {
    //                auto t_table = std::make_shared<DataBackendMemory>();
    //                it = t->emplace(sub_id, KeyValue{sub_k, std::make_shared<DataTable>(t_table)}).first;
    //                t = &t_table->m_pimpl_->m_table_;
    //            } else if (it->second.second->isTable() &&
    //                       static_cast<DataTable*>(it->second.second.get())->backend_type() ==
    //                       typeid(DataBackendMemory)) {
    //                t =
    //                &static_cast<DataBackendMemory*>(static_cast<DataTable*>(it->second.second.get())->backend().get())
    //                         ->m_pimpl_->m_table_;
    //            } else {
    //                break;
    //            }
    //        }
    //        start_pos = pos + 1;
    //    }

    return std::make_pair(p, url.substr(0, start_pos));
};

DataBackendMemory::DataBackendMemory(std::string const& url, std::string const& status) : m_pimpl_(new pimpl_s) {
    if (url != "") { Open(url, status); }
}
DataBackendMemory::DataBackendMemory(const DataBackendMemory&){};

DataBackendMemory::~DataBackendMemory() {}
std::ostream& print_kv(std::ostream& os, int indent, KeyValue const& v) {
    //    if (v.second->isTable()) { os << std::endl << std::setw(indent + 1) << " "; }
    //    os << v.first << " = " << std::boolalpha << *v.second;
    return os;
}

std::ostream& DataBackendMemory::Print(std::ostream& os, int indent) const {
    //    if (!DataEntity::isNull()) { DataEntity::Print(os, indent + 1); }

    if (!m_pimpl_->m_table_.empty()) {
        auto it = m_pimpl_->m_table_.begin();
        auto ie = m_pimpl_->m_table_.end();
        if (it != ie) {
            os << "{ ";
            print_kv(os, indent, it->second);
            //            os << it->first << " = " << *it->second;
            ++it;
            for (; it != ie; ++it) {
                os << " , ";
                print_kv(os, indent, it->second);
                // os << " , " << it->first << " = " << *it->second;
            }

            os << " }";
        }
    };
    return os;
};
DataBackend* DataBackendMemory::Copy() const { return new DataBackendMemory(*this); }

bool DataBackendMemory::empty() const { return m_pimpl_->m_table_.empty(); };
void DataBackendMemory::Clear() { m_pimpl_->m_table_.clear(); };
void DataBackendMemory::Reset() { m_pimpl_->m_table_.clear(); };

std::pair<DataEntity*, bool> DataBackendMemory::Insert(std::string const& k){

};

std::pair<DataEntity*, bool> DataBackendMemory::Insert(std::string const& url, DataEntity const& v,
                                                       bool assign_is_exists) {
    auto res = m_pimpl_->Traversal(url, true);
    if (res.first != nullptr) {
        res.first->second = v;
    } else {
        RUNTIME_ERROR << " Error: can not insert entity at [ url:" << res.second << "]" << std::endl;
    }
};
DataEntity* DataBackendMemory::Find(std::string const& url) const {}

bool DataBackendMemory::Erase(std::string const& url) { return false; }
void DataBackendMemory::Open(std::string const& url, std::string const& status) { UNIMPLEMENTED; }
void DataBackendMemory::Close() { UNIMPLEMENTED; };
void DataBackendMemory::Flush() { UNIMPLEMENTED; };
void DataBackendMemory::Parse(std::string const& str) {
    size_type start_pos = 0;
    size_type end_pos = str.size();
    while (start_pos < end_pos) {
        size_type pos0 = str.find(';', start_pos);
        if (pos0 == std::string::npos) { pos0 = end_pos; }
        std::string key = str.substr(start_pos, pos0 - start_pos);
        size_type pos1 = key.find('=');
        std::string value = "";
        if (pos1 != std::string::npos) {
            value = key.substr(pos1 + 1);
            key = key.substr(0, pos1);
        }
        start_pos = pos0 + 1;
    }
}
}  // namespace data {
}  // namespace simpla{
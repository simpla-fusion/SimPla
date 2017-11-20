//
// Created by salmon on 17-8-18.
//

#include "DataEntry.h"
#include <iomanip>
#include <map>
#include <regex>
#include <vector>
#include "DataEntity.h"
#include "simpla/utilities/ParsingURI.h"

namespace simpla {
namespace data {
DataEntry::DataEntry(DataEntry const& other) : m_type_(other.m_type_), m_entity_(other.m_entity_) {}
DataEntry::DataEntry(eNodeType etype) : m_type_(etype) {}
DataEntry::DataEntry(std::shared_ptr<DataEntity> const& v) : m_type_(DN_ENTITY), m_entity_(v) {}
DataEntry::DataEntry(std::shared_ptr<const DataEntity> const& v) : m_type_(DN_ENTITY), m_entity_(v->Copy()) {}
DataEntry::~DataEntry() {
    if (isRoot()) { Flush(); }
};
std::shared_ptr<DataEntry> DataEntry::Copy() const { return std::shared_ptr<DataEntry>(new DataEntry(*this)); }

std::shared_ptr<DataEntry> DataEntry::Create(std::string const& s) {
    //    if (DataEntry::s_num_of_pre_registered_ == 0) { RUNTIME_ERROR << "No database is registered!" << s <<
    //    std::endl; }
    std::string uri = s.empty() ? "mem://" : s;

    std::string scheme;
    std::string path;
    std::string authority;
    std::string query;
    std::string fragment;

    std::tie(scheme, authority, path, query, fragment) = ParsingURI(uri);
    auto res = Factory<DataEntry>::Create(scheme);
    ASSERT(res != nullptr);
    if (SP_SUCCESS != res->Connect(authority, path, query, fragment)) {
        RUNTIME_ERROR << "Fail to connect  Data Backend [ " << scheme << " : " << authority << path << " ]"
                      << std::endl;
    }
    return res;
};
std::shared_ptr<DataEntry> DataEntry::Create(eNodeType e_type, std::string const& url) {
    return Create(url)->CreateNode(e_type);
}
std::shared_ptr<DataEntry> DataEntry::New(std::shared_ptr<DataEntity> const& v) {
    return std::shared_ptr<DataEntry>(new DataEntry(v));
}
KeyValue::KeyValue(std::string k) : m_key_(std::move(k)), m_node_(DataEntry::New(DataLight::New(true))) {}
KeyValue::KeyValue(KeyValue const& other) = default;
KeyValue::KeyValue(KeyValue&& other) noexcept = default;
KeyValue::~KeyValue() = default;
KeyValue& KeyValue::operator=(KeyValue const& other) {
    m_node_ = DataEntry::Create(DataEntry::DN_TABLE, "");
    m_node_->Set(other.m_key_, other.m_node_);
    return *this;
}

template <typename U>
std::shared_ptr<DataEntry> make_node(U const& u) {
    return DataEntry::New(DataLight::New(u));
}
std::shared_ptr<DataEntry> make_node(KeyValue const& kv) {
    auto res = DataEntry::Create(DataEntry::DN_TABLE);
    res->Set(kv.m_key_, kv.m_node_);
    return res;
}
std::shared_ptr<DataEntry> make_node(std::initializer_list<KeyValue> const& u) {
    auto res = DataEntry::Create(DataEntry::DN_TABLE);
    for (auto const& v : u) { res->Set(v.m_key_, v.m_node_); }
    return res;
}
template <typename U>
std::shared_ptr<DataEntry> make_node(std::initializer_list<U> const& u, ENABLE_IF(traits::is_light_data<U>::value)) {
    return DataEntry::New(DataLight::New(u));
}
template <typename U>
std::shared_ptr<DataEntry> make_node(std::initializer_list<U> const& u, ENABLE_IF(!traits::is_light_data<U>::value)) {
    auto res = DataEntry::Create()->CreateNode(DataEntry::DN_ARRAY);
    for (auto const& v : u) { res->Add(make_node(v)); }
    return res;
}
KeyValue& KeyValue::operator=(std::initializer_list<KeyValue> const& u) {
    m_node_ = make_node(u);
    return *this;
}
KeyValue& KeyValue::operator=(std::initializer_list<std::initializer_list<KeyValue>> const& u) {
    m_node_ = make_node(u);
    return *this;
}
KeyValue& KeyValue::operator=(std::initializer_list<std::initializer_list<std::initializer_list<KeyValue>>> const& u) {
    m_node_ = make_node(u);
    return *this;
}

std::istream& DataEntry::Parse(std::istream& is) {
    Parse(std::string(std::istreambuf_iterator<char>(is), {}));
    return is;
}
std::ostream& DataEntry::Print(std::ostream& os, int indent) const {
    switch (type()) {
        case DN_ENTITY: {
            auto entity = GetEntity();
            if (entity != nullptr) GetEntity()->Print(os, indent + 1);
        } break;
        case DN_ARRAY: {
            bool new_line = this->size() > 1;
            os << "[";
            if (this->size() > 0) {
                this->Get(0)->Print(os, indent + 1);
                for (size_type i = 1, ie = this->size(); i < ie; ++i) {
                    auto v = this->Get(i);
                    os << ", ";
                    //                    if (new_line && v->type() != DataEntry::DN_ENTITY) {
                    //                        os << std::endl << std::setw(indent + 1) << " ";
                    //                    }
                    this->Get(i)->Print(os, indent + 1);
                }
            }
            os << "]";
        } break;
        case DN_TABLE: {
            os << "{ ";
            bool is_first = true;
            bool new_line = this->size() > 1;
            this->Foreach([&](auto k, auto v) {
                ASSERT(v != nullptr);
                if (is_first) {
                    is_first = false;
                } else {
                    os << ", ";
                }
                if (new_line) { os << std::endl << std::setw(indent + 1) << " "; }
                FancyPrint(os, k, indent);
                os << " = ";
                v->Print(os, indent + 1);
            });

            if (new_line) { os << std::endl << std::setw(indent) << " "; }
            os << "}";
        } break;
        case DN_FUNCTION:
            os << "<FUNCTION>";
            break;
        default:
            os << "<N/A>";
            break;
    }

    return os;
}
std::ostream& operator<<(std::ostream& os, DataEntry const& entry) { return entry.Print(os, 0); }
std::istream& operator>>(std::istream& is, DataEntry& entry) { return entry.Parse(is); }

DataEntry::eNodeType DataEntry::type() const { return m_type_; }
size_type DataEntry::size() const { return m_entity_ == nullptr ? 0 : 1; }
std::shared_ptr<DataEntry> DataEntry::CreateNode(eNodeType e_type) const { return DataEntry::Create(e_type, ""); };
std::shared_ptr<DataEntry> DataEntry::CreateNode(std::string const& url, eNodeType e_type) {
    auto node = DataEntry::Create(e_type, "");
    Set(url, node);
    return node;
};
std::shared_ptr<const DataEntity> DataEntry::GetEntity() const { return m_entity_; }
std::shared_ptr<DataEntity> DataEntry::GetEntity() { return m_entity_; }
void DataEntry::SetEntity(std::shared_ptr<const DataEntity> const& e) { m_entity_ = e->Copy(); }
void DataEntry::SetEntity(std::shared_ptr<DataEntity> const& e) { m_entity_ = e; }

std::shared_ptr<const DataEntity> DataEntry::GetEntity(int N) const {
    std::shared_ptr<const DataEntity> res = GetEntity();
    if (res == nullptr && size() > 0) { res = Get(N)->GetEntity(); }
    return res;
}
std::shared_ptr<DataEntity> DataEntry::GetEntity(int N) {
    std::shared_ptr<DataEntity> res = GetEntity();
    if (res == nullptr && size() > 0) { res = Get(N)->GetEntity(); }
    return res;
}

size_type DataEntry::Set(std::string const& uri, const std::shared_ptr<DataEntry>& v) {
    UNIMPLEMENTED;
    return 0;
}
size_type DataEntry::Set(std::string const& uri, const std::shared_ptr<const DataEntry>& v) {
    return Set(uri, v->Copy());
}
size_type DataEntry::Set(index_type s, std::shared_ptr<DataEntry> const& v) { return Set(std::to_string(s), v); }
size_type DataEntry::Set(index_type s, std::shared_ptr<const DataEntry> const& v) { return Set(s, v->Copy()); }

size_type DataEntry::Add(std::string const& uri, const std::shared_ptr<DataEntry>& v) { return 0; }
size_type DataEntry::Add(std::string const& uri, const std::shared_ptr<const DataEntry>& v) {
    return Add(uri, v->Copy());
}
size_type DataEntry::Add(index_type s, std::shared_ptr<DataEntry> const& v) { return Add(std::to_string(s), v); }
size_type DataEntry::Add(index_type s, std::shared_ptr<const DataEntry> const& v) { return Add(s, v->Copy()); }
size_type DataEntry::Delete(std::string const& s) { return 0; }
size_type DataEntry::Delete(index_type s) { return Delete(std::to_string(s)); }
std::shared_ptr<const DataEntry> DataEntry::Get(index_type s) const { return Get(std::to_string(s)); }
std::shared_ptr<DataEntry> DataEntry::Get(index_type s) { return Get(std::to_string(s)); }
std::shared_ptr<const DataEntry> DataEntry::Get(std::string const& uri) const { return nullptr; }
std::shared_ptr<DataEntry> DataEntry::Get(std::string const& uri) { return nullptr; }
void DataEntry::Foreach(
    std::function<void(std::string const&, std::shared_ptr<const DataEntry> const&)> const& f) const {
    UNIMPLEMENTED;
}
void DataEntry::Foreach(std::function<void(std::string const&, std::shared_ptr<DataEntry> const&)> const& f) {
    UNIMPLEMENTED;
}

size_type DataEntry::Add(const std::shared_ptr<DataEntry>& v) { return Add(size(), v); }
size_type DataEntry::Add(const std::shared_ptr<const DataEntry>& v) { return Add(v->Copy()); }

size_type DataEntry::Set(const std::shared_ptr<DataEntry>& v) {
    size_type count = 0;
    if (v != nullptr) {
        v->Foreach([&](std::string k, std::shared_ptr<DataEntry> const& v) { count += Set(k, v); });
    }
    return count;
}
size_type DataEntry::Set(const std::shared_ptr<const DataEntry>& v) { return Set(v->Copy()); }
size_type DataEntry::SetValue(KeyValue const& kv) {
    ASSERT(type() == DN_TABLE);
    return Set(kv.m_key_, kv.m_node_);
}
size_type DataEntry::SetValue(std::initializer_list<KeyValue> const& u) {
    ASSERT(type() == DN_TABLE);
    size_type count = 0;
    for (auto const& kv : u) { count += Set(kv.m_key_, kv.m_node_); }
    return count;
}
size_type DataEntry::SetValue(std::string const& url, KeyValue const& kv) { return Set(url, make_node(kv)); };
size_type DataEntry::SetValue(std::string const& url, std::initializer_list<KeyValue> const& kv) {
    return Set(url, make_node(kv));
};
size_type DataEntry::SetValue(std::string const& url,
                              std::initializer_list<std::initializer_list<KeyValue>> const& kv) {
    return Set(url, make_node(kv));
};
size_type DataEntry::AddValue(std::string const& url, KeyValue const& kv) { return Add(url, make_node(kv)); };
size_type DataEntry::AddValue(std::string const& url, std::initializer_list<KeyValue> const& kv) {
    return Add(url, make_node(kv));
};
size_type DataEntry::AddValue(std::string const& url,
                              std::initializer_list<std::initializer_list<KeyValue>> const& kv) {
    return Add(url, make_node(kv));
};

//    static std::regex const sub_group_regex(R"(([^/?#]+)/)", std::regex::optimize);
// static std::regex const match_path_regex(R"(^(/?([/\S]+/)*)?([^/]+)?$)", std::regex::optimize);

/**
 * @brief Traverse  a hierarchical table base on URI  example: /ab/c/d/e
 *           if ''' return_if_not_exist ''' return when sub table does not exist
 *           else create a new table
 * @tparam T  table type
 * @tparam FunCheckTable
 * @tparam FunGetTable
 * @tparam FunAddTable
 * @param self
 * @param uri
 * @param check check  if sub obj is a table
 * @param get  return sub table
 * @param add  create a new table
 * @param return_if_not_exist
 * @return
 */
// std::pair<std::string, std::shared_ptr<DataEntry>> RecursiveFindNode(std::shared_ptr<DataEntry> root, std::string
// uri,
//                                                                    int flag) {
//    std::pair<std::string, std::shared_ptr<DataEntry>> res{"", root};
//
//    if (uri.empty() || uri == ".") { return res; }
//
//    if (uri[0] == '/') {
//        root = root->Root();
//    } else if (uri.substr(0, 3) == "../") {
//        root = root->Parent();
//        uri = uri.substr(3);
//    }
//    std::smatch uri_match_result;
//
//    if (!std::regex_match(uri, uri_match_result, match_path_regex)) {
//        RUNTIME_ERROR << "illegal URI: [" << uri << "]" << std::endl;
//    }
//    std::string path = uri_match_result.str(2);
//
//    if (!path.empty()) {
//        std::smatch sub_match_result;
//        auto t = root;
//
//        for (auto pos = path.cbegin(), end = path.cend();
//             std::regex_search(pos, end, sub_match_result, sub_group_regex); pos =
//             sub_match_result.suffix().first) {
//            std::string k = sub_match_result.str(1);
//            res.second = t->Get(k);
//            t = res.second;
//            if (res.second == nullptr) {
//                res.first = sub_match_result.suffix().str() + uri_match_result[3].str();
//                break;
//            }
//        }
//    }
//    auto key = uri_match_result.str(3);
//    if (!key.empty()) {
//        res.first = "";
//        res.second = res.second->Get(key);
//    }
//    return res;
//};

// DataEntry& DataEntry::operator+=(KeyValue const& u) {
//    Add(u.m_node_);
//    return *this;
//}
// DataEntry& DataEntry::operator+=(std::initializer_list<KeyValue> const& u) {
//    for (auto const& v : u) { Add(v.m_node_); }
//    return *this;
//}

}  // namespace data {
}  // namespace simpla {

//
// Created by salmon on 17-3-10.
//
#include <sys/stat.h>
#include <fstream>
#include <regex>
#include <sstream>
#include "../DataBlock.h"
#include "../DataEntity.h"
#include "../DataNode.h"
#include "simpla/parallel/MPIComm.h"

#include "HDF5Common.h"

namespace simpla {
namespace data {

struct DataNodeHDF5 : public DataNode {
    SP_DATA_NODE_HEAD(DataNodeHDF5, DataNode)

   protected:
   public:
    std::shared_ptr<DataNode> CreateNode(eNodeType e_type) const override;

    int Connect(std::string const& authority, std::string const& path, std::string const& query,
                std::string const& fragment) override;
    int Disconnect() override;
    int Flush() override;
    bool isValid() const override;
    void Clear() override;

    size_type size() const override;

    size_type Set(std::string const& uri, const std::shared_ptr<DataNode>& v) override;
    size_type Add(std::string const& uri, const std::shared_ptr<DataNode>& v) override;
    size_type Delete(std::string const& uri) override;
    std::shared_ptr<DataNode> Get(std::string const& uri) const override;
    void Foreach(std::function<void(std::string, std::shared_ptr<DataNode> const&)> const& f) const override;

    size_type Set(index_type s, const std::shared_ptr<DataNode>& v) override;
    size_type Add(index_type s, const std::shared_ptr<DataNode>& v) override;
    size_type Delete(index_type s) override;
    std::shared_ptr<DataNode> Get(index_type s) const override;

   private:
    hid_t m_file_ = -1;
    hid_t m_group_ = -1;
};
REGISTER_CREATOR(DataNodeHDF5, h5);
DataNodeHDF5::DataNodeHDF5(DataNode::eNodeType e_type) : base_type(e_type){};
DataNodeHDF5::~DataNodeHDF5() { Disconnect(); }

std::shared_ptr<DataNode> DataNodeHDF5::CreateNode(eNodeType e_type) const {
    std::shared_ptr<DataNode> res = nullptr;
    switch (e_type) {
        case DN_ENTITY:
            res = DataNode::New();
            break;
        case DN_ARRAY:
            res = DataNodeHDF5::New(DN_ARRAY);
            break;
        case DN_TABLE:
            res = DataNodeHDF5::New(DN_TABLE);
            break;
        case DN_FUNCTION:
            break;
        case DN_NULL:
        default:
            break;
    }
    res->SetParent(Self());
    return res;
};

int DataNodeHDF5::Connect(std::string const& authority, std::string const& path, std::string const& query,
                          std::string const& fragment) {
    Disconnect();
    std::string filename = path;
#ifdef MPI_FOUND
    if (GLOBAL_COMM.size() > 1) {
        auto pos = path.rfind('.');
        std::string prefix = (pos != std::string::npos) ? path.substr(0, pos) : path;
        int digital = static_cast<int>(std::floor(std::log(static_cast<double>(GLOBAL_COMM.size())))) + 1;
        if (GLOBAL_COMM.rank() == 0) {
            std::ofstream summary(prefix + ".summary.txt");
            for (int i = 0, ie = GLOBAL_COMM.size(); i < ie; ++i) {
                summary << prefix << "." << std::setfill('0') << std::setw(digital) << i << ".h5" << std::endl;
            }
        }
        std::ostringstream os;
        os << prefix << "." << std::setfill('0') << std::setw(digital) << GLOBAL_COMM.rank() << ".h5";
        filename = os.str();
    }

#endif
    if (filename.empty()) { filename = "simpla_unamed.h5"; }

    // = AutoIncreaseFileName(authority + "/" + path, "// .h5");

    LOGGER << "Create HDF5 File: [" << filename << "]" << std::endl;
    //    if (!(query.empty() && fragment.empty())) {
    //        TODO << "Parser query : [ " << query << " ] and fragment : [ " << fragment << " ]" << std::endl;
    //    }
    //    mkdir(authority.c_str(), 0777);
    H5_ERROR(m_file_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    H5_ERROR(m_group_ = H5Gopen(m_file_, "/", H5P_DEFAULT));
    return SP_SUCCESS;
};
int DataNodeHDF5::Disconnect() {
    if (m_group_ > -1) {
        H5_ERROR(H5Gclose(m_group_));
        m_group_ = -1;
    }

    if (m_file_ > -1) {
        H5Fclose(m_file_);
        m_file_ = -1;
    } else if (Parent() != nullptr) {
        Parent()->Disconnect();
    }
    return SP_SUCCESS;
}
int DataNodeHDF5::Flush() {
    if (m_file_ != -1) {
        H5_ERROR(H5Fflush(m_file_, H5F_SCOPE_GLOBAL));
    } else if (m_group_ != -1) {
        H5_ERROR(H5Gflush(m_group_));
    }
    return SP_SUCCESS;
}
bool DataNodeHDF5::isValid() const { return m_group_ != -1; }

size_type DataNodeHDF5::size() const {
    size_type count = 0;
    switch (type()) {
        case DN_TABLE:
        case DN_ARRAY: {
            H5G_info_t g_info;
            H5_ERROR(H5Gget_info(m_group_, &g_info));
            count += g_info.nlinks;
            H5O_info_t o_info;
            H5_ERROR(H5Oget_info(m_group_, &o_info));
            count += o_info.num_attrs;
        } break;
        case DN_ENTITY:
        case DN_FUNCTION:
            count = 1;
            break;
        default:
            break;
    }
    return count;
}

std::shared_ptr<DataNode> DataNodeHDF5::Get(std::string const& uri) const {
    if (uri.empty()) { return nullptr; }
    if (uri[0] == SP_URL_SPLIT_CHAR) { return Root()->Get(uri.substr(1)); }

    auto obj = const_cast<this_type*>(this)->shared_from_this();
    std::string k = uri;
    while (obj != nullptr && !k.empty()) {
        auto tail = k.find(SP_URL_SPLIT_CHAR);
        auto sub_key = k.substr(0, tail);
        k = tail == std::string::npos ? "" : k.substr(tail + 1);
        if (auto p = std::dynamic_pointer_cast<this_type>(obj)) {
            H5O_info_t o_info;
            if (H5Lexists(p->m_group_, sub_key.c_str(), H5P_DEFAULT) &&
                H5Oget_info_by_name(p->m_group_, sub_key.c_str(), &o_info, H5P_DEFAULT) >= 0) {
                switch (o_info.type) {
                    case H5O_TYPE_GROUP:
                        obj = p->CreateNode(DN_TABLE);
                        std::dynamic_pointer_cast<this_type>(obj)->m_group_ =
                            H5Gopen(p->m_group_, sub_key.c_str(), H5P_DEFAULT);
                        if (obj->Check("_DN_TYPE_", "DN_ARRAY")) { obj->m_type_ = DN_ARRAY; }
                        break;
                    case H5O_TYPE_DATASET: {
                        auto d_id = H5Dopen(p->m_group_, sub_key.c_str(), H5P_DEFAULT);
                        obj = DataNode::New(HDF5GetEntity(d_id, false));
                        H5_ERROR(H5Dclose(d_id));
                    } break;
                    default:
                        RUNTIME_ERROR << "Undefined data type!" << std::endl;
                        break;
                }
            } else if (H5Aexists(p->m_group_, sub_key.c_str()) != 0) {
                auto a_id = H5Aopen(p->m_group_, sub_key.c_str(), H5P_DEFAULT);
                obj = DataNode::New(HDF5GetEntity(a_id, true));
                H5_ERROR(H5Aclose(a_id));
                break;
            }
        }
    }

    return obj;
};

size_type DataNodeHDF5::Delete(std::string const& url) {
    if (m_group_ == -1 || url.empty()) { return 0; }
    size_type count = 0;
    auto pos = url.rfind(SP_URL_SPLIT_CHAR);
    hid_t grp = m_group_;
    std::string key = url;
    if (pos != std::string::npos) {
        grp = H5Gopen(grp, url.substr(0, pos).c_str(), H5P_DEFAULT);
        key = url.substr(pos + 1);
    }

    if (H5Aexists(grp, key.c_str()) != 0) {
        H5_ERROR(H5Adelete(grp, key.c_str()));
        ++count;
    } else {
        H5O_info_t o_info;
        if (H5Lexists(grp, url.c_str(), H5P_DEFAULT) &&
            H5Oget_info_by_name(m_group_, url.c_str(), &o_info, H5P_DEFAULT) >= 0) {
            FIXME << "Unable delete group/dataset";
        }
    }
    if (grp != m_group_) { H5_ERROR(H5Gclose(grp)); }

    return count;
}
void DataNodeHDF5::Clear() {
    if (m_group_ != -1) {}
}

size_type DataNodeHDF5::Set(std::string const& uri, const std::shared_ptr<DataNode>& v) {
    if (uri.empty() || v == nullptr) { return 0; }
    if (uri[0] == SP_URL_SPLIT_CHAR) { return Root()->Set(uri.substr(1), v); }

    size_type count = 0;
    auto obj = Self();
    std::string k = uri;
    while (obj != nullptr && !k.empty()) {  // assign node
        size_type tail = k.find(SP_URL_SPLIT_CHAR);
        if (tail == std::string::npos) {
            count = HDF5Set(obj->m_group_, k, v);
            break;
        } else {
            // find or create a sub-group
            auto grp = HDF5CreateOrOpenGroup(obj->m_group_, k.substr(0, tail));
            obj = std::dynamic_pointer_cast<this_type>(obj->CreateNode(DN_TABLE));
            obj->m_group_ = grp;
            k = k.substr(tail + 1);
        }
    }
    return count;
}
size_type DataNodeHDF5::Add(std::string const& uri, const std::shared_ptr<DataNode>& v) {
    if (uri.empty() || v == nullptr) { return 0; }
    if (uri[0] == SP_URL_SPLIT_CHAR) { return Root()->Set(uri.substr(1), v); }

    size_type count = 0;
    auto obj = Self();
    std::string k = uri;
    while (obj != nullptr && !k.empty()) {  // assign node
        size_type tail = k.find(SP_URL_SPLIT_CHAR);

        // find or create a sub-group
        auto grp = HDF5CreateOrOpenGroup(obj->m_group_, k.substr(0, tail));
        obj = std::dynamic_pointer_cast<this_type>(obj->CreateNode(DN_ARRAY));
        obj->m_group_ = grp;

        if (tail == std::string::npos) {
            count = HDF5Set(obj->m_group_, std::to_string(obj->size()), v);
            break;
        } else {
        }
        k = k.substr(tail + 1);
    }
    return count;
}

void DataNodeHDF5::Foreach(std::function<void(std::string, std::shared_ptr<DataNode> const&)> const& fun) const {
    if (m_group_ == -1) { return; };
    H5G_info_t g_info;
    H5_ERROR(H5Gget_info(m_group_, &g_info));

    size_type count = 0;
    for (hsize_t i = 0; i < g_info.nlinks; ++i) {
        ssize_t num = H5Lget_name_by_idx(m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        char buffer[num + 1];
        H5Lget_name_by_idx(m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, buffer, static_cast<size_t>(num + 1),
                           H5P_DEFAULT);

        fun(std::string(buffer), Get(std::string(buffer)));
    }
    H5O_info_t o_info;
    H5_ERROR(H5Oget_info(m_group_, &o_info));
    for (hsize_t i = 0; i < o_info.num_attrs; ++i) {
        ssize_t num = H5Aget_name_by_idx(m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        char buffer[num + 1];
        H5_ERROR(H5Aget_name_by_idx(m_group_, ".", H5_INDEX_NAME, H5_ITER_INC, i, buffer, static_cast<size_t>(num + 1),
                                    H5P_DEFAULT));

        fun(std::string(buffer), Get(std::string(buffer)));
    }
}
size_type DataNodeHDF5::Set(index_type s, const std::shared_ptr<DataNode>& v) { return Set(std::to_string(s), v); }
size_type DataNodeHDF5::Add(index_type s, const std::shared_ptr<DataNode>& v) { return Add(std::to_string(s), v); }
size_type DataNodeHDF5::Delete(index_type s) { return Delete(std::to_string(s)); }
std::shared_ptr<DataNode> DataNodeHDF5::Get(index_type s) const { return Get(std::to_string(s)); }

}  // namespace data{
}  // namespace simpla{
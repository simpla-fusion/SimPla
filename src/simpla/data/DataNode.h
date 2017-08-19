//
// Created by salmon on 17-8-18.
//

#ifndef SIMPLA_DATAENTRY_H
#define SIMPLA_DATAENTRY_H

#include <memory>
namespace simpla {
namespace data {
class DataEntity;
class DataNode : public std::enable_shared_from_this<DataNode> {
   public:
    DataNode() = default;
    virtual ~DataNode() = default;
    SP_DEFAULT_CONSTRUCT(DataNode);

    virtual size_type GetNumberOfChildren() const { return 0; }
    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri) { return nullptr; }
    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri) const { return nullptr; }

    virtual std::shared_ptr<DataNode> Root() { return GetNode("/"); }
    virtual std::shared_ptr<DataNode> Parent() const { return GetNode(".."); }

    virtual std::shared_ptr<DataNode> Child() const { return nullptr; }
    virtual std::shared_ptr<DataNode> Next() const { return nullptr; }

    virtual std::string Key() const { return ""; }
    virtual std::shared_ptr<DataEntity> Value() const { return nullptr; }

    virtual int SetValue(std::shared_ptr<DataEntity> const& v) { return 0; }
    virtual std::shared_ptr<DataNode> GetNodeByIndex(index_type idx) const { return nullptr; }
    virtual int SetNodeByIndex(index_type idx, std::shared_ptr<DataNode> const& v) { return 0; }

    virtual std::shared_ptr<DataNode> GetNodeByName(std::string const& idx) const { return nullptr; }
    virtual int SetNodeByName(std::string const& k, std::shared_ptr<DataNode> const& v) { return 0; }
    virtual int AddNode(std::shared_ptr<DataNode> const& v) { return SetNodeByIndex(-1, v); }
};
class DataNodeWithKey : public DataNode {
    std::string m_key_;
    std::shared_ptr<DataEntity> m_value_;
    explicit DataNodeWithKey(std::string k, std::shared_ptr<DataEntity> v = nullptr)
        : m_key_(std::move(k)), m_value_(std::move(v)) {}
    template <typename... Args>
    explicit DataNodeWithKey(std::string k, Args&&... args)
        : m_key_(std::move(k)), m_value_(DataEntity::New(std::forward<Args>(args)...)) {}

   public:
    template <typename... Args>
    static std::shared_ptr<DataNodeWithKey> New(Args&&... args) {
        return std::shared_ptr<DataNodeWithKey>(new std::shared_ptr<DataNodeWithKey>(std::forward<Args>(args)...));
    };
    ~DataNodeWithKey() override = default;
    SP_DEFAULT_CONSTRUCT(DataNodeWithKey);

    std::shared_ptr<DataEntity> Value() const override { return m_value_; }
    std::string Key() const override { return m_key_; }
};

class DataNodeWithOutKey : public DataNode {
    std::shared_ptr<DataEntity> m_value_;

   protected:
    explicit DataNodeWithOutKey(std::shared_ptr<DataEntity> v) : m_value_(std::move(v)) {}
    template <typename... Args>
    explicit DataNodeWithOutKey(Args&&... args) : m_value_(DataEntity::New(std::forward<Args>(args)...)) {}

   public:
    template <typename... Args>
    static std::shared_ptr<DataNodeWithOutKey> New(Args&&... args) {
        return std::shared_ptr<DataNodeWithOutKey>(
            new std::shared_ptr<DataNodeWithOutKey>(std::forward<Args>(args)...));
    };
    ~DataNodeWithOutKey() override = default;
    SP_DEFAULT_CONSTRUCT(DataNodeWithOutKey);

    std::shared_ptr<DataEntity> Value() const override { return m_value_; }
};
std::ostream& operator<<(std::ostream, DataNode const&);
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATAENTRY_H

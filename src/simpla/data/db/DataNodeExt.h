//
// Created by salmon on 17-8-30.
//

#ifndef SIMPLA_DATANODEEXT_H
#define SIMPLA_DATANODEEXT_H

namespace simpla {
namespace data {

class DataNodeEntity : public DataNode {
    SP_DEFINE_FANCY_TYPE_NAME(DataNodeEntity, DataNode)
   public:
    DataNode::eNodeType type() const override { return DataNode::DN_ENTITY; }
    size_type size() const override { return 1; }

    std::shared_ptr<DataEntity> GetEntity() const override = 0;
    virtual size_type SetEntity(std::shared_ptr<DataEntity> const&) = 0;
};
#define SP_DATA_NODE_ENTITY_HEAD(_CLASS_NAME_)              \
    SP_DEFINE_FANCY_TYPE_NAME(_CLASS_NAME_, DataNodeEntity) \
    SP_DATA_NODE_HEAD(_CLASS_NAME_)                         \
   public:                                                  \
    std::shared_ptr<DataEntity> GetEntity() const override; \
    size_type SetEntity(std::shared_ptr<DataEntity> const& entity) override;

class DataNodeArray : public DataNode {
    SP_DEFINE_FANCY_TYPE_NAME(DataNodeArray, DataNode)
   public:
    DataNode::eNodeType type() const override { return DataNode::DN_ARRAY; }
    size_type size() const override = 0;

    std::shared_ptr<DataNode> CreateChild() const override = 0;

    size_type Set(size_type s, std::shared_ptr<DataNode> const& v) override = 0;
    size_type Add(size_type s, std::shared_ptr<DataNode> const& v) override = 0;
    size_type Delete(size_type s) override = 0;
    std::shared_ptr<DataNode> Get(size_type s) const override = 0;
    virtual size_type PushBack(std::shared_ptr<DataNode> const& v) = 0;
};
#define SP_DATA_NODE_ARRAY_HEAD(_CLASS_NAME_)                                \
    SP_DEFINE_FANCY_TYPE_NAME(_CLASS_NAME_, DataNodeArray)                   \
    SP_DATA_NODE_HEAD(_CLASS_NAME_)                                          \
   public:                                                                   \
    size_type size() const override;                                         \
                                                                             \
    std::shared_ptr<DataNode> CreateChild() const override;                  \
                                                                             \
    size_type Set(size_type s, std::shared_ptr<DataNode> const& v) override; \
    size_type Add(size_type s, std::shared_ptr<DataNode> const& v) override; \
    size_type Delete(size_type s) override;                                  \
    std::shared_ptr<DataNode> Get(size_type s) const override;               \
    size_type PushBack(std::shared_ptr<DataNode> const& v) override;

class DataNodeTable : public DataNode {
    SP_DEFINE_FANCY_TYPE_NAME(DataNodeTable, DataNode)
   public:
    DataNode::eNodeType type() const override { return DataNode::DN_TABLE; }
    size_type size() const override = 0;

    std::shared_ptr<DataNode> CreateChild() const override = 0;

    size_type Set(std::string const& uri, std::shared_ptr<DataNode> const& v) override = 0;
    size_type Add(std::string const& uri, std::shared_ptr<DataNode> const& v) override = 0;
    size_type Delete(std::string const& uri) override = 0;
    std::shared_ptr<DataNode> Get(std::string const& uri) const override = 0;
    size_type Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& f) const override = 0;
};

#define SP_DATA_NODE_TABLE_HEAD(_CLASS_NAME_)                                           \
    SP_DEFINE_FANCY_TYPE_NAME(_CLASS_NAME_, DataNodeTable)                              \
    SP_DATA_NODE_HEAD(_CLASS_NAME_)                                                     \
                                                                                        \
   public:                                                                              \
    size_type size() const override;                                                    \
    std::shared_ptr<DataNode> CreateChild() const override;                             \
    size_type Set(std::string const& uri, std::shared_ptr<DataNode> const& v) override; \
    size_type Add(std::string const& uri, std::shared_ptr<DataNode> const& v) override; \
    size_type Delete(std::string const& uri) override;                                  \
    std::shared_ptr<DataNode> Get(std::string const& uri) const override;               \
    size_type Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& f) const override;

struct DataNodeFunction : public DataNode {
    SP_DEFINE_FANCY_TYPE_NAME(DataNodeFunction, DataNode)
   public:
    DataNode::eNodeType type() const override { return DataNode::DN_FUNCTION; }
    size_type size() const override { return 0; };
};

#define SP_DATA_NODE_FUNCTION_HEAD(_CLASS_NAME_)              \
    SP_DEFINE_FANCY_TYPE_NAME(_CLASS_NAME_, DataNodeFunction) \
    SP_DATA_NODE_HEAD(_CLASS_NAME_)
}
}
#endif  // SIMPLA_DATANODEEXT_H

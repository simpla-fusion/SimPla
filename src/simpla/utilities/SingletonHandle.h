//
// Created by salmon on 17-11-7.
//

#ifndef SIMPLA_SINGLETONHANDLE_H
#define SIMPLA_SINGLETONHANDLE_H
namespace simpla {
template <class T>
class SingletonHandle {
    std::shared_ptr<GeoEngine> m_engine_ = nullptr;
    void Initialize(std::string const &s);
    void Initialize(std::shared_ptr<data::DataNode> const &d = nullptr);
    void Initialize(int argc, char **argv);
    void Finalize();
}

    struct GeoEngineHolder {
        std::shared_ptr<GeoEngine> m_engine_ = nullptr;
        void Initialize(std::string const &s);
        void Initialize(std::shared_ptr<data::DataNode> const &d = nullptr);
        void Initialize(int argc, char **argv);
        void Finalize();
        GeoEngine &get();
        GeoEngine const &get() const;
    };
    void GeoEngineHolder::Initialize(std::string const &s) { m_engine_ = GeoEngine::New(s); }
    void GeoEngineHolder::Initialize(std::shared_ptr<data::DataNode> const &d) { m_engine_ = GeoEngine::New(d); }
    void GeoEngineHolder::Initialize(int argc, char **argv) { UNIMPLEMENTED; }
    void GeoEngineHolder::Finalize() { m_engine_.reset(); }
    GeoEngine &GeoEngineHolder::get() {
        if (m_engine_ == nullptr) { Initialize(); }
        ASSERT(m_engine_ != nullptr);
        return *m_engine_;
    }
    GeoEngine const &GeoEngineHolder::get() const {
        ASSERT(m_engine_ != nullptr);
        return *m_engine_;
    }
}  // namespace simpla {

#endif  // SIMPLA_SINGLETONHANDLE_H

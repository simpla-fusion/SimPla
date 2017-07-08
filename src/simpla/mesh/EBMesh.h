//
// Created by salmon on 17-7-8.
//

#ifndef SIMPLA_EBMESH_H
#define SIMPLA_EBMESH_H
namespace simpla {
namespace mesh {
template <typename...>
struct EBMesh;

template <typename TM>
struct EBMesh<TM> {
    typedef EBMesh<TM> this_type;
    EBMesh(TM *) {}
    ~EBMesh() {}

    template <typename TL, typename TR>
    void Fill(TL &lhs, TR const &rhs) const {
        //        CalculusPolicy<this_type>::Fill<traits::iform<TL>::value>(*this, lhs.get(), rhs);
    }

    template <typename TL, typename... Args>
    decltype(auto) GetEntity(TL &lhs, Args &&... args) const {
        return 0;
        //        return CalculusPolicy<this_type>::GetEntity<traits::iform<TL>::value>(*this, lhs.get(),
        //                                                                              std::forward<Args>(args)...);
    }
};

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_EBMESH_H

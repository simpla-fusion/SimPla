//
// Created by salmon on 16-6-7.
//


#include <memory>
#include "../SmallObjPool.h"

using namespace simpla;
struct point_s
{
    double x, y, z;
};

int main(int argc, char **argv)
{
    std::shared_ptr<sp::spPagePool> pool = sp::makePagePool(sizeof(point_s));
    std::shared_ptr<sp::spPage> pg = sp::makePage(pool);

    {

        size_t count = 0;
        SP_OBJ_INSERT(200, point_s, p, pg.get())
        {

            p->x = count + 20000;
            ++count;
        }
        printf("  count=%0lu \n", sp::spPageCount(pg.get()));

    }
    {
        size_t count = 0;
        SP_OBJ_FOREACH(point_s, p, pg.get())
        {
            p->x = count;
            ++count;
        }
        printf("  count=%0lu \n", sp::spPageCount(pg.get()));
    }
    {

        size_t count = 0;
        SP_OBJ_REMOVE_IF(point_s, p, pg.get(), tag)
        {
            tag = ((count % 3) == 0) ? 0 : 1;
            ++count;
        }
        printf("  count=%0lu \n", sp::spPageCount(pg.get()));

    }
}
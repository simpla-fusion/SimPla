//
// Created by salmon on 16-6-7.
//
#include <stdio.h>
#include <stdlib.h>
#include "../SmallObjPool.h"

struct point_s
{
    double x, y, z;
};

int main(int argc, char **argv)
{

    struct spPagePool *p_pool = spPagePoolCreate(sizeof(struct point_s));
    struct spPage *pg = spPageCreate(p_pool);


    {
        size_t count = 0;
        SP_OBJ_INSERT(200, struct point_s, p, pg, p_pool)
        {
            p->x = count + 20000;
            ++count;
        }
        printf("  count=%0lu \n", spPageCount(pg));

    }
    {
        size_t count = 0;
        SP_ELEMENT_FOREACH(struct point_s, p, pg)
        {
            p->x = count;
            ++count;
        }
        printf("  count=%0lu \n", spPageCount(pg));
    }
    {

        size_t count = 0;
        SP_ELEMENT_REMOVE_IF(struct point_s, p, pg, tag)
        {
            tag = ((count % 3) == 0) ? 0 : 1;
            ++count;
        }
        printf("  count=%0lu \n", spPageCount(pg));

    }


    printf("done!");
    spPageClose(pg, p_pool);
    spPagePoolClose(p_pool);
    exit(0);
}
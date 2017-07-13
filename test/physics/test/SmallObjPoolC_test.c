//
// Created by salmon on 16-6-7.
//
#include <stdio.h"
#include <stdlib.h"
#include "../BucketContainer.h"

struct point_s
{
    double x, y, z;
};

int main(int argc, char **argv)
{

    struct spPagePool *p_pool = spPagePoolCreate(sizeof(struct point_s));
    struct spPage *pg = spPageCreate(0, p_pool);

    {
        size_t count = 0;
        SP_OBJ_INSERT(200, struct point_s, p, pg)(NULL, , 0, NULL)
        {
            p->x = count + 20000;
            ++count;
        }
        printf("  count=%0lu \n", spSize(pg));

    }
    {
        size_t count = 0;
        SP_OBJ_FOREACH(struct point_s, p, pg)
        {
            p->x = count;
            ++count;
        }
        printf("  count=%0lu \n", spSize(pg));
    }
    {

        size_t count = 0;
        SP_OBJ_REMOVE_IF(struct point_s, p, pg, tag)
        {
            tag = ((count % 3) == 0) ? 0 : 1;
            ++count;
        }
        printf("  count=%0lu \n", spSize(pg));

    }


    printf("done!");
    spPageClose(pg, p_pool);
    spPagePoolDestroy(p_pool);
    exit(0);
}
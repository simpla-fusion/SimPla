//
// Created by salmon on 16-8-16.
//

#ifndef SIMPLA_SPMESHSFCZORDER_H
#define SIMPLA_SPMESHSFCZORDER_H

/**
 *  Space filling curvce : Z-Order (Morton)
 *
 */


//convert_database_r (x,y) to d
extern inline int spSFCCodeMorton2d(int n, int x, int y)
{

    return d;
}

//convert_database_r d to (x,y)
extern inline void spSFCEncodeMorton2d(int n, int d, int *x, int *y)
{

}

/**
 *  Hillbert
 **/
//rotate/flip a quadrant appropriately
extern inline void spSFCRotateHilbert2d(int n, int *x, int *y, int rx, int ry)
{
    if (ry == 0)
    {
        if (rx == 1)
        {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }

        //Swap x and y
        int t = *x;
        *x = *y;
        *y = t;
    }
}
//convert_database_r (x,y) to d
extern inline int spSFCCodeHilbert2d(int n, int x, int y)
{
    int rx, ry, s, d = 0;
    for (s = n / 2; s > 0; s /= 2)
    {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        spSFCRotateHilbert2d(s, &x, &y, rx, ry);
    }
    return d;
}

//convert_database_r d to (x,y)
extern inline void spSFCEncodeHilbert2d(int n, int d, int *x, int *y)
{
    int rx, ry, s, t = d;
    *x = *y = 0;
    for (s = 1; s < n; s *= 2)
    {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        spSFCRotateHilbert2d(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
}


#endif //SIMPLA_SPMESHSFCZORDER_H

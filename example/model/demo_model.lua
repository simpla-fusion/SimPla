
NX = 10
NY = 10
NZ = 1
LX = 10   --m --100000*rhoi --0.6
LY = 10   --2.0*math.pi/k0
LZ = 0   -- 2.0*math.pi/18
GW = 5

dimensions={NX ,NY ,NZ }

xmin={0.0,0.0,0.0}

xmax={LX,LY,LZ}



Domain={
  Dimensions={10 ,10 ,1},

  Polylines={
    { 0.1 , 1 ,0 },
    { 0.2 , 9.2 ,0 },
    { 9  , 7 ,0 },
    { 9  , 4 ,0 },
    { 4  , 2.2 ,0 }
  }

}

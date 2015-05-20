
NX = 10
NY = 10
NZ = 1
LX = 10   --m --100000*rhoi --0.6
LY = 10   --2.0*math.pi/k0
LZ = 0   -- 2.0*math.pi/18
GW = 5

Mesh={

    Dimensions={NX ,NY ,NZ },

    Box={{0.0,0.0,0.0},{LX,LY,LZ}}
}

NodeId=1

Polylines={
  { 0.0 , 1 ,0 },
  { 0.0 , 9.2 ,0 },
  { 8.3  , 7 ,0 },
  { 8.3  , 4 ,0 },
  { 4  , 2.2 ,0 }
}

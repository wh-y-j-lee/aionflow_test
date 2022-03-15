import os
import shutil
Ibpp = [
    [1.213396484375, 0.8600716145833333, 0.6985362955729166, 0.6584032389322916, 0.6849548339843748,  0.6581201985677083, 0.7548777669270834],
    [0.6781825358072916, 0.5208554687500001, 0.4952764485677082, 0.46510823567708337, 0.46543627929687503, 0.35492000325520834, 0.5092376302083333],
    [0.31613256835937503, 0.33581868489583333, 0.36005086263020836, 0.33273413085937503, 0.3216608072916667, 0.19616520182291666, 0.35370100911458335],
    [0.13072729492187501, 0.2382386881510417, 0.265823486328125, 0.23811816406250003, 0.22285115559895832, 0.12693098958333335, 0.25438053385416665]
]


lstVideo = ['Beauty', 'HoneyBee', 'ReadySteadyGo', 'YachtRide', 'Bosphorus', 'Jockey', 'ShakeNDry']

for vididx, vidname in enumerate(lstVideo):
    os.makedirs('../db/UVG/UVG_8Frames/'+vidname+'/gt')
    for imidx in range(8):
        shutil.copy('../db/UVG/images/'+vidname+'/im'+str(imidx+1).zfill(3)+'.png', '../db/UVG/UVG_8Frames/'+vidname+'/gt/im'+str(imidx+1).zfill(3)+'.png')
    os.makedirs('../db/UVG/UVG_8Frames/'+vidname+'/iframe')
    for qfidx, qf in enumerate([20, 23, 26, 29]):
        shutil.copy('../db/UVG/ref/'+vidname+'/H265L'+str(qf)+'/im0001.png', '../db/UVG/UVG_8Frames/'+vidname+'/iframe/H265L'+str(qf)+'.png')
        currentbpp = Ibpp[qfidx][vididx]
        with open('../db/UVG/UVG_8Frames/'+vidname+'/iframe/H265L'+str(qf)+'_bpp.txt', 'w') as f:
            f.write(str(currentbpp))

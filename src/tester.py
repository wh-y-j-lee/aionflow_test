import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from math import log10
from torchvision.utils import save_image as imwrite
import os
import copy
import cv2
from piq import psnr


def RGB2YUV(rgb):
    # rgb range: 0 ~ 1
    rgb *= 255
    r = rgb[:, 0:1, :, :]
    g = rgb[:, 1:2, :, :]
    b = rgb[:, 2:3, :, :]
    # y = 0.299 * r + 0.587 * g + 0.114 * b
    # u = (b - y) * 0.564 + 128
    # v = (r - y) * 0.713 + 128

    y = (0.257 * r) + (0.504 * g) + (0.098 * b) +16
    u = - (0.148 * r) - (0.291 * g) + (0.439 * b) + 128
    v = (0.439 * r) - (0.368 * g) - (0.071 * b) + 128

    yuv = torch.cat([y, u, v], dim=1)

    return torch.round(torch.clamp(yuv, min=0, max=255))


def YUV2RGB(yuv):
    y = yuv[:, 0:1, :, :]
    u = yuv[:, 1:2, :, :]
    v = yuv[:, 2:3, :, :]

    # b = (u - 128) / 0.564 + y
    # r = (v - 128) / 0.713 + y
    # g = (y - 0.299*r - 0.114*b) / 0.587

    # b = 1.164*(y - 16) + 2.018*(u - 128)
    # g = 1.164*(y - 16) - 0.813*(v - 128) - 0.391*(u - 128)
    # r = 1.164*(y - 16) + 1.596*(v - 128)
    
    r = 1.1641*(y - 16) - 0.0018*(u - 128) + 1.5958*(v - 128)
    g = 1.1641*(y - 16) - 0.3914*(u - 128) - 0.8135*(v - 128)
    b = 1.1641*(y - 16) + 2.0178*(u - 128) - 0.0012*(v - 128)

    rgb = torch.cat([r, g, b], dim=1)
    return torch.clamp(rgb, min=0, max=255)



class UVG_8Frames:
    def __init__(self, args):
        print('Loading UVG_8Frames dataset...')
        self.args = args
        self.lstVideo = ['Beauty', 'HoneyBee', 'ReadySteadyGo', 'Bosphorus', 'Jockey']
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.strOutDir = args.out_dir + '/UVG_8Frames'
        strInDir = args.test_input

        self.lstGT_yuv = []
        self.lstGT_rgb = []
        self.lstIframe = []
        self.lstBpp = []
        if not os.path.exists(self.strOutDir):
            os.makedirs(self.strOutDir)
        for vid in self.lstVideo:
            if not os.path.exists(self.strOutDir+'/'+vid):
                os.makedirs(self.strOutDir+'/'+vid)
            self.lstIframe.append(self.transform(Image.open(strInDir + '/' + vid + '/iframe/'+args.intra+'.png')).unsqueeze(0))
            with open(strInDir + '/' + vid + '/iframe/'+args.intra+'_bpp.txt') as f:
                strFirstline = f.readline()
                self.lstBpp.append(float(strFirstline))
            lstFrame_yuv = []
            lstFrame_rgb = []
            loc = strInDir + '/' + vid + '/gt/gt.yuv'
            fp = open(loc, "rb")
            for intIdx in range(8):
                w = args.test_width_size
                h = args.test_height_size

                y = np.frombuffer(fp.read(w * h), dtype=np.uint8).reshape(h, w)
                u = np.frombuffer(fp.read(w * h//4), dtype=np.uint8).reshape(h//2, w//2)
                v = np.frombuffer(fp.read(w * h//4), dtype=np.uint8).reshape(h//2, w//2)

                u = cv2.resize(u, (w, h), interpolation=cv2.INTER_NEAREST)
                v = cv2.resize(v, (w, h), interpolation=cv2.INTER_NEAREST)

                yuv = np.dstack((y, u, v))
                yuv = torch.from_numpy(yuv).unsqueeze(dim=0)
                yuv = yuv.permute(0, 3, 1, 2).contiguous().float()
                lstFrame_yuv.append(yuv)

                rgb = YUV2RGB(yuv)
                rgb = rgb / 255.0

                lstFrame_rgb.append(rgb)

            fp.close()
            self.lstGT_yuv.append(lstFrame_yuv)
            self.lstGT_rgb.append(lstFrame_rgb)
            self.constPSNRBias = 20*log10(255)

    def Test(self, model, current_epoch, logfile=None):
        torch.cuda.empty_cache()
        test_model = copy.deepcopy(model).cpu()
        test_model.model.update()

        with torch.no_grad():
            test_model.eval()
            strMSG = '{:<7s}{:<3d}'.format('Epoch: ', current_epoch) + '\n'
            if logfile is not None:
                logfile.write(strMSG)
            print(strMSG, end='')

            fltTotMeanPSNR = 0
            fltTotMeanBPP = 0
            for intIdxVid, strVid in enumerate(self.lstVideo):
                # compress
                tenIframe = self.lstIframe[intIdxVid]
                tenRef = None

                file_name = self.strOutDir+'/'+strVid+'/video_ep_'+str(current_epoch)+'.txt'
                with open(file_name, 'wb') as f:
                    for intIdxFrame in range(8):
                        tenInput = self.lstGT_rgb[intIdxVid][intIdxFrame]
                        if intIdxFrame == 0:
                            tenOut = tenIframe.clone().detach()
                        else:
                            compress_result = test_model.model.compress(tenInput, tenRef)
                            
                            strings = compress_result['strings']
                            shape = compress_result['shape']

                            tenOut = test_model.model.decompress(tenRef, strings, shape)
                            
                            # save strings
                            for idx in range(len(strings)):
                                f.write(np.array(len(strings[idx][0]), dtype=np.int32).tobytes())
                                f.write(strings[idx][0])
                            for idx in range(len(shape)):
                                # save shape
                                f.write(np.array(shape[idx][0], dtype=np.int32).tobytes())
                                f.write(np.array(shape[idx][1], dtype=np.int32).tobytes())
                        tenRef = tenOut.clone().detach()
                             
                # -------------------------------------------------------------------------------------
                # decompress
                tenIframe = self.lstIframe[intIdxVid]
                tenRef = None
                fltMeanPSNR = 0
                fltMeanBPP = 0
                
                with open(file_name, 'rb') as f:
                    for intIdxFrame in range(8):
                        tenInput = self.lstGT_yuv[intIdxVid][intIdxFrame]
                        if intIdxFrame == 0:
                            tenOut = tenIframe.clone().detach()

                            intBatch, intChannel, intHeight, intWidth = tenInput.size()
                            totalBitsPerFrame = intBatch * intHeight * intWidth
                            iframeTotalBits = self.lstBpp[intIdxVid] * totalBitsPerFrame
                            otherBPP = (os.path.getsize(file_name) * 8)
                            totalBits = iframeTotalBits + otherBPP
                            BPP = totalBits / (totalBitsPerFrame * 8)
                        else:
                            strings_input = list()
                            shape_input = list()

                            # read file
                            for _ in range(len(strings)):
                                length_ = np.frombuffer(f.read(4), dtype=np.int32)[0]   # read 4 bytes
                                strings_input.append([f.read(length_)])
                            for _ in range(len(shape)):
                                shape_0 = np.frombuffer(f.read(4), dtype=np.int32)[0]
                                shape_1 = np.frombuffer(f.read(4), dtype=np.int32)[0]
                                shape_input.append([shape_0, shape_1])

                            tenOut = test_model.model.decompress(tenRef, strings_input, shape_input)
                        tenRef = tenOut.clone().detach()

                        if current_epoch % self.args.test_every == 0:
                            imwrite(tenOut.clone().detach(), self.strOutDir+'/'+strVid+'/im'+str(intIdxFrame).zfill(3)+'_ep'+str(current_epoch)+'.png', range=(0, 1))
                        fltMeanPSNR += psnr(tenInput.clone().detach(), RGB2YUV(tenOut.clone().detach()), 255.0)

                fltMeanPSNR /= 8
                fltMeanBPP = BPP
                fltTotMeanPSNR += fltMeanPSNR
                fltTotMeanBPP += fltMeanBPP
                strMSG = '{:<20s}{:<6s}{:<20.16f}{:<6s}{:<20.16f}'.format('['+strVid+']', 'PSNR: ', fltMeanPSNR, ' BPP: ', fltMeanBPP) + '\n'
                print(strMSG, end='')
                if logfile is not None:
                    logfile.write(strMSG)
            fltTotMeanPSNR /= len(self.lstVideo)
            fltTotMeanBPP /= len(self.lstVideo)
            strMSG = '{:<20s}{:<6s}{:<20.16f}{:<6s}{:<20.16f}'.format('[Average]', 'PSNR: ', fltTotMeanPSNR, ' BPP: ', fltTotMeanBPP) + '\n'
            print(strMSG, end='')
            if logfile is not None:
                logfile.write(strMSG)

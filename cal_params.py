import argparse
from net import Net
import os
import time
from thop import profile
from thop import clever_format

import torch
from calflops import calculate_flops
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD Parameter and FLOPs")
parser.add_argument("--model_names", default=['RepirDet'], type=list,
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'RISTDnet', 'UIUNet', 'U-Net', 'RDIAN', 'ISTDU-Net'")
# from torchsummary import summary
global opt
opt = parser.parse_args()

if __name__ == '__main__':
    # opt.f = open('./params_' + (time.ctime()).replace(' ', '_') + '.txt', 'w')
    input_img = torch.rand(1,1,512,512).cuda()
    for model_name in opt.model_names:
        print(model_name)
        net = Net(model_name, mode='test').cuda()
        # if 'RepirDet' in opt.model_names:
        #     net.model.switch_to_deploy()

        macs, params = profile(net, inputs=(input_img, ))
        macs, params = clever_format([macs, params], "%.3f")
        # flops, macs, params = calculate_flops(net, input_shape=(1, 1, 512, 512))
        # summary(model=net, input_size=(1, 256, 256),batch_size=1, device="cuda")
        print(params)
        # print('FLOPs: %2fGFLOPs' % (flops/1e9))
        print(macs)
        # opt.f.write(model_name + '\n')
        # opt.f.write('Params: %2fM\n' % (params/1e6))
        # opt.f.write('FLOPs: %2fGFLOPs\n' % (flops/1e9))
        # opt.f.write('\n')
    # opt.f.close()
        
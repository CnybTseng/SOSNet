import sys
import torch
import timeit
sys.path.append('../JDE')
from mot.models.backbones import ShuffleNetV2
from sosnet import SOSNet

if __name__ == '__main__':
    print('SOSNet PK ShuffleNetV2')
    model1 = ShuffleNetV2(
        stage_repeat={'stage2': 4, 'stage3': 8, 'stage4': 4},
        stage_out_channels={'conv1': 24, 'stage2': 48, 'stage3': 96,
            'stage4': 192, 'conv5': 1024}).cuda().eval()
    arch={
        'conv1':  {'out_channels': 64},
        'stage2': {'out_channels': 256, 'repeate': 2, 'out': True},
        'stage3': {'out_channels': 384, 'repeate': 2, 'out': True},
        'stage4': {'out_channels': 512, 'repeate': 2, 'out': True},
        'conv5':  {'out_channels': 1024}}
    model2 = SOSNet(arch).cuda().eval()
    x = torch.rand(1, 3, 224, 224).cuda()
    loops = 1000
    with torch.no_grad():
        start = timeit.default_timer()
        for _ in range(loops):
            y = model1(x)
            torch.cuda.synchronize()
        end = timeit.default_timer()
        latency = (end - start) / loops
        print('ShuffleNetV2 latency: {} seconds.'.format(latency))
    for yi in y:
        print(yi.shape)
    with torch.no_grad():
        start = timeit.default_timer()
        for _ in range(loops):
            y = model2(x)
            torch.cuda.synchronize()
        end = timeit.default_timer()
        latency = (end - start) / loops
        print('SOSNet latency: {} seconds.'.format(latency))
    for yi in y:
        print(yi.shape)
    with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
        model2(x)
    print(prof.key_averages().table())
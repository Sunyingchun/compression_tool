# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/10/10 下午2:22

import time
from thop import profile
from tqdm import tqdm
from model_define.model import ResNet34
from timeit import default_timer as timer
import torch
from datetime import datetime


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def test_speed(model, device='gpu', test_time=10000):
    model.eval()
    inputs = torch.rand([1, 3, 112, 112])
    if device == 'gpu':
        model = model.to('cuda')
        inputs = inputs.to('cuda')
    else:
        model = model.to('cpu')
    print('Testing forward time,this may take a few minutes')
    start_time = timer()
    with torch.no_grad():
        for i in tqdm(range(test_time)):
            model(inputs)
    count = timer() - start_time
    forward_time = (count / test_time) * 1000
    print('平均forward时间为{}ms'.format(forward_time))
    return forward_time


def cal_flops(model, input_shape, device='gpu'):
    input_random = torch.rand(input_shape)
    if device == 'gpu':
        input_random = input_random.to('cuda')
        model = model.to('cuda')
    else:
        model = model.to('cpu')
    flops, params = profile(model, inputs=(input_random, ), verbose=False)
    return flops / (1024 * 1024 * 1024), params / (1024 * 1024)


def main():
    model = ResNet34()
    state_dict = torch.load('/home/user1/linx/program/LightFaceNet/work_space/models/model_train_best'
                            '/resnet34_model_2019-10-12-19-20_accuracy:0.7216981_step:84816_lin.pth')
    model.load_state_dict(state_dict)
    test_speed(model)

    # model = torch.load('/home/user1/linx/program/LightFaceNet/work_space/models/pruned_model/model_resnet34.pkl')
    # state_dict = torch.load('/home/user1/linx/program/LightFaceNet/work_space/models/pruned_model'
    #                         '/resnet34_best_pruned_0.6556604.pt')
    # model.load_state_dict(state_dict)
    # test_speed(model)


if __name__ == '__main__':
    main()


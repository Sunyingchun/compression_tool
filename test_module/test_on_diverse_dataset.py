# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/11/12 下午2:02
from test_module.test_on_face_recognition import TestOnFaceRecognition
from test_module.test_on_face_classification import TestOnFaceClassification


def test(args, model):
    if args.model == 'mobilenetv3' or args.model == 'mobilefacenet_lzc' or args.model == 'resnet34_lzc':
        test = TestOnFaceClassification(model, args.test_root_path, args.img_list_label_path)
        acc = test.test(args.test_batch_size)
        return acc
    else:
        test = TestOnFaceRecognition(model, args.test_root_path, args.img_list_label_path, args.data_source)
        accuracy = test.test3(args.test_batch_size)
        return accuracy
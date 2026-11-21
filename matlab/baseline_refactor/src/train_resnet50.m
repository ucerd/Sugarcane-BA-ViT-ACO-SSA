function [net,info] = train_resnet50(imdsTrain,cfg)
[net,info] = train_cnn(imdsTrain,"resnet50",cfg);
end

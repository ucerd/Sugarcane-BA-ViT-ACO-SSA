function [net,info] = train_alexnet(imdsTrain,cfg)
[net,info] = train_cnn(imdsTrain,"alexnet",cfg);
end

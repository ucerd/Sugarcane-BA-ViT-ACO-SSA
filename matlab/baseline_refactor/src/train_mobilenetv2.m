function [net,info] = train_mobilenetv2(imdsTrain,cfg)
[net,info] = train_cnn(imdsTrain,"mobilenetv2",cfg);
end

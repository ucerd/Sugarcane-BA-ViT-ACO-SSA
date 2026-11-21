function [net,info] = train_cnn(imdsTrain,modelName,cfg)
% Fixed epochs; outer evaluation fold is NEVER used for model selection.
switch string(modelName)
    case "resnet50", base = resnet50;
    case "mobilenetv2", base = mobilenetv2;
    case "alexnet", base = alexnet; % new optional comparator, not historical code
    otherwise, error('Unsupported model: %s',modelName);
end
inputSize = base.Layers(1).InputSize;
graph = layerGraph(base);
layers = graph.Layers;
% Supported models end in a learnable FC or 1x1 conv, then softmax/output.
idx = find(arrayfun(@(x) isa(x,'nnet.cnn.layer.ClassificationOutputLayer'),layers));
assert(isscalar(idx),'Expected one classification output layer.');
oldOutput = layers(idx).Name;
current = string(oldOutput);
connections = graph.Connections;
% Walk back from classification output through the single final branch.
for j=1:numel(layers)
    dest = extractBefore(string(connections.Destination)+"/","/");
    prev = string(connections.Source(dest==current));
    assert(isscalar(prev),'Unexpected branching in classifier head.');
    current = extractBefore(prev+"/","/");
    old = layers(strcmp({layers.Name},current));
    assert(isscalar(old),'Invalid classifier head connection.');
    if isa(old,'nnet.cnn.layer.FullyConnectedLayer') || isa(old,'nnet.cnn.layer.Convolution2DLayer'), break; end
end
assert(isa(old,'nnet.cnn.layer.FullyConnectedLayer') || isa(old,'nnet.cnn.layer.Convolution2DLayer'),'Terminal learnable layer not found.');
n = numel(categories(imdsTrain.Labels));
if isa(old,'nnet.cnn.layer.FullyConnectedLayer')
    new = fullyConnectedLayer(n,'Name',old.Name);
else
    assert(isequal(old.FilterSize,[1 1]),'Expected terminal 1x1 convolution.');
    new = convolution2dLayer(1,n,'Name',old.Name,'Padding','same');
end
graph = replaceLayer(graph,old.Name,new);
graph = replaceLayer(graph,oldOutput,classificationLayer('Name',oldOutput));
if cfg.trainingAugmentation
    augmenter = imageDataAugmenter('RandXReflection',true,'RandRotation',[-15 15]);
else
    augmenter = 'none';
end
augTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'ColorPreprocessing','gray2rgb','DataAugmentation',augmenter);
opts = trainingOptions('adam','MiniBatchSize',cfg.miniBatchSize, ...
    'MaxEpochs',cfg.maxEpochs,'InitialLearnRate',cfg.initialLearnRate, ...
    'Shuffle','every-epoch','Verbose',false,'Plots','none', ...
    'ExecutionEnvironment',char(cfg.executionEnvironment));
[net,info] = trainNetwork(augTrain,graph,opts);
end

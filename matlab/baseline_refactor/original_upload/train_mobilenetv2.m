function net = train_mobilenetv2(imdsTrain, imdsTest)

% Resize input to 224x224x3
inputSize = [224 224 3];
augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augTest  = augmentedImageDatastore(inputSize, imdsTest);

net0 = mobilenetv2;
lgraph = layerGraph(net0);
analyzeNetwork(net0)
numClasses = numel(categories(imdsTrain.Labels));

newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_new')
    softmaxLayer('Name','softmax_new')
    classificationLayer('Name','classoutput_new')
];

lgraph = replaceLayer(lgraph,'Logits', newLayers(1));
lgraph = replaceLayer(lgraph,'Logits_softmax', newLayers(2));
lgraph = replaceLayer(lgraph,'ClassificationLayer_Logits', newLayers(3));

options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 5, ...
    'ValidationData', augTest, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(augTrain, lgraph, options);

end

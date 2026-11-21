function net = train_resnet50(imdsTrain, imdsTest)

% Resize images to match ResNet50 input size
inputSize = [224 224 3];
augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augTest  = augmentedImageDatastore(inputSize, imdsTest);

% Load pretrained ResNet-50
net0 = resnet50;
lgraph = layerGraph(net0);

% Number of classes
numClasses = numel(categories(imdsTrain.Labels));

% Replace final layers
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_new')
    softmaxLayer('Name','softmax_new')
    classificationLayer('Name','classoutput_new')
];

lgraph = replaceLayer(lgraph,'fc1000', newLayers(1));
lgraph = replaceLayer(lgraph,'fc1000_softmax', newLayers(2));
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000', newLayers(3));

% Training options
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 5, ...
    'ValidationData', augTest, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train
net = trainNetwork(augTrain, lgraph, options);

end

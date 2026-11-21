clear; clc; close all;

% Load dataset
[imdsTrain, imdsTest, varieties] = load_dataset( ...
    "path/of csv file" ...
);

% Train Models
net_resnet = train_resnet50(imdsTrain, imdsTest);
net_mobilenet = train_mobilenetv2(imdsTrain, imdsTest);

% Evaluate Models
metrics_resnet = evaluate_model(net_resnet, imdsTest, "results_resnet50");
metrics_mobile = evaluate_model(net_mobilenet, imdsTest, "results_mobilenetv2");

% ROC Comparison
plot_roc_comparison(net_resnet, net_mobilenet, imdsTest);

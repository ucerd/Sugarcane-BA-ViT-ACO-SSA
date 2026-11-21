function plot_roc_comparison(netA, netB, imdsTest)

% Resize images for both networks
inputSizeA = netA.Layers(1).InputSize;
inputSizeB = netB.Layers(1).InputSize;

augTestA = augmentedImageDatastore(inputSizeA, imdsTest);
augTestB = augmentedImageDatastore(inputSizeB, imdsTest);

% True labels
trueLabels = imdsTest.Labels;
classes = categories(trueLabels);
numClasses = numel(classes);

% Prediction scores
[~, scoresA] = classify(netA, augTestA);
[~, scoresB] = classify(netB, augTestB);

% One-hot encoded ground truth
trueOneHot = onehotencode(trueLabels, 2);

% Standard ROC evaluation resolution
numPoints = 200;
meanX_A = zeros(numPoints,1);
meanY_A = zeros(numPoints,1);
meanX_B = zeros(numPoints,1);
meanY_B = zeros(numPoints,1);

AUC_A = 0; AUC_B = 0;

for i = 1:numClasses
    % ROC for model A
    [XA, YA, ~, aucA] = perfcurve(trueOneHot(:, i), scoresA(:, i), 1);
    AUC_A = AUC_A + aucA;
    meanX_A = meanX_A + interp1(linspace(0,1,length(XA)), YA, linspace(0,1,numPoints),'linear',0)';

    % ROC for model B
    [XB, YB, ~, aucB] = perfcurve(trueOneHot(:, i), scoresB(:, i), 1);
    AUC_B = AUC_B + aucB;
    meanX_B = meanX_B + interp1(linspace(0,1,length(XB)), YB, linspace(0,1,numPoints),'linear',0)';
end

% Macro-average
meanY_A = meanX_A / numClasses;
meanY_B = meanX_B / numClasses;
AUC_A = AUC_A / numClasses;
AUC_B = AUC_B / numClasses;

% Plot
figure;
plot(linspace(0,1,numPoints), meanY_A, 'LineWidth', 2); hold on;
plot(linspace(0,1,numPoints), meanY_B, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Macro-Average ROC Comparison');
legend(["ResNet50 (AUC=" + num2str(AUC_A,3) + ")", ...
        "MobileNetV2 (AUC=" + num2str(AUC_B,3) + ")"], 'Location', 'best');
saveas(gcf, "ROC_comparison.png");

end

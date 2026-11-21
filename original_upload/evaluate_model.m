function metrics = evaluate_model(net, imdsTest, savePrefix)

% Ensure savePrefix is string
savePrefix = string(savePrefix);

% Resize images to network input size automatically
inputSize = net.Layers(1).InputSize;
augTest = augmentedImageDatastore(inputSize, imdsTest);

% Get predictions
[preds, scores] = classify(net, augTest);
trueLabels = imdsTest.Labels;

% Confusion matrix
C = confusionmat(trueLabels, preds);

figure;
confusionchart(C, categories(trueLabels));
saveas(gcf, savePrefix + "_confusion.png");

% Accuracy
accuracy = sum(preds == trueLabels) / numel(trueLabels);

% Precision, Recall, F1
precision = diag(C) ./ sum(C,2);
recall = diag(C) ./ sum(C,1)';
f1 = 2 * (precision .* recall) ./ (precision + recall);

% Create table for class-level metrics
classMetrics = table(categories(trueLabels), precision, recall, f1, ...
    'VariableNames', {'Class', 'Precision', 'Recall', 'F1'});

% Save table
writetable(classMetrics, savePrefix + "_metrics.csv");

% Return struct instead of forcing accuracy into table
metrics = struct();
metrics.ClassMetrics = classMetrics;
metrics.Accuracy = accuracy;

disp("Overall Accuracy = " + accuracy);
disp(classMetrics);

end
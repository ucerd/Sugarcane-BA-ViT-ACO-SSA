function [preds,scores] = evaluate_model(net,imdsTest,classes,cfg)
% Deterministic resizing only; reorder score columns to explicit class order.
inputSize = net.Layers(1).InputSize;
augTest = augmentedImageDatastore(inputSize(1:2),imdsTest,'ColorPreprocessing','gray2rgb');
[preds,scores] = classify(net,augTest,'MiniBatchSize',cfg.miniBatchSize, ...
    'ExecutionEnvironment',char(cfg.executionEnvironment));
netClasses = string(net.Layers(end).Classes(:));
[present,order] = ismember(string(classes(:)),netClasses);
assert(all(present) && numel(netClasses)==numel(classes),'Network class order mismatch.');
scores = scores(:,order);
preds = string(preds(:));
assert(all(ismember(preds,classes)) && all(isfinite(scores),'all'),'Invalid predictions or scores.');
end

function cfg = config()
% New baseline defaults; these are NOT the original BA-ViT experiment settings.
cfg.seed = 42;
cfg.numFolds = 5;
cfg.generateNewFolds = false; % explicit opt-in: never silently invent historical folds
cfg.models = ["resnet50", "mobilenetv2"]; % optionally append "alexnet" (new run)
cfg.maxEpochs = 5;
cfg.miniBatchSize = 32;
cfg.initialLearnRate = 0.001;
cfg.executionEnvironment = "auto"; % use "cpu" for more deterministic execution
cfg.trainingAugmentation = false; % matches uploaded CNN scripts (resize only)
cfg.saveNetworks = true;
end

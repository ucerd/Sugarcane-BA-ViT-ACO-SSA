function run_all(manifestPath,datasetRoot,outputDir,cfg)
% Run NEW, supplemental grouped-CV CNN baseline experiments only.
root = fileparts(mfilename('fullpath')); addpath(fullfile(root,'src'));
if nargin<4, cfg = config(); end
assert(nargin>=3,'Usage: run_all(manifestPath,datasetRoot,outputDir,cfg)');
assert(cfg.numFolds>=2 && cfg.numFolds==fix(cfg.numFolds),'Invalid number of folds.');
models = string(cfg.models(:));
assert(~isempty(models) && numel(unique(models))==numel(models) && ...
    all(ismember(models,["resnet50","mobilenetv2","alexnet"])),'Invalid model list.');
if isfolder(outputDir)
    entries = dir(outputDir); assert(numel(entries)<=2,'Output directory must be empty.');
else
    mkdir(outputDir);
end
[imds,T,classes] = load_dataset(manifestPath,datasetRoot,cfg);
writetable(T,fullfile(outputDir,'effective_manifest.csv'));
env = struct('MATLAB',version,'Release',version('-release'),'Computer',computer,'Toolboxes',ver);
if ~strcmpi(cfg.executionEnvironment,'cpu')
    try, env.GPU = evalc('disp(gpuDevice)'); catch, env.GPU = 'GPU not available or not queried'; end
end
save(fullfile(outputDir,'run_configuration.mat'),'cfg','env','classes');
fid = fopen(fullfile(outputDir,'run_configuration.json'),'w'); assert(fid~=-1);
fprintf(fid,'%s',jsonencode(struct('config',cfg,'environment',env),'PrettyPrint',true)); fclose(fid);
N = height(T); K = numel(classes); scoreSets = cell(numel(models),1);
allFold = table(); allPooled = table();
for m=1:numel(models)
    modelName = models(m);
    modelDir = fullfile(string(outputDir),modelName); mkdir(modelDir);
    predictions = strings(N,1); scores = nan(N,K); seen = zeros(N,1);
    for f=1:cfg.numFolds
        rng(cfg.seed+f,'twister'); % same fold seed across all models
        evalIdx = T.fold==f; trainIdx = ~evalIdx;
        imdsTrain = subset(imds,find(trainIdx)); imdsTest = subset(imds,find(evalIdx));
        foldDir = fullfile(modelDir,"fold_"+f); mkdir(foldDir);
        tic; [net,info] = train_cnn(imdsTrain,modelName,cfg); seconds = toc;
        [p,s] = evaluate_model(net,imdsTest,classes,cfg);
        predictions(evalIdx)=p; scores(evalIdx,:)=s; seen(evalIdx)=seen(evalIdx)+1;
        metric = save_metrics(T.variety(evalIdx),p,classes,fullfile(foldDir,"test"));
        row = table(modelName,f,sum(trainIdx),sum(evalIdx),metric.Accuracy,metric.Macro(3),seconds, ...
            'VariableNames',{'Model','Fold','TrainImages','TestImages','Accuracy','MacroF1','TrainingSeconds'});
        allFold = [allFold;row]; %#ok<AGROW>
        save(fullfile(foldDir,'training_info.mat'),'info');
        if cfg.saveNetworks, save(fullfile(foldDir,'network.mat'),'net','-v7.3'); end
    end
    assert(all(seen==1) && all(isfinite(scores),'all'),'Missing or repeated out-of-fold predictions.');
    scoreSets{m}=scores;
    oof = table(T.image_id,T.source_image_id,T.fold,T.variety,predictions, ...
        'VariableNames',{'image_id','source_image_id','fold','true_label','predicted_label'});
    scoreNames = string(matlab.lang.makeUniqueStrings(cellstr("score_"+string(matlab.lang.makeValidName(cellstr(classes))))));
    oof = [oof array2table(scores,'VariableNames',cellstr(scoreNames))];
    writetable(oof,fullfile(modelDir,'out_of_fold_predictions.csv'));
    writetable(table(classes(:),scoreNames(:),'VariableNames',{'Class','ScoreColumn'}),fullfile(modelDir,'score_columns.csv'));
    metric = save_metrics(T.variety,predictions,classes,fullfile(modelDir,"pooled"));
    orig = T.is_original;
    save_metrics(T.variety(orig),predictions(orig),classes,fullfile(modelDir,"original_images_only"));
    row = table(modelName,metric.Accuracy,metric.Macro(3), ...
        mean(allFold.Accuracy(allFold.Model==modelName)),std(allFold.Accuracy(allFold.Model==modelName)), ...
        'VariableNames',{'Model','PooledAccuracy','PooledMacroF1','MeanFoldAccuracy','SDFoldAccuracy'});
    allPooled = [allPooled;row]; %#ok<AGROW>
    writetable(allFold,fullfile(outputDir,'fold_metrics.csv'));
    writetable(allPooled,fullfile(outputDir,'model_summary.csv'));
end
plot_roc_comparison(T.variety,scoreSets,classes,models,outputDir);
fprintf('Completed NEW CNN runs. Outputs: %s\n',outputDir);
end

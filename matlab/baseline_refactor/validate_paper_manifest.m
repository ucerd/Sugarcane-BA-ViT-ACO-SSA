function validate_paper_manifest(manifestPath,datasetRoot)
% Check structure/counts only, not provenance or historical authenticity.
root=fileparts(mfilename('fullpath')); addpath(fullfile(root,'src'));
cfg=config(); cfg.generateNewFolds=false;
[~,T,~]=load_dataset(manifestPath,datasetRoot,cfg);
classes=["CP77-400";"CPF-237";"CPF-250";"CPF-251";"CPF-253"];
counts=[563;560;550;553;564];
assert(height(T)==2790 && sum(T.is_original)==1000,'Unexpected manuscript image counts.');
for i=1:numel(classes)
    mask=T.variety==classes(i);
    assert(sum(mask)==counts(i) && sum(mask & T.is_original)==200,'Unexpected count for %s.',classes(i));
end
assert(numel(unique(T.variety))==5,'Unexpected varieties.');
disp('Manifest count/group checks passed; original provenance still requires author verification.');
end

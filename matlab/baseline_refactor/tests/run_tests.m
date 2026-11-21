function run_tests()
% Lightweight checks, no model weights or research images required.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(root,fullfile(root,'src'));
m=metrics_from_confusion([8 2;1 9],["A";"B"]);
assert(abs(m.Accuracy-.85)<1e-12);
assert(abs(m.ClassMetrics.Precision(1)-8/9)<1e-12);
assert(abs(m.ClassMetrics.Recall(1)-8/10)<1e-12);
assert(abs(m.ClassMetrics.Specificity(1)-9/10)<1e-12);
z=metrics_from_confusion([10 0;5 0],["A";"B"]);
assert(z.ClassMetrics.Precision(2)==0 && z.ClassMetrics.F1(2)==0);
variety=repelem(["A";"B"],10);
source_image_id=repelem("id"+string((1:10)'),2);
T=table(variety,source_image_id);
T.fold=assign_grouped_folds(T,5,42);
assert(isequal(T.fold,assign_grouped_folds(T,5,42)));
validate_folds(T,5);
T.fold(1)=mod(T.fold(1),5)+1;
rejected=false; try, validate_folds(T,5); catch, rejected=true; end
assert(rejected,'Leakage test failed.');
% Generate tiny synthetic images only in a temporary directory and test loading.
folder=tempname; mkdir(folder); cleanup=onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>
T.fold=assign_grouped_folds(T,5,42);
T.is_original=repmat([true;false],10,1);
T.image_id=T.source_image_id; T.image_id(~T.is_original)=T.image_id(~T.is_original)+"_derived";
T.relative_path=T.image_id+".png";
for j=1:height(T), imwrite(uint8(zeros(8,8,3)),fullfile(folder,T.relative_path(j))); end
manifest=fullfile(folder,'manifest.csv'); writetable(T,manifest);
[imds,U,classes]=load_dataset(manifest,folder,config());
assert(numel(imds.Files)==20 && height(U)==20 && numel(classes)==2);
assert(isequal(U.fold,T.fold));
m=audit_published_metrics(fullfile(folder,'audit'));
assert(sum(m.Confusion,'all')==2790 && trace(m.Confusion)==2626);
assert(abs(m.Accuracy-2626/2790)<1e-12);
disp('All lightweight MATLAB tests passed.');
end

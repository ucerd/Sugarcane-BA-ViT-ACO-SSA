function m = audit_published_metrics(outputDir)
% Arithmetic check of the manuscript's transcribed Table 4 only.
% This does NOT rerun the BA-ViT/ACO-SSA experiment or reconstruct predictions.
root = fileparts(mfilename('fullpath')); addpath(fullfile(root,'src'));
if nargin==0, outputDir=fullfile(root,'results','paper_table_audit'); end
if ~isfolder(outputDir), mkdir(outputDir); end
C = [535 0 6 21 1; 2 541 5 5 7; 4 8 523 0 15; 29 13 2 500 9; 5 8 16 8 527];
classes = ["CP77-400";"CPF-237";"CPF-250";"CPF-251";"CPF-253"];
m = metrics_from_confusion(C,classes);
writematrix(C,fullfile(outputDir,'transcribed_table4.csv'));
writetable(m.ClassMetrics,fullfile(outputDir,'recomputed_class_metrics.csv'));
writetable(array2table([m.Accuracy m.Macro],'VariableNames', ...
    {'Accuracy','MacroPrecision','MacroRecall','MacroF1','MacroSpecificity','MacroGMean'}), ...
    fullfile(outputDir,'recomputed_summary.csv'));
fprintf('Table 4 arithmetic only: accuracy %.4f%%, macro F1 %.4f%%\n',100*m.Accuracy,100*m.Macro(3));
end

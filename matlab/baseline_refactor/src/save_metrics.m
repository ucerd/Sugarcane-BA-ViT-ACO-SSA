function m = save_metrics(truth,pred,classes,prefix)
classes = string(classes(:));
assert(numel(truth)==numel(pred) && all(ismember(string(truth),classes)) && ...
    all(ismember(string(pred),classes)),'Invalid label vectors.');
C = confusionmat(cellstr(string(truth(:))),cellstr(string(pred(:))),'Order',cellstr(classes));
m = metrics_from_confusion(C,classes);
writetable(m.ClassMetrics,prefix+"_class_metrics.csv");
writematrix(C,prefix+"_confusion.csv");
writetable(table(classes,'VariableNames',{'Class'}),prefix+"_confusion_order.csv");
summary = array2table([m.Accuracy m.Macro],'VariableNames', ...
    {'Accuracy','MacroPrecision','MacroRecall','MacroF1','MacroSpecificity','MacroGMean'});
writetable(summary,prefix+"_summary.csv");
f = figure('Visible','off'); c = onCleanup(@() close(f)); %#ok<NASGU>
confusionchart(C,cellstr(classes));
saveas(f,prefix+"_confusion.png");
end

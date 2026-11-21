function metrics = metrics_from_confusion(C,classes)
% Rows = actual labels; columns = predicted labels.
assert(ismatrix(C) && size(C,1)==size(C,2) && size(C,1)==numel(classes) && ...
    all(isfinite(C) & C>=0 & C==fix(C),'all') && sum(C,'all')>0,'Invalid confusion matrix.');
TP = diag(C); FP = sum(C,1)'-TP; FN = sum(C,2)-TP;
TN = sum(C,'all')-TP-FP-FN;
precision = safe_div(TP,TP+FP);
recall = safe_div(TP,TP+FN);
specificity = safe_div(TN,TN+FP);
f1 = safe_div(2*TP,2*TP+FP+FN);
gmean = sqrt(recall.*specificity);
metrics.ClassMetrics = table(string(classes(:)),precision,recall,f1,specificity,gmean, ...
    'VariableNames',{'Class','Precision','Recall','F1','Specificity','GMean'});
metrics.Accuracy = sum(TP)/sum(C,'all');
metrics.Macro = mean([precision recall f1 specificity gmean],1);
metrics.Confusion = C;
end
function y = safe_div(a,b)
y = zeros(size(a)); mask = b~=0; y(mask) = a(mask)./b(mask);
end

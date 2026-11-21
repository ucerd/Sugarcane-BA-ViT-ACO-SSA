function aucTable = plot_roc_comparison(truth,scoreSets,classes,modelNames,outputDir)
% Each score set is N-by-K, in the explicitly supplied class order.
% Interpolate on actual FPR coordinates; class AUC from perfcurve, not fprGrid.
classes = string(classes(:)); modelNames = string(modelNames(:));
fprGrid = linspace(0,1,201)';
auc = zeros(numel(classes),numel(modelNames));
f = figure('Visible','off'); c = onCleanup(@() close(f)); %#ok<NASGU>
hold on; legendText = strings(numel(modelNames),1);
for m=1:numel(modelNames)
    S = scoreSets{m};
    assert(isequal(size(S),[numel(truth),numel(classes)]) && all(isfinite(S),'all'),'Invalid scores.');
    curves = zeros(numel(fprGrid),numel(classes));
    for k=1:numel(classes)
        y = string(truth(:))==classes(k);
        assert(any(y) && any(~y),'ROC requires positive and negative samples.');
        [x,t,~,auc(k,m)] = perfcurve(y,S(:,k),true);
        [ux,~,g] = unique(x,'sorted');
        uy = accumarray(g,t,[],@max);
        % Repeated FPRs collapse to upper TPR for the display only.
        curves(:,k) = interp1(ux,uy,fprGrid,'linear');
    end
    plot(fprGrid,mean(curves,2),'LineWidth',1.8);
    legendText(m) = modelNames(m)+ " (macro AUC="+compose('%.4f',mean(auc(:,m)))+")";
end
plot([0 1],[0 1],'k:','HandleVisibility','off');
xlabel('False positive rate'); ylabel('True positive rate');
title('New CNN experiments: pooled out-of-fold ROC');
legend(cellstr(legendText),'Location','southeast'); axis([0 1 0 1]); grid on;
saveas(f,fullfile(outputDir,'ROC_comparison.png'));
aucTable = array2table([auc;mean(auc,1)],'VariableNames',cellstr(modelNames));
aucTable = addvars(aucTable,[classes;"Macro-average"],'Before',1,'NewVariableNames','Class');
writetable(aucTable,fullfile(outputDir,'ROC_AUC.csv'));
end

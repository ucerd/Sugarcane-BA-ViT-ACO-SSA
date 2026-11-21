function run_feature_reference(pythonOutput, outputDir, seed, counts, population, iterations)
% MATLAB downstream reference using Python's fold-specific BA-ViT exports.
% Does not train the historical MATLAB BA-ViT architecture.
if nargin<3, seed=42; end
if nargin<4, counts=[100 250 500 750 1000]; end
if nargin<5, population=10; end
if nargin<6, iterations=100; end
assert(~isfolder(outputDir),'Use a fresh output directory.'); mkdir(outputDir);
T=readtable(fullfile(pythonOutput,'effective_manifest.csv'),'TextType','string');
assert(all(ismember(T.fold,1:5)) && numel(unique(T.image_id))==height(T));
G=findgroups(T.source_image_id);
for g=1:max(G), assert(numel(unique(T.fold(G==g)))==1 && numel(unique(T.variety(G==g)))==1); end
names=["LSVM","QSVM","CuSVM","FGSVM","MGSVM","CGSVM","FKNN","CKNN","CrKNN"];
P=table('Size',[0 6],'VariableTypes',{'string','double','string','double','string','string'},'VariableNames',{'image_id','fold','model','features','truth','prediction'});
for f=1:5
    tr=T(T.fold~=f,:); te=T(T.fold==f,:); folder=fullfile(pythonOutput,sprintf('fold_%d',f));
    A=readtable(fullfile(folder,'train_features.csv'),'TextType','string'); B=readtable(fullfile(folder,'test_features.csv'),'TextType','string');
    [ok,ai]=ismember(tr.image_id,A.image_id); assert(all(ok)&&height(A)==height(tr)); [ok,bi]=ismember(te.image_id,B.image_id); assert(all(ok)&&height(B)==height(te));
    X=A{ai,2:end}; Z=B{bi,2:end}; y=tr.variety; yt=te.variety;
    for k=counts
        [ix,h]=hybrid_select_reference(X,y,tr.source_image_id,k,seed+f,population,iterations);
        writematrix(ix,fullfile(outputDir,sprintf('mask_f%d_k%d.csv',f,k))); writematrix(h,fullfile(outputDir,sprintf('fitness_f%d_k%d.csv',f,k)));
        for j=1:numel(names)
            name=names(j);
            if j<=6
                if j==1, L=templateSVM('KernelFunction','linear','Standardize',true);
                elseif j<=3, L=templateSVM('KernelFunction','polynomial','PolynomialOrder',j,'Standardize',true);
                else, scales=[sqrt(k)/4 sqrt(k) sqrt(k)*4]; L=templateSVM('KernelFunction','gaussian','KernelScale',scales(j-3),'Standardize',true); end
                mdl=fitcecoc(X(:,ix),y,'Learners',L);
            else
                n=[1 10 100]; dist={'euclidean','cosine','euclidean'}; if n(j-6)>height(tr), continue; end
                mdl=fitcknn(X(:,ix),y,'NumNeighbors',n(j-6),'Distance',dist{j-6},'Standardize',true);
            end
            pred=string(predict(mdl,Z(:,ix))); N=height(te);
            P=[P;table(te.image_id,repmat(f,N,1),repmat(name,N,1),repmat(k,N,1),yt,pred,'VariableNames',P.Properties.VariableNames)]; %#ok<AGROW>
        end
    end
end
writetable(P,fullfile(outputDir,'predictions.csv'));
[G,model,features]=findgroups(P.model,P.features); accuracy=splitapply(@(a,b)mean(a==b),P.truth,P.prediction,G);
writetable(table(model,features,accuracy),fullfile(outputDir,'pooled_accuracy.csv'));
end

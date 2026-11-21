function [best, history] = hybrid_select_reference(X,y,groups,k,seed,pop,iters)
% New ACO->SSA reference search, using training-side group holdout only.
rng(seed); d=size(X,2); assert(k>=1 && k<=d && pop>=2 && iters>=2);
y=string(y); groups=string(groups); [g,~,gi]=unique(groups); gy=strings(numel(g),1);
for j=1:numel(g), labels=unique(y(gi==j)); assert(numel(labels)==1); gy(j)=labels; end
val=false(numel(g),1); classes=unique(gy);
for j=1:numel(classes)
    ix=find(gy==classes(j)); assert(numel(ix)>=3); ix=ix(randperm(numel(ix))); val(ix(1:max(1,floor(numel(ix)/3))))=true;
end
b=val(gi); a=~b; pher=ones(1,d); loss=inf; best=[]; history=zeros(iters,1); na=floor(iters/2);
for it=1:na
    pher=.8*pher;
    for j=1:pop
        ix=sort(randsample(d,k,false,pher)); evaluate(ix);
    end
    pher(best)=pher(best)+1/(1+loss); history(it)=loss;
end
pos=rand(pop,d); pos(1,best)=1;
for it=na+1:iters
    food=zeros(1,d); food(best)=1; c1=2*exp(-(4*(it-na)/(iters-na))^2);
    pos(1,:)=max(0,min(1,food+c1*(2*rand(1,d)-1)));
    for j=2:pop, pos(j,:)=(pos(j,:)+pos(j-1,:))/2; end
    for j=1:pop
        [~,ix]=maxk(pos(j,:),k); evaluate(sort(ix));
    end
    history(it)=loss;
end
    function evaluate(ix)
        learner=templateSVM('KernelFunction','polynomial','PolynomialOrder',3,'Standardize',true);
        mdl=fitcecoc(X(a,ix),y(a),'Learners',learner); pred=predict(mdl,X(b,ix));
        cost=.99*(1-mean(string(pred)==y(b)))+.01*k/d;
        if cost<loss, loss=cost; best=ix; end
    end
end

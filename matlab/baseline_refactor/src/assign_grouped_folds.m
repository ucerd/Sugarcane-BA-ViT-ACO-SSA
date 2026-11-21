function folds = assign_grouped_folds(T,k,seed)
% New deterministic, class-stratified assignment at source-image level.
% balances number of groups, not necessarily total derived views per fold.
state = rng; cleanup = onCleanup(@() rng(state)); %#ok<NASGU>
rng(seed,'twister');
folds = zeros(height(T),1);
classes = sort(unique(string(T.variety)));
for c=1:numel(classes)
    rows = string(T.variety)==classes(c);
    ids = sort(unique(string(T.source_image_id(rows))));
    assert(numel(ids)>=k, 'Each variety must have at least %d source images.',k);
    ids = ids(randperm(numel(ids)));
    for j=1:numel(ids)
        folds(rows & string(T.source_image_id)==ids(j)) = mod(j-1,k)+1;
    end
end
validate_folds(table(string(T.variety),string(T.source_image_id),folds, ...
    'VariableNames',{'variety','source_image_id','fold'}),k);
end

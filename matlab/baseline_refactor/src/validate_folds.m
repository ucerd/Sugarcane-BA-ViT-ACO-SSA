function validate_folds(T,k)
G = findgroups(string(T.source_image_id));
assert(all(isfinite(G)) && all(isfinite(T.fold) & T.fold==fix(T.fold) & T.fold>=1 & T.fold<=k),'Invalid groups or folds.');
for g=1:max(G)
    assert(numel(unique(T.fold(G==g)))==1,'Source image leakage across folds.');
    assert(numel(unique(string(T.variety(G==g))))==1,'Mixed labels within source group.');
end
classes = unique(string(T.variety));
for j=1:k
    assert(isequal(sort(unique(string(T.variety(T.fold==j)))),sort(classes)), 'Fold %d lacks a class.',j);
    assert(isequal(sort(unique(string(T.variety(T.fold~=j)))),sort(classes)), 'Training fold %d lacks a class.',j);
end
end

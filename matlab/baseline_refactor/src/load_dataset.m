function [imds, T, classes] = load_dataset(manifestPath, datasetRoot, cfg)
% Read explicit image-parent groups; disallow implicit image-level splitting.
required = ["image_id","relative_path","variety","source_image_id","is_original"];
opts = detectImportOptions(manifestPath,'TextType','string');
assert(all(ismember(required,string(opts.VariableNames))), ...
    'Required manifest columns are missing.');
opts = setvartype(opts,cellstr(required),'string');
T = readtable(manifestPath,opts);
assert(~isempty(T) && all(ismember(required,string(T.Properties.VariableNames))), ...
    'Manifest must contain image_id, relative_path, variety, source_image_id, is_original.');
for field = required(1:4)
    T.(field) = strip(string(T.(field)));
    assert(all(~ismissing(T.(field)) & strlength(T.(field))>0), 'Empty value in %s.',field);
end
original = lower(strip(string(T.is_original)));
assert(all(ismember(original,["1","0","true","false"])), 'is_original must be 0/1 or true/false.');
T.is_original = ismember(original,["1","true"]);
assert(numel(unique(T.image_id))==height(T),'Duplicate image_id.');
assert(isfolder(datasetRoot),'Dataset root does not exist.');
[ok,attr] = fileattrib(datasetRoot); assert(ok,'Cannot resolve dataset root.');
datasetRoot = attr.Name;
% Use portable relative paths, prohibit absolute and parent traversal paths.
p = replace(T.relative_path, '\', '/');
assert(all(~startsWith(p,'/') & ~contains(p,':')), 'Use relative paths.');
for i=1:height(T)
    parts = split(p(i),'/');
    assert(all(~ismember(parts,[".","..",""])), 'Invalid relative path: %s',p(i));
end
T.relative_path = p;
assert(numel(unique(lower(p)))==height(T),'Duplicate image paths (case-insensitive).');
paths = fullfile(string(datasetRoot),p);
assert(all(isfile(paths)),'One or more manifest image files are missing.');
classes = sort(unique(T.variety));
assert(numel(classes)>=2,'At least two varieties required.');
[G, groupIDs] = findgroups(T.source_image_id);
for g=1:numel(groupIDs)
    rows = G==g;
    assert(numel(unique(T.variety(rows)))==1,'Mixed labels in source group %s.',groupIDs(g));
    assert(sum(T.is_original(rows))==1,'Expected exactly one original in group %s.',groupIDs(g));
    assert(T.image_id(rows & T.is_original)==groupIDs(g), 'Original image_id must equal source_image_id.');
end
if ismember('fold',T.Properties.VariableNames)
    folds = str2double(string(T.fold));
    assert(all(isfinite(folds) & folds==fix(folds) & folds>=1 & folds<=cfg.numFolds), 'Invalid fold values.');
    T.fold = folds;
elseif cfg.generateNewFolds
    T.fold = assign_grouped_folds(T,cfg.numFolds,cfg.seed);
else
    error('No fold column. Supply verified assignments or explicitly set cfg.generateNewFolds=true for a NEW experiment.');
end
validate_folds(T,cfg.numFolds);
imds = imageDatastore(cellstr(paths),'Labels',categorical(T.variety,classes));
assert(isequal(replace(string(imds.Files(:)),'\','/'),replace(string(paths(:)),'\','/')), ...
    'Datastore order differs from manifest; resolve before running.');
end

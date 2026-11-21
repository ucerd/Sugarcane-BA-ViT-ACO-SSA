function [imdsTrain, imdsTest, varieties] = load_dataset(manifestPath, datasetRoot)

    % Read metadata file
    data = readtable(manifestPath);

    % Construct full image paths
    data.fullpath = fullfile(datasetRoot, data.relative_path);

    % Assign labels
    labels = categorical(data.variety);

    % Create datastore
    imds = imageDatastore(data.fullpath, ...
        'Labels', labels, ...
        'IncludeSubfolders', false);

    % Split dataset into train & test
    trainIdx = strcmp(data.split, 'train');
    testIdx  = strcmp(data.split, 'test');

    imdsTrain = subset(imds, trainIdx);
    imdsTest  = subset(imds, testIdx);

    % Return sorted list of varieties (class names)
    varieties = categories(imdsTrain.Labels);

end

function [features, empmean, CRFDATA, boxesalldata] = computeBBoxBfeature(features, empmean, imlist, CRFDATA, varname, ind_class, threshold, ALLPATHS, train, boxesalldata)
% detection feature: BBOX -- bi

r = length(features) + 1;
features{r} = [];
features{r}.sample = cell(1, length(imlist));
gamma_score = 1.5;
BinaryStates = 2;
getboxes = 0;
if nargin < 10 || isempty(boxesalldata)
   boxesalldata = cell(length(imlist), 1);
   getboxes = 1;
end;
vars = CRFDATA.vars;
LOC = CRFDATA.COUNT_UNARY;
featparams = CRFDATA.featparams;

for i = 1:length(imlist)
    imname = imlist{i};
    if getboxes
       boxdata = load(fullfile(ALLPATHS.BBOX_PATH,[imname '.mat']));
       boxesalldata{i} = boxdata.lsvmdata;
    else
       boxdata.lsvmdata = boxesalldata{i};
    end;
    
    iclass = geticlass(ALLPATHS.classes, ind_class);
    lsvmdata = boxdata.lsvmdata;
    boxes = lsvmdata(iclass).boxes;
    b = lsvmdata(iclass).b; % b is pascal overlap with groundtruth
    nboxes = size(boxes, 1);
%     
%                                                                     data = load(fullfile(ALLPATHS.SCENE_PATH,[imname '.mat']), 'potential'); %labelSuperpixel
%                                                                     correspondence_file = load (fullfile( ALLPATHS.SCENE_PATH, ['/potential/correspondence_obj_scene.mat']));
%                                                                     correspondence_obj_scene = correspondence_file.correspondence_obj_scene;
%                                                                     U_scene = data.potential;
%                                                                     if featparams.uselogisticScene
%                                                                        U_scene = 1./(1+exp(-gamma_score*(U_scene)));  % logistic score!
%                                                                       end;
     
    if numel(boxes)
        if featparams.uselogistic
           scores = 1./(1+exp(-gamma_score*(boxes(:,end)+threshold)));
        else
           scores = boxes(:, end) + threshold;
        end;
        score = [zeros(nboxes, 1), scores];
        U = score;
%                                                                         U = U.*U_scene(correspondence_obj_scene(iclass));
        %check
        if min(score(:)) < 0
            fprintf('Error:Score cannot be negative\n');
        end
        % Compute Emperical Means
        if train
           labelBox = double(b(:, 1) >= featparams.boxthresh)+1;
	       empmean(r) = computeEmpiricalMeans(empmean(r), U, labelBox, 1);
        end;
    end;

    % unary setting
    [~, blockstart, ~] = getvardata(vars, varname, i);
    nstates = nboxes * BinaryStates;
    if blockstart == 0 && nboxes > 0, 
        [vars, ~, blockstart, ~] = setvardata(vars, varname, i, LOC(i) + 1, LOC(i) + nboxes, nstates); 
    end;
    pntr = blockstart;
    for s = 1 : nboxes
        features{r}.sample{i}.local{pntr}.NumStates = BinaryStates;
        features{r}.sample{i}.local{pntr}.pot = U(s,:)';
        pntr = pntr + 1;
    end
    LOC(i) = LOC(i) + nboxes;
    
    % pairwise setting
    features{r}.sample{i}.factor= cell(0,0);
end

CRFDATA = updateData(CRFDATA, vars, LOC);

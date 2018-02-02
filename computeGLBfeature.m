function [features, empmean, CRFDATA] = computeGLBfeature(features, empmean, imlist, CRFDATA, ALLPATHS, train)
% SUPERPIXEL FEATURE

r = length(features) + 1;
rstart = r;
features{r} = [];
features{r}.sample = cell(1, length(imlist));
varname = 'SP';
NumStates = CRFDATA.NumClasses;
vars = CRFDATA.vars;
LOC = CRFDATA.COUNT_UNARY;
M = CRFDATA.COUNT_FACTORS;
featparams = CRFDATA.featparams;

for i = 1:length(imlist)
    imname = imlist{i};
    I = load(fullfile(ALLPATHS.PIXEL_PATH,[imname '.mat'])); %labelSuperpixel
    P1 = I.potential;
    U1 = zeros(size(P1));
    
    data = load(fullfile(ALLPATHS.SCENE_PATH,[imname '.mat']), 'potential'); % potential for each labele of scene
    correspondence_file = load (fullfile( ALLPATHS.SCENE_PATH, ['/potential/correspondence_obj_scene.mat']));
    correspondence_obj_scene = correspondence_file.correspondence_obj_scene;
    U_scene = data.potential;
    for iclass = 1:NumStates
        if correspondence_obj_scene(iclass) ~= -1
            U1(:, iclass) = U_scene(correspondence_obj_scene(iclass)); 
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if train
       labelSP = CRFDATA.labelsSP{i};
       row = CRFDATA.indsSP{i};
       U1 = U1(row,:);      
    else
        labelSP = [];
    end;
    
    [~, blockstart, ~] = getvardata(vars, varname, i);
    
    % Compute Emperical Means and SP features
    r = rstart;
    empmean = getempmean(empmean, r, U1, labelSP, train);
    features = addpot(features, r, i, NumStates, blockstart, U1, length(imlist));
end   

CRFDATA = updateData(CRFDATA, vars, LOC, M);
%--------------------------------------- 


function features = addpot(features, r, i, NumStates, blockstart, U, nimages)

if r > length(features)
    features{r} = [];
    features{r}.sample = cell(1, nimages);
end;
if ~isfield(features{r}.sample{i}, 'local')
    local = cell(1, blockstart + size(U, 1) - 1);
else
    local = features{r}.sample{i}.local;
end;
numloc = length(local);
if  numloc < blockstart + size(U, 1) - 1
    local = [local, cell(1, blockstart + size(U, 1) - 1 - numloc)];
end;

pntr = blockstart;
for s = 1:size(U, 1)
    local{pntr}.NumStates = NumStates;
    local{pntr}.pot = U(s,:)';
    pntr = pntr + 1;
end
features{r}.sample{i}.local = local;
% pairwise setting
features{r}.sample{i}.factor= cell(0,0);


function empmean = getempmean(empmean, r, U, labelSP, train)

if train
    empmean(r) = computeEmpiricalMeans(empmean(r), U, labelSP, 1);
end;

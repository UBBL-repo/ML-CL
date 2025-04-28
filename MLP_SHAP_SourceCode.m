clear; clc; close all;
load Shap_all_sourcedata.mat

%%
% Sort the predictors by mean(abs(shap))
[sortedMeanAbsShapValues,sortedPredictorIndices]=sort(mean(abs(shap)));
% Loop over the predictors, plot a row of points for each predictor using
% the scatter function with "density" jitter.
% The multiple calls to scatter are needed so that the jitter is normalized
% per-row, rather than globally over all the rows.
for p=1:d
   scatter(shap(:,sortedPredictorIndices(p)), ... % x-value of each point is the shapley value
           p*ones(N,1), ... % y-value of each point is an integer corresponding to a predictor (to be jittered below)
           [], ... % Marker size for each data point, taking the default here
           normalize(U_train(samples,sortedPredictorIndices(p)),'range',[1 256]), ... % Colors based on feature values
           'filled', ... % Fills the circles representing data points
           'YJitter','density', ... % YJitter according to the density of the points in this row
           'YJitterWidth',0.8)
   if (p==1) hold on; end
end
title('MLP Shapley Summary plot');
xlabel('Shapley Value (impact on model output)')
yticks([1:d]);
FeatureNames = {'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', ...
    'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', ...
    'Targetx', 'Targety', 'MB Kinetic', 'numPulse', 'isTumor', 'Pressure'};
yticklabels(FeatureNames(sortedPredictorIndices));
% Set colormap as desired 
colormap(CoolBlueToWarmRedColormap); % This colormap is like the one used in many Shapley summary plots
% colormap(parula); % This is the default colormap
cb= colorbar('Ticks', [1 256], 'TickLabels', {'Low', 'High'});
cb.Label.String = "Scaled Feature Value";
cb.Label.FontSize = 12;
cb.Label.Rotation = 270;
set(gca, 'YGrid', 'on');
xline(0, 'LineWidth', 1);
hold off;

%% Global feature Importance
mean_shap = mean(abs(shap));
figure;
bar(mean_shap);
title('MLP Shapley global importance plot');
ylabel('Mean(|SHAP Value|)')
xticks([1:d]);
xticklabels(FeatureNames);
set(gca, 'YGrid', 'on');


function colormap = CoolBlueToWarmRedColormap()

% Define start point, middle luminance, and end point in L*ch colorspace
% https://www.mathworks.com/help/images/device-independent-color-spaces.html
% The three components of L*ch are Luminance, chroma, and hue.
blue_lch = [54 70 4.6588]; % Starting blue point
l_mid = 40; % luminance of the midpoint
red_lch = [54 90 6.6378909]; % Ending red point

nsteps = 256;

% Build matrix of L*ch colors that is nsteps x 3 in size
% Luminance changes linearly from start to middle, and middle to end.
% Chroma and hue change linearly from start to end.
lch=[[linspace(blue_lch(1), l_mid, nsteps/2), linspace(l_mid, red_lch(1), nsteps/2)]', ... luminance column
    [linspace(blue_lch(2), red_lch(2), nsteps)]', ... chroma column
    [linspace(blue_lch(3), red_lch(3), nsteps)]']; ... hue column

% Convert L*ch to L*a*b, where a = c * cos(h) and b = c * sin(h)
lab=[lch(:,1) lch(:,2).*cos(lch(:,3)) lch(:,2).*sin(lch(:,3))];

% Convert L*a*b to RGB
colormap=lab2rgb(lab,'OutputType','uint8');

end

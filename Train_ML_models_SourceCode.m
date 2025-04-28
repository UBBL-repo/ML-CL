clear; clc; close all;

% Data loading (replace with actual data file)
load 500kHzNC.mat;

% Preprocessing steps (log transform, normalization, etc.)
T.('Concentration') = concentration;
T.H2_500kHz = log10(T.H2_500kHz); T.H3_500kHz = log10(T.H3_500kHz); 
T.H4_500kHz = log10(T.H4_500kHz); T.H5_500kHz = log10(T.H5_500kHz);
T.H6_500kHz = log10(T.H6_500kHz); T.H7_500kHz = log10(T.H7_500kHz); 
T.H8_500kHz = log10(T.H8_500kHz); T.U2_500kHz = log10(T.U2_500kHz);
T.U3_500kHz = log10(T.U3_500kHz); T.U4_500kHz = log10(T.U4_500kHz); 
T.U5_500kHz = log10(T.U5_500kHz); T.U6_500kHz = log10(T.U6_500kHz);
T.U7_500kHz = log10(T.U7_500kHz); T.U8_500kHz = log10(T.U8_500kHz);
T.BB_500kHz = log10(T.BB_500kHz);

T.isControlled =[]; T.whatfrequency = [];
T.BB_1500kHz = []; T.H2_1500kHz = []; T.H3_1500kHz = [];
T.H4_1500kHz = []; T.U2_1500kHz = []; T.U3_1500kHz = []; T.U4_1500kHz = [];

% Define inputs & Labels
U = [T.U2_500kHz T.U3_500kHz T.U4_500kHz T.U5_500kHz T.U6_500kHz T.U7_500kHz T.U8_500kHz ...
    T.Targetx T.Targety T.Concentration T.pressure T.isTumor];

% U = [T.H2_500kHz T.H3_500kHz T.H4_500kHz T.H5_500kHz T.H6_500kHz T.H7_500kHz T.H8_500kHz ...
%     T.U2_500kHz T.U3_500kHz T.U4_500kHz T.U5_500kHz T.U6_500kHz T.U7_500kHz T.U8_500kHz ...
%     T.Targetx T.Targety T.numPulse T.Concentration T.pressure T.isTumor];

%optional normalization
max_vals = max(U);
min_vals = min(U);
[U] = normalize(U, 1, 'range');

Y = T.BB_500kHz > 3.8;
T.BB_500kHz = [];  % Remove label

% Initialize metrics storage
num_algorithms = 4; % Linear SVM, RBF SVM, MLP with Cross-Entropy, MLP with MSE
metrics_smote = zeros(num_algorithms, 5); % For SMOTE
metrics_no_smote = zeros(num_algorithms, 5); % For No SMOTE (undersampling)

% Train-Test Split
cv = cvpartition(size(U, 1), 'HoldOut', 0.2);
train_idx = ~cv.test;
U_train = U(train_idx, :);
Y_train = Y(train_idx);
U_test = U(~train_idx, :);
Y_test = Y(~train_idx);

%% SMOTE
disp('Applying SMOTE...');
[U_smote, Y_smote] = smote(U_train, [5, 1], 'Class', ~logical(Y_train)); % Balance the dataset
Y_smote = ~Y_smote;

%% No SMOTE (Undersampling)
disp('Applying undersampling...');
Truth_ind = find(Y_train == 1);
False_ind = find(Y_train == 0);
U_true = U_train(Truth_ind, :);
Y_true = Y_train(Truth_ind, :);
U_false = U_train(False_ind, :);
Y_false = Y_train(False_ind, :);
undersample_idx = randsample(size(U_false, 1), size(U_true, 1)); % Match minority class size
U_undersample = [U_true; U_false(undersample_idx, :)];
Y_undersample = [Y_true; Y_false(undersample_idx, :)];

%% Train and Evaluate Algorithms
algorithms = {'Linear SVM', 'RBF SVM', 'MLP CE', 'MLP MSE'};

for i = 1:num_algorithms
    switch i
        case 1  % Linear SVM
            disp('Training Linear SVM...');
            mdl_smote = fitcsvm(U_smote, Y_smote, 'KernelFunction', 'linear');
            mdl_no_smote = fitcsvm(U_undersample, Y_undersample, 'KernelFunction', 'linear');
            % save('Linear_SVM.mat', 'mdl_no_smote','min_vals', 'max_vals')
            % save('Linear_SVM_SMOTE.mat', 'mdl_smote','min_vals', 'max_vals')
        case 2  % RBF SVM
            disp('Training RBF SVM...');
            mdl_smote = fitcsvm(U_smote, Y_smote, 'KernelFunction', 'rbf');
            mdl_no_smote = fitcsvm(U_undersample, Y_undersample, 'KernelFunction', 'rbf');
            % save('RBF_SVM.mat', 'mdl_no_smote','min_vals', 'max_vals')
            % save('RBF_SVM_SMOTE.mat', 'mdl_smote','min_vals', 'max_vals')          
        case 3  % MLP with Cross-Entropy
            disp('Training MLP with Cross-Entropy...');
            net_smote = patternnet(10);
            net_smote.trainParam.epochs = 1000;
            net_smote = train(net_smote, U_smote', Y_smote');
            net_no_smote = patternnet(10);
            net_no_smote.trainParam.epochs = 1000;
            net_no_smote = train(net_no_smote, U_undersample', Y_undersample');
            % save('MLP_CE.mat', 'net_no_smote','min_vals', 'max_vals')
            % save('MLP_CE_SMOTE.mat', 'net_smote','min_vals', 'max_vals')               
        case 4  % Standard MLP with MSE
            disp('Training Standard MLP with MSE...');
            net_smote = feedforwardnet(10);
            net_smote.trainParam.epochs = 1000;
            net_smote = train(net_smote, U_smote', Y_smote');
            net_no_smote = feedforwardnet(10);
            net_no_smote.trainParam.epochs = 1000;
            net_no_smote = train(net_no_smote, U_undersample', Y_undersample');
            % save('MLP_MSE.mat', 'net_no_smote','min_vals', 'max_vals')
            % save('MLP_MSE_SMOTE.mat', 'net_smote','min_vals', 'max_vals')  
    end

    % Predictions and metrics
switch i
    case {1, 2}  % Linear SVM, RBF SVM, Boosting
        Y_pred_smote = predict(mdl_smote, U_test);
        Y_pred_no_smote = predict(mdl_no_smote, U_test);
        Y_train_pred_smote = predict(mdl_smote, U_smote);
        Y_train_pred_no_smote = predict(mdl_no_smote, U_undersample);

    case {3, 4}  % MLPs
        Y_pred_smote = net_smote(U_test') > 0.5;
        Y_pred_no_smote = net_no_smote(U_test') > 0.5;
        Y_train_pred_smote = net_smote(U_smote') > 0.5;
        Y_train_pred_no_smote = net_no_smote(U_undersample') > 0.5;
end

    % Store metrics
    metrics_smote(i, :) = get_metrics(Y_test, Y_pred_smote, algorithms{i}, 'SMOTE', true);
    metrics_no_smote(i, :) = get_metrics(Y_test, Y_pred_no_smote, algorithms{i}, 'Undersampling', true);
    metrics_train_smote(i, :) = get_metrics(Y_smote, Y_train_pred_smote, algorithms{i}, 'SMOTE', true);
    metrics_train_no_smote(i, :) = get_metrics(Y_undersample, Y_train_pred_no_smote, algorithms{i}, 'Undersampling', true);
end

%% Visualization
metric_names = {'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1 Score'};

fig = figure('Units', 'inches', 'Position', [0 0 8.5 11]);
t = tiledlayout(3, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% Indices to include: [1, 2, 5, 6] → Linear SVM, RBF SVM, MLP CE, MLP MSE
include_idx = [1, 2, 3, 4];
filtered_algorithms = algorithms(include_idx);

for m = 1:5
    nexttile;
    bar([metrics_smote(include_idx, m), metrics_no_smote(include_idx, m)]);
    set(gca, 'XTickLabel', filtered_algorithms);
    xlabel('Algorithms');
    ylabel(metric_names{m});
    title(['SMOTE vs Undersampling - ' metric_names{m}]);
    legend('SMOTE', 'Undersampling');
    xtickangle(45);
    grid on;
end

% Leave one tile blank or remove it
nexttile(6);
axis off;

% Save once for all tiles
% saveas(fig, 'All_Metrics_Tiled.svg');

% Helper function to compute performance metrics
function metrics = get_metrics(Y_true, Y_pred, algo_name, dataset_type, save_fig)
    % Ensure matching types
    Y_true = logical(Y_true);
    Y_pred = logical(Y_pred);

    % Compute confusion matrix
    C = confusionmat(Y_true, Y_pred);

    sensitivity = C(2, 2) / (C(2, 2) + C(2, 1));
    specificity = C(1, 1) / (C(1, 1) + C(1, 2));
    accuracy = (C(1, 1) + C(2, 2)) / sum(C(:));
    precision = C(2, 2) / (C(2, 2) + C(1, 2));
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity);
    metrics = [accuracy, precision, sensitivity, specificity, f1_score];

    % Optional plotting
    if nargin > 2 && save_fig
        persistent tile_idx tfig tlayout
        if isempty(tile_idx) || tile_idx > 24
            tile_idx = 1;
            tfig = figure('Units','inches','Position',[0 0 8.5 11]);
            tlayout = tiledlayout(6, 4, 'Padding', 'compact', 'TileSpacing', 'compact');
        end
        figure(tfig);
        nexttile(tile_idx);
        confusionchart(logical(Y_true), logical(Y_pred), ...
            'Title', sprintf('%s (%s)', algo_name, dataset_type));
        tile_idx = tile_idx + 1;
    end
end

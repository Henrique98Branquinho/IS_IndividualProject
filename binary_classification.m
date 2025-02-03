%% Load data and create train-test sets
Train = readtable('train_data.csv');
Test = readtable('test_data.csv');

% Split features (X) and labels (Y)
X_train = table2array(Train(:, 1:end-1));  % All columns except 'STATUS'
Y_train = table2array(Train(:, end));     % 'STATUS' column as label
X_test = table2array(Test(:, 1:end-1));   % All columns except 'STATUS'
Y_test = table2array(Test(:, end));      % 'STATUS' column as label

%% 🛠 Step 1: Remove Features with Zero Variance
var_X = var(X_train);  % Compute variance of each feature
zero_variance_idx = find(var_X == 0);  % Find indices of zero-variance features

if ~isempty(zero_variance_idx)
    fprintf("Removing %d zero-variance features...\n", length(zero_variance_idx));
    X_train(:, zero_variance_idx) = [];
    X_test(:, zero_variance_idx) = []; % Remove from test set as well
end

%% 🛠 Step 2: Handle NaN and Inf Values
% Replace NaN with column mean
for i = 1:size(X_train,2)  % Loop over columns
    X_train(isnan(X_train(:,i)), i) = mean(X_train(:,i), 'omitnan');
end

for i = 1:size(X_test,2)
    X_test(isnan(X_test(:,i)), i) = mean(X_test(:,i), 'omitnan');
end


% Replace Inf values with max finite value
for i = 1:size(X_train,2)
    X_train(isinf(X_train(:,i)), i) = max(X_train(~isinf(X_train(:,i)), i));
end

for i = 1:size(X_test,2)
    X_test(isinf(X_test(:,i)), i) = max(X_test(~isinf(X_test(:,i)), i));
end

%% Train initial Takagi-Sugeno model
opt = genfisOptions('FCMClustering','FISType','sugeno');
opt.NumClusters = 5;
ts_model = genfis(X_train,Y_train,opt);

%% Check initial performance on test set
Y_pred_initial = evalfis(ts_model, X_test);
Y_pred_initial(Y_pred_initial>=0.5) = 1;
Y_pred_initial(Y_pred_initial<0.5) = 0;
class_report_initial = classperf(Y_test, Y_pred_initial);
fprintf('Initial Accuracy: %4.3f \n', class_report_initial.CorrectRate);
fprintf('Initial Sensitivity: %4.3f \n', class_report_initial.Sensitivity);
fprintf('Initial Specificity: %4.3f \n', class_report_initial.Specificity);

%% Tune initial model using ANFIS
[in,out,rule] = getTunableSettings(ts_model);
anfis_model = tunefis(ts_model,[in;out],X_train,Y_train,tunefisOptions("Method","anfis"));

%% Check ANFIS tuned model performance
Y_pred_final = evalfis(anfis_model, X_test);
Y_pred_final(Y_pred_final>=0.5) = 1;
Y_pred_final(Y_pred_final<0.5) = 0;
class_report_final = classperf(Y_test, Y_pred_final);
fprintf('Final Accuracy: %4.3f \n', class_report_final.CorrectRate);
fprintf('Final Sensitivity: %4.3f \n', class_report_final.Sensitivity);
fprintf('Final Specificity: %4.3f \n', class_report_final.Specificity);
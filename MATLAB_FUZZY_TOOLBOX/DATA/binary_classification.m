%% Load data and create train-test sets
Train = readtable('train_data.csv');
Test = readtable('test_data.csv');

% Split features (X) and labels (Y)
X_train = table2array(Train(:, 1:end-1));  % All columns except 'STATUS'
Y_train = table2array(Train(:, end));     % 'STATUS' column as label
X_test = table2array(Test(:, 1:end-1));   % All columns except 'STATUS'
Y_test = table2array(Test(:, end));      % 'STATUS' column as label


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
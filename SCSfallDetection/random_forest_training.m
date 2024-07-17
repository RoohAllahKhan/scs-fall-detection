opts = detectImportOptions('Train.csv');
opts.VariableTypes = {'double', 'double', 'double', 'double', 'double', 'char', 'double', 'double', 'double', 'double', 'double', 'double'};

% Step 1: Load and preprocess the dataset
trainData = readtable('Train.csv', opts);
testData = readtable('Test.csv', opts);

% Handle missing values in training data
for col = 1:width(trainData)
    if isnumeric(trainData{:, col})
        trainData{isnan(trainData{:, col}), col} = mean(trainData{~isnan(trainData{:, col}), col});
    end
end

% Handle missing values in test data
for col = 1:width(testData)
    if isnumeric(testData{:, col})
        testData{isnan(testData{:, col}), col} = mean(testData{~isnan(testData{:, col}), col});
    end
end

% Encode categorical variable 'label' into numeric labels
trainData.label = categorical(trainData.label);
testData.label = categorical(testData.label);

% Step 2: Select predictors and response
X_train = trainData{:, {'acc_max', 'gyro_max', 'acc_kurtosis', 'gyro_kurtosis', 'lin_max', 'acc_skewness', 'gyro_skewness', 'post_gyro_max', 'post_lin_max'}};
y_train = categorical(trainData.fall);

X_test = testData{:, {'acc_max', 'gyro_max', 'acc_kurtosis', 'gyro_kurtosis', 'lin_max', 'acc_skewness', 'gyro_skewness', 'post_gyro_max', 'post_lin_max'}};
y_test = categorical(testData.fall);

% Step 3: Train the Random Forest model
numTrees = 100; % Number of trees in the forest
trainAccuracyHistory = zeros(1, numTrees);
testAccuracyHistory = zeros(1, numTrees);

% Train the Random Forest model and record accuracy over iterations
for t = 1:numTrees
    mdl = TreeBagger(t, X_train, y_train, 'Method', 'classification');

    % Training accuracy
    trainPredictions = categorical(predict(mdl, X_train));
    trainAccuracy = sum(trainPredictions == y_train) / numel(y_train);
    trainAccuracyHistory(t) = trainAccuracy;

    % Testing accuracy
    predictedLabels = categorical(predict(mdl, X_test));
    testAccuracy = sum(predictedLabels == y_test) / numel(y_test);
    testAccuracyHistory(t) = testAccuracy;
end
save('fall_detection_model.mat', 'mdl');

% mdl = TreeBagger(numTrees, X_train, y_train, 'Method', 'classification');

% Optionally, evaluate the model on the training data
% trainPredictions = predict(mdl, X_train);
% trainPredictions = categorical(trainPredictions);
% y_train = categorical(y_train);

% trainAccuracy = sum(trainPredictions == y_train) / numel(y_train);
disp(['Training Accuracy: ', num2str(trainAccuracy)]);

% Step 4: Validate the model on test data
% predictedLabels = predict(mdl, X_test);
% predictedLabels = categorical(predictedLabels);
% y_test = categorical(y_test);

% Evaluate performance (e.g., accuracy, confusion matrix)
accuracy = sum(predictedLabels == y_test) / numel(y_test);
confMat = confusionmat(y_test, predictedLabels);
disp(['Test Accuracy: ', num2str(accuracy)]);
disp('Confusion Matrix:');
disp(confMat);

% Step 5: Plotting (example: Confusion Matrix)
figure;
confusionchart(confMat);
title('Confusion Matrix');

% Additional plotting (ROC curve, etc.) can be added based on your preference
% Step 5: Plotting
figure;

% Plot training accuracy
subplot(1, 2, 1);
bar({'Training', 'Testing'}, [trainAccuracy, accuracy]);
ylim([0, 1]);
ylabel('Accuracy');
title('Training vs Testing Accuracy');
grid on;

% Plot confusion matrix
subplot(1, 2, 2);
confusionchart(confMat);
title('Confusion Matrix');

% Adjust subplot parameters for better visualization
sgtitle('Model Performance');

% Plotting line graph for training and testing accuracy
figure;
plot(1:numTrees, trainAccuracyHistory, 'o-', 'LineWidth', 2, 'DisplayName', 'Training Accuracy');
hold on;
plot(1:numTrees, testAccuracyHistory, 's-', 'LineWidth', 2, 'DisplayName', 'Testing Accuracy');
hold off;
xlabel('Number of Trees');
ylabel('Accuracy');
title('Training and Testing Accuracy vs Number of Trees');
grid on;
legend('show');
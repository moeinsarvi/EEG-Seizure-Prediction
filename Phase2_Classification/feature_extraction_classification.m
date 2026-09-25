clc;
clear all;
close all;

data0 = edfread("chb01_02.edf")
data1 = edfread("chb01_03.edf")
data2 = edfread("chb01_04.edf")
data3 = edfread("chb01_15.edf")
data4 = edfread("chb01_16.edf")
data5 = edfread("chb01_18.edf")
data6 = edfread("chb01_26.edf")


%% PSD
clc
signal1_firstChannel = reshape([data1.SignalLabel1_FP1_F7{:}], [], 1);

epoch_lenght = 16 * 256;
firstEpoch1 = signal1_firstChannel(1:16*256);
window = hamming(16*256);
noverlap = floor(16*256 / 5);
nfft = 2^nextpow2(16*256);
[psd, freq] = periodogram(firstEpoch1, window, nfft, 256, 'psd');

figure;
plot(freq, 10 * log10(psd));
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
title('PSD of $$\textit{chb01\_03}$$ First Channel', 'Interpreter', 'latex');
grid on;


%% PSD for All Channels

channelLabels = { ...
    'SignalLabel1_FP1_F7', 'SignalLabel2_F7_T7', 'SignalLabel3_T7_P7', 'SignalLabel4_P7_O1', ...
    'SignalLabel5_FP1_F3', 'SignalLabel6_F3_C3', 'SignalLabel7_C3_P3', 'SignalLabel8_P3_O1', ...
    'SignalLabel9_FP2_F4', 'SignalLabel10_F4_C4', 'SignalLabel11_C4_P4', 'SignalLabel12_P4_O2', ...
    'SignalLabel13_FP2_F8', 'SignalLabel14_F8_T8', 'SignalLabel15_T8_P8', 'SignalLabel16_P8_O2', ...
    'SignalLabel17_FZ_CZ', 'SignalLabel18_CZ_PZ', 'SignalLabel19_P7_T7', 'SignalLabel20_T7_FT9', ...
    'SignalLabel21_FT9_FT10', 'SignalLabel22_FT10_T8', 'SignalLabel23_T8_P8' ...
};

fs = 256;
epoch_length = 16 * fs;

window = hamming(epoch_length);
noverlap = floor(epoch_length / 5);
nfft = 2^nextpow2(epoch_length);

signal1_allChannels = reshape([data1.(channelLabels{1}){:}], [], 1);

for i = 2:length(channelLabels)
    signal1_allChannels = signal1_allChannels + reshape([data1.(channelLabels{i}){:}], [], 1);    
end

firstEpoch = signal1_allChannels(1:epoch_length);

[psd, freq] = periodogram(firstEpoch, window, nfft, fs, 'psd');

figure;
plot(freq, 10 * log10(psd));
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
title('PSD of $$\textit{chb01\_03}$$ All Channels', 'Interpreter', 'latex');
grid on;

%% Shanon Entropy for One Epoch
clc

samplingRate = 256;
epochDuration = 16;
numBins = 256;

rawSignal = cell2mat(data1.SignalLabel1_FP1_F7);
signalVector = rawSignal(:);

epochSampleCount = epochDuration * samplingRate;
epochSignal = signalVector(1:epochSampleCount);
[frequencyCounts, ~] = histcounts(epochSignal, numBins);

signalProbabilities = frequencyCounts / sum(frequencyCounts);
signalProbabilities(signalProbabilities == 0) = [];

shannonEntropy = -sum(signalProbabilities .* log2(signalProbabilities));
fprintf('Shannon entropy of the first %d-second epoch: %f\n', epochDuration, shannonEntropy);

%% Shanon Entropy for All Epochs
flattenedData = cell2mat(data1.SignalLabel1_FP1_F7);
flattenedData = flattenedData(:);

samplesPerEpoch = 16 * 256;
totalEpochs = floor(length(flattenedData) / samplesPerEpoch);

shannonEntropy = zeros(1, totalEpochs);

for epochIndex = 1:totalEpochs
    startIdx = (epochIndex - 1) * samplesPerEpoch + 1;
    endIdx = epochIndex * samplesPerEpoch;
    epochData = flattenedData(startIdx:endIdx);
    
    binCount = 256;
    histogramCounts = histcounts(epochData, binCount);
    probDist = histogramCounts / sum(histogramCounts);
    probDist(probDist == 0) = [];
    
    shannonEntropy(epochIndex) = -sum(probDist .* log2(probDist));
end

figure;
plot(1:totalEpochs, shannonEntropy, '-o', 'Color', [0 0.5 0.5], 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y');
xlabel('Epoch Number');
ylabel('Shannon Entropy');
grid on;

%% Feature extraction
clc;
data = data1.SignalLabel1_FP1_F7;
EEG = cell2mat(data);
EEG = EEG(:);

Fs = 256; 
epochSec = 16;
epochSamp = epochSec * Fs;
totalSec = 10 * 60;
totalSamp = totalSec * Fs;
numEpochs = floor(totalSamp / epochSamp);

epochLabels = randi([0, 1], numEpochs, 1);

features = zeros(numEpochs, 5);

for idx = 1:numEpochs
    startIdx = (idx - 1) * epochSamp + 1;
    endIdx = idx * epochSamp;
    epochData = EEG(startIdx:endIdx);
    
    fftData = fft(epochData);
    
    psdData = (1/(Fs * epochSamp)) * abs(fftData).^2;
    psdData = psdData(1:floor(epochSamp / 2) + 1);
    psdData(2:end-1) = 2 * psdData(2:end-1);
    
    psdData = psdData / sum(psdData);
    
    psdData(psdData == 0) = [];
    entropy = -sum(psdData .* log2(psdData));
    
    meanVal = mean(epochData);
    stdVal = std(epochData);
    minVal = min(epochData);
    maxVal = max(epochData);
    
    features(idx, :) = [meanVal, stdVal, minVal, maxVal, entropy];
end

featureTable = array2table(features, 'VariableNames', {'Mean', 'StdDev', 'Min', 'Max', 'ShannonEntropy'});

disp(featureTable);

%% Feature Selection 
clc;
mu = 0;
numFeat = size(features, 2);
pVals = zeros(1, numFeat);
tVals = zeros(1, numFeat);

for i = 1:numFeat
    [~, p, ~, stats] = ttest(features(:, i), mu, 'Alpha', 0.001);
    pVals(i) = p;
    tVals(i) = stats.tstat;
end

selFeat = features(:, pVals < 0.001);

disp('P-values:');
disp(pVals);
disp('T-statistics:');
disp(tVals);
disp('Selected features:');
disp(selFeat);


%% SVM
clc;
cv = cvpartition(epochLabels, 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

X_train = features(trainIdx, :);
y_train = epochLabels(trainIdx); 
X_test = features(testIdx, :); 
y_test = epochLabels(testIdx); 

SVMModel = fitcsvm(X_train, y_train, 'KernelFunction', 'linear');

y_pred = predict(SVMModel, X_test);

confMat = confusionmat(y_test, y_pred);

tp = confMat(1,1);
tn = confMat(2,2);
fp = confMat(2,1);
fn = confMat(1,2);

accuracy = (tp + tn) / sum(confMat(:));
sensitivity = tp / (tp + fn); 
specificity = tn / (tn + fp);

numTestSamples = size(X_test, 1);
tic;
for i = 1:numTestSamples
    predict(SVMModel, X_test(i, :));
end
latency = toc / numTestSamples;

disp('Confusion Matrix:');
disp(confMat);
disp(['Accuracy: ', num2str(accuracy)]);
disp(['Sensitivity: ', num2str(sensitivity)]);
disp(['Specificity: ', num2str(specificity)]);
disp(['Latency (average prediction time): ', num2str(latency), ' seconds']);

%% SVM Reproducable
clc;
rng(1);
cv = cvpartition(epochLabels, 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

X_train = features(trainIdx, :);
y_train = epochLabels(trainIdx); 
X_test = features(testIdx, :); 
y_test = epochLabels(testIdx); 

SVMModel = fitcsvm(X_train, y_train, 'KernelFunction', 'linear');

y_pred = predict(SVMModel, X_test);

confMat = confusionmat(y_test, y_pred);

tp = confMat(1,1);
tn = confMat(2,2);
fp = confMat(2,1);
fn = confMat(1,2);
accuracy = (tp + tn) / sum(confMat(:));
sensitivity = tp / (tp + fn); 
specificity = tn / (tn + fp);
numTestSamples = size(X_test, 1);
tic;
for i = 1:numTestSamples
    predict(SVMModel, X_test(i, :));
end
latency = toc / numTestSamples;

disp('Confusion Matrix:');
disp(confMat);
disp(['Accuracy: ', num2str(accuracy)]);
disp(['Sensitivity: ', num2str(sensitivity)]);
disp(['Specificity: ', num2str(specificity)]);
disp(['Latency (average prediction time): ', num2str(latency), ' seconds']);

%% KNN
clc;
cv = cvpartition(epochLabels, 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

X_train = features(trainIdx, :); 
y_train = epochLabels(trainIdx); 
X_test = features(testIdx, :);
y_test = epochLabels(testIdx);

KNNModel = fitcknn(X_train, y_train, 'Distance', 'minkowski', 'NumNeighbors', 5);

y_pred = predict(KNNModel, X_test);

confMat = confusionmat(y_test, y_pred);

tp = confMat(1,1);
tn = confMat(2,2);
fp = confMat(2,1);
fn = confMat(1,2);

accuracy = (tp + tn) / sum(confMat(:));
sensitivity = tp / (tp + fn);
specificity = tn / (tn + fp);

numTestSamples = size(X_test, 1);
tic;
for i = 1:numTestSamples
    predict(KNNModel, X_test(i, :));
end
latency = toc / numTestSamples;

disp('Confusion Matrix:');
disp(confMat);
disp(['Accuracy: ', num2str(accuracy)]);
disp(['Sensitivity: ', num2str(sensitivity)]);
disp(['Specificity: ', num2str(specificity)]);
disp(['Latency (average prediction time): ', num2str(latency), ' seconds']);

%% KNN Reproducable
clc;
rng(1);
cv = cvpartition(epochLabels, 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

X_train = features(trainIdx, :); 
y_train = epochLabels(trainIdx); 
X_test = features(testIdx, :); 
y_test = epochLabels(testIdx); 

KNNModel = fitcknn(X_train, y_train, 'Distance', 'minkowski', 'NumNeighbors', 5);

y_pred = predict(KNNModel, X_test);

confMat = confusionmat(y_test, y_pred);

tp = confMat(1,1);
tn = confMat(2,2); 
fp = confMat(2,1); 
fn = confMat(1,2);

accuracy = (tp + tn) / sum(confMat(:)); 
sensitivity = tp / (tp + fn); 
specificity = tn / (tn + fp); 


numTestSamples = size(X_test, 1);
tic;
for i = 1:numTestSamples
    predict(KNNModel, X_test(i, :));
end
latency = toc / numTestSamples; 

disp('Confusion Matrix:');
disp(confMat);
disp(['Accuracy: ', num2str(accuracy)]);
disp(['Sensitivity: ', num2str(sensitivity)]);
disp(['Specificity: ', num2str(specificity)]);
disp(['Latency (average prediction time): ', num2str(latency), ' seconds']);


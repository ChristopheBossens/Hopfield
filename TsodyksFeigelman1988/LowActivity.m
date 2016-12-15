%% The following code is loosely inspired by:
% Tsodyks & Feigel'man (1988). We explore the capacity of a Hopfield
% network to store low activity patterns, the effect of different threshold
% values on this capacity and why threshold values need to be adjusted in
% the case of low activity patterns
clc;clear;


%% In this code we simulate a network with adjusted and unadjusted pattern
% values. For each network we test the stability of stored patterns when
% different activity values and threshold values are used. We say that a
% network has reached capacity at the moment that at least one of the
% stored patterns is no longer a stable state of the network
networkSize = 100;
nExemplars = 30;

hopfield = Hopfield(networkSize);
hopfield.SetUnitModel('V');
activityLevels = 0.01:0.01:0.99;   % Mean activity of a pattern
thresholdValues = -0.15:.005:0.15;

nThresholdValues = length(thresholdValues);
nActivityValues = length(activityLevels);

adjustedCapacity = zeros(nThresholdValues,nActivityValues);

for activityIndex = 1:nActivityValues
    clc;display(['Testing activity ' num2str(activityIndex) '/' num2str(nActivityValues)]);
    exemplarMatrix = hopfield.GeneratePatternMatrix(nExemplars, activityLevels(activityIndex));
    
    for thresholdIndex = 1:nThresholdValues                   
        % Add patterns with correction for activity levels
        hopfield.ResetWeightMatrix();    
        hopfield.SetThreshold(thresholdValues(thresholdIndex));
        unstablePatternDetected = 0;
        trainingIndex = 1;
        while trainingIndex <= nExemplars && unstablePatternDetected == 0
            hopfield.StorePattern(exemplarMatrix(trainingIndex,:),1/networkSize);
            for testIndex = 1:trainingIndex
                output = hopfield.UpdateState(exemplarMatrix(testIndex,:));
                if sum(exemplarMatrix(testIndex,:)==output) ~= networkSize
                    unstablePatternDetected = 1;
                    adjustedCapacity(thresholdIndex, activityIndex) = trainingIndex-1;
                end
            end
            trainingIndex = trainingIndex + 1;
        end
    end
end
adjustedCapacity = adjustedCapacity./networkSize;
%%
cMin = min(adjustedCapacity(:));
cMax = max(adjustedCapacity(:));
clf,
imshow(adjustedCapacity,[cMin cMax],'initialMagnification','fit')
set(gca,'XTick',[1 round(nActivityValues/2) nActivityValues],'XTickLabel',[activityLevels(1) activityLevels(round(nActivityValues/2)) activityLevels(end)])
set(gca,'YTick',[1 round(nThresholdValues/2) nThresholdValues],'YTickLabel',[thresholdValues(1) thresholdValues(round(nThresholdValues/2)) thresholdValues(end)])
xlabel('Mean pattern activity'),ylabel('Activation threshold'),title('Adjusted patterns'),h2 = colorbar,ylabel(h2,'Capacity')
colormap jet, axis on


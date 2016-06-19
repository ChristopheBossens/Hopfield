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
networkSize = 128;
nExemplars = 64;

hopfieldUnadjusted = Hopfield(networkSize);
hopfieldAdjusted = Hopfield(networkSize);

meanActivityValues = -0.95:0.1:0.95;   % Mean activity of a pattern
thresholdValues = -0.7:.025:0.7;

nThresholdValues = length(thresholdValues);
nActivityValues = length(meanActivityValues);

unadjustedCapacity = zeros(nThresholdValues,nActivityValues);
adjustedCapacity = zeros(nThresholdValues,nActivityValues);

for activityIndex = 1:nActivityValues
    clc;display(['Testing activity ' num2str(activityIndex) '/' num2str(nActivityValues)]);
    % Generate exemplars with given level of activity
    currentActivityLevel = meanActivityValues(activityIndex);
    exemplarMatrix = zeros(nExemplars,networkSize);
    for exemplarIndex = 1:nExemplars
        exemplarMatrix(exemplarIndex,:) = hopfieldUnadjusted.GeneratePattern((1+currentActivityLevel)/2);
    end
    
    for thresholdIndex = 1:nThresholdValues
        currentThreshold = thresholdValues(thresholdIndex);
        hopfieldAdjusted.ResetWeights();
        hopfieldUnadjusted.ResetWeights();
        hopfieldAdjusted.SetThreshold(currentThreshold);
        hopfieldUnadjusted.SetThreshold(currentThreshold);
        
        % Train the unadjusted model 
        unstablePatternDetected = 0;
        for trainingIndex = 1:nExemplars
            hopfieldUnadjusted.AddPattern(exemplarMatrix(trainingIndex,:),1/networkSize);
            for testIndex = 1:trainingIndex
                [resp,itUnadjusted] = hopfieldUnadjusted.Converge(exemplarMatrix(testIndex,:));
                if itUnadjusted > 1
                    unstablePatternDetected = 1;
                    unadjustedCapacity(thresholdIndex,activityIndex) = trainingIndex;
                    break;
                end
            end
            if unstablePatternDetected == 1
                break;
            end
        end
        
        % Train the adjusted model
        unstablePatternDetected = 0;
        for trainingIndex = 1:nExemplars
            hopfieldAdjusted.AddPattern(exemplarMatrix(trainingIndex,:)-currentActivityLevel,1/networkSize);
            for testIndex = 1:trainingIndex
                [resp,itAdjusted] = hopfieldAdjusted.Converge(exemplarMatrix(testIndex,:));
                if itAdjusted > 1
                    unstablePatternDetected = 1;
                    adjustedCapacity(thresholdIndex,activityIndex) = trainingIndex;
                    break;
                end
            end
            if unstablePatternDetected == 1
                break;
            end
        end
    end
end
adjustedCapacity = adjustedCapacity./networkSize;
unadjustedCapacity = unadjustedCapacity./networkSize;
%%
cMin = min([adjustedCapacity(:); unadjustedCapacity(:)]);
cMax = max([adjustedCapacity(:); unadjustedCapacity(:)]);
clf,
subplot(2,1,1),imshow(unadjustedCapacity,[cMin cMax]),axis on
set(gca,'XTick',[1 round(nActivityValues/2) nActivityValues],'XTickLabel',[meanActivityValues(1) meanActivityValues(round(nActivityValues/2)) meanActivityValues(end)])
set(gca,'YTick',[1 round(nThresholdValues/2) nThresholdValues],'YTickLabel',[thresholdValues(1) thresholdValues(round(nThresholdValues/2)) thresholdValues(end)])
xlabel('Mean pattern activity'),ylabel('Activation threshold'),title('Unadjusted patterns'),h1 = colorbar,ylabel(h1,'Capacity')
subplot(2,1,2),imshow(adjustedCapacity,[cMin cMax])
set(gca,'XTick',[1 round(nActivityValues/2) nActivityValues],'XTickLabel',[meanActivityValues(1) meanActivityValues(round(nActivityValues/2)) meanActivityValues(end)])
set(gca,'YTick',[1 round(nThresholdValues/2) nThresholdValues],'YTickLabel',[thresholdValues(1) thresholdValues(round(nThresholdValues/2)) thresholdValues(end)])
xlabel('Mean pattern activity'),ylabel('Activation threshold'),title('Adjusted patterns'),h2 = colorbar,ylabel(h2,'Capacity')
colormap jet, axis on

%% The following is a small illustration of why threshold values need to be
% adjusted. For balanced patterns, the overall distribution of synaptic inputs on
% units that should be inactive is clearly separated from the synaptic input 
% distribution of units that should be active. A zero threshold separates both distributions
% However, in the case of sparse activity, the distributions shift and the units that
% need to be inactive receive synaptic input that is above zero.
clear;
networkSize = 300;
nExemplars = 50;

hopfield = Hopfield(networkSize);
hopfield.SetUnitModel('V');
p = 0.2;

exemplarMatrix = zeros(nExemplars,networkSize);
for exemplarIndex = 1:nExemplars
    exemplarMatrix(exemplarIndex,:) = hopfield.GeneratePattern(p);
    hopfield.AddPattern(exemplarMatrix(exemplarIndex,:)-(p),1/networkSize);
end
[activeDistribution, inactiveDistribution, binValues] = hopfield.GetActivityDistribution();

clf,
hold on
bar(binValues,activeDistribution,'g')
bar(binValues,inactiveDistribution,'r')
line([0 0],[0 max([activeDistribution inactiveDistribution])],'Color','k','LineWidth',2,'LineStyle',':')
set(gca,'XLim',[binValues(1) binValues(end)]),xlabel('Input potential'),ylabel('Count')
title(['Active vs inactive unit potentials, p = ' num2str(p)])
legend('Active unit','Inactive unit','Threshold','Location','NorthEastOutside')

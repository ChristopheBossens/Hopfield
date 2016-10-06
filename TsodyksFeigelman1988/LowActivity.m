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
networkSize = 64;
nExemplars = 20;

hopfield = Hopfield(networkSize);
hopfield.SetUnitModel('V');
activityLevels = 0.05:0.05:0.95;   % Mean activity of a pattern
thresholdValues = -0.3:.0125:0.3;

nThresholdValues = length(thresholdValues);
nActivityValues = length(activityLevels);

unadjustedCapacity = zeros(nThresholdValues,nActivityValues);
adjustedCapacity = zeros(nThresholdValues,nActivityValues);

for activityIndex = 1:nActivityValues
    clc;display(['Testing activity ' num2str(activityIndex) '/' num2str(nActivityValues)]);
    exemplarMatrix = hopfield.GeneratePatternMatrix(nExemplars, activityLevels(activityIndex));
    
    for thresholdIndex = 1:nThresholdValues            
        % Add patterns without correcting for activity levels
        hopfield.ResetWeightMatrix();
        unstablePatternDetected = 0;
        trainingIndex = 1;
        while trainingIndex <= nExemplars && unstablePatternDetected == 0
            hopfield.AddPattern(exemplarMatrix(trainingIndex,:)-mean(exemplarMatrix(trainingIndex,:)),1/networkSize);
            for testIndex = 1:trainingIndex
                output = hopfield.Iterate(exemplarMatrix(testIndex,:));
                if sum(exemplarMatrix(testIndex,:)==output) ~= networkSize
                    unstablePatternDetected = 1;
                    unadjustedCapacity(thresholdIndex, activityIndex) = trainingIndex-1;
                end
            end
            trainingIndex = trainingIndex + 1;
        end
        
        % Add patterns with correction for activity levels
        hopfield.ResetWeightMatrix();    
        hopfield.SetThreshold(thresholdValues(thresholdIndex));
        unstablePatternDetected = 0;
        trainingIndex = 1;
        while trainingIndex <= nExemplars && unstablePatternDetected == 0
            hopfield.AddPattern(exemplarMatrix(trainingIndex,:)-mean(exemplarMatrix(trainingIndex,:)),1/networkSize);
            for testIndex = 1:trainingIndex
                output = hopfield.Iterate(exemplarMatrix(testIndex,:));
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
unadjustedCapacity = unadjustedCapacity./networkSize;
%%
cMin = min([adjustedCapacity(:); unadjustedCapacity(:)]);
cMax = max([adjustedCapacity(:); unadjustedCapacity(:)]);
clf,
subplot(2,1,1),imshow(unadjustedCapacity,[cMin cMax]),axis on
set(gca,'XTick',[1 round(nActivityValues/2) nActivityValues],'XTickLabel',[activityLevels(1) activityLevels(round(nActivityValues/2)) activityLevels(end)])
set(gca,'YTick',[1 round(nThresholdValues/2) nThresholdValues],'YTickLabel',[thresholdValues(1) thresholdValues(round(nThresholdValues/2)) thresholdValues(end)])
xlabel('Mean pattern activity'),ylabel('Activation threshold'),title('Unadjusted patterns'),h1 = colorbar,ylabel(h1,'Capacity')
subplot(2,1,2),imshow(adjustedCapacity,[cMin cMax])
set(gca,'XTick',[1 round(nActivityValues/2) nActivityValues],'XTickLabel',[activityLevels(1) activityLevels(round(nActivityValues/2)) activityLevels(end)])
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
networkSize = 100;
nExemplars = 5;

hopfield = Hopfield(networkSize);
hopfield.SetUnitModel('S');
p = 0.5;

exemplarMatrix = hopfield.GeneratePatternMatrix(nExemplars,p);
%hopfield.AddPatternMatrix(exemplarMatrix-mean(exemplarMatrix(1,:)),1);
hopfield.LearnDeltaPatterns(exemplarMatrix,0.5,100);
[distribution, binValues] = hopfield.GetPotentialDistribution();

clf,
hold on
bar(binValues,distribution(1,:),'g')
bar(binValues,distribution(2,:),'r')
line([0 0],[0 max(distribution(:))],'Color','k','LineWidth',2,'LineStyle',':')
set(gca,'XLim',[binValues(1) binValues(end)]),xlabel('Input potential'),ylabel('Count')
title(['Active vs inactive unit potentials, p = ' num2str(p)])
legend('Active unit','Inactive unit','Threshold','Location','NorthEastOutside')
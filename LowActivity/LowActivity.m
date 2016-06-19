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

hopfieldUnadjusted = Hopfield();
hopfieldAdjusted = Hopfield();

meanActivityValues = -0.95:0.01:0.95;   % Mean activity of a pattern
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
        exemplarMatrix(exemplarIndex,:) = hopfieldUnadjusted.GeneratePattern(networkSize,(1+currentActivityLevel)/2);
    end
    
    for thresholdIndex = 1:nThresholdValues
        currentThreshold = thresholdValues(thresholdIndex);
        hopfieldAdjusted.ResetWeights();
        hopfieldUnadjusted.ResetWeights();
        
        % Train the network with a given threshold value
        adjustedStable = 1;
        unadjustedStable = 1; 
        for trainingIndex = 1:nExemplars
            if unadjustedStable == 1
                hopfieldUnadjusted.AddPattern(exemplarMatrix(trainingIndex,:),1/networkSize);
            end
            if adjustedStable == 1
                hopfieldAdjusted.AddPattern(exemplarMatrix(trainingIndex,:)-currentActivityLevel,1/networkSize);
            end
            
            for testIndex = 1:trainingIndex
                if unadjustedStable == 1
                    [resp,itUnadjusted] = hopfieldUnadjusted.Converge(exemplarMatrix(testIndex,:),'async',currentThreshold);
                    if itUnadjusted > 1
                        unadjustedStable = 0;
                        unadjustedCapacity(thresholdIndex,activityIndex) = trainingIndex;
                    end
                end
                if adjustedStable == 1
                    [resp,itAdjusted] = hopfieldAdjusted.Converge(exemplarMatrix(testIndex,:),'async',currentThreshold);
                    if itAdjusted > 1
                        adjustedStable = 0;
                        adjustedCapacity(thresholdIndex,activityIndex) = trainingIndex;
                    end
                end
                
                if unadjustedStable == 0 && adjustedStable == 0;
                    break
                end
            end    
            
            if unadjustedStable == 0 && adjustedStable == 0;
                break
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
% adjusted in low activity patterns.
clear;
networkSize = 300;
nExemplars = 50;

<<<<<<< HEAD
hopfield = Hopfield('V','V');
p = 0.2;
=======
hopfield = Hopfield();
p = 0.1;
>>>>>>> 9b8942cf68153d8972bba1eb3b5b3ecdb034ebaf

exemplarMatrix = zeros(nExemplars,networkSize);
for exemplarIndex = 1:nExemplars
    exemplarMatrix(exemplarIndex,:) = hopfield.GeneratePattern(networkSize,p);
<<<<<<< HEAD
    %hopfield.AddPattern(exemplarMatrix(exemplarIndex,:)-(2*p-1),1/networkSize);
    hopfield.AddPattern(exemplarMatrix(exemplarIndex,:)-(p),1/networkSize);
=======
    hopfield.AddPattern(exemplarMatrix(exemplarIndex,:)-mean(exemplarMatrix(exemplarIndex,:)),7/networkSize);
>>>>>>> 9b8942cf68153d8972bba1eb3b5b3ecdb034ebaf
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

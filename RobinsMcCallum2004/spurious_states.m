%% Set the simulation parameters
clear
networkSize = 32;
nPatterns = 4;
nStateSamples = 1000;
patternActivity = 0.5;

% Run the simulation
hopfield = Hopfield(networkSize);
patternMatrix = hopfield.GeneratePatternMatrix(nPatterns,0.5);

hopfield.ResetWeights();
for patternIndex = 1:nPatterns
    hopfield.AddPattern(patternMatrix(patternIndex,:),1);
end

[stablePatternMatrix, stablePatternCount, meanPatternIterations] = hopfield.SampleWithRandomStates(nStateSamples);
isSpuriousState = hopfield.TestSpuriousStates(stablePatternMatrix,patternMatrix);
nStableStates = length(stablePatternCount);
%% Construct figure 2 of the paper
% Get the sorted unit energy
unitEnergy = zeros(nStableStates,networkSize);

for i = 1:nStableStates
   unitEnergy(i,:) = sort(hopfield.GetUnitEnergy(stablePatternMatrix(i,:))); 
end

mOld = mean(unitEnergy(isSpuriousState==0,:));
mSpurious = mean(unitEnergy(isSpuriousState==1,:));
eOld = std(unitEnergy(isSpuriousState==0,:));
eSpurious = std(unitEnergy(isSpuriousState == 1,:));

% Produce the picture
clf,
hold on
plot(1:networkSize,mOld,'g','LineWidth',2)
plot(1:networkSize,mSpurious,'r', 'LineWidth',2)
errorbar(1:networkSize,mOld,eOld,'.g')
errorbar(1:networkSize,mSpurious,eSpurious,'.r')
xlabel('Unit')
ylabel('Unit energy')
title('Sorted unit energy')
legend('Memory','Spurious','Location','SouthEast')

%% Construct figure 3 of the paper
% Calculate the energy ratio (needs the unit energy from figure 2)
energyRatio = zeros(nStableStates,1);
for i = 1:nStableStates
   energyRatio(i) = sum(unitEnergy(i,1:3))/sum(unitEnergy(i,(end-2):end));
end
[temp,sortedIndex] = sort(isSpuriousState);
patternLabel = cell(1,nStableStates);
lIndex = 1;
sIndex = 1;
for i = 1:nStableStates
    if temp(i) == 0
        patternLabel{i} = ['L' num2str(lIndex)];
        lIndex = lIndex + 1;
    else
        patternLabel{i} = ['S' num2str(sIndex)];
        sIndex = sIndex + 1;
    end
end

% Produce the picture
clf,
bar(1:nStableStates,energyRatio(sortedIndex))
xlabel('Pattern')
ylabel('Energy ratio')
title('Stable state energy ratio')
set(gca,'XTick',[1:nStableStates],'XTickLabel',patternLabel)
%% Additional plots, showing network energy and iteration statistics
networkEnergy = zeros(1,nPatterns);
for i = 1:nStableStates
    networkEnergy(i) = hopfield.GetEnergy(stablePatternMatrix(i,:));
end
subplot(1,2,1),
boxplot(networkEnergy,isSpuriousState,'labels',{'Memory','Spurious'})
title('Network energy')
ylabel('Network energy')

subplot(1,2,2),
boxplot(meanPatternIterations,isSpuriousState,'labels',{'Memory','Spurious'})
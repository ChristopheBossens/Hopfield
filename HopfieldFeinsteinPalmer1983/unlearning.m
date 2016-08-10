% Demonstrating the effects of unlearning on network recall capabilities
clear
networkSize = 32;
p = 0.5;
c = 0.3;

nExemplars = 5;round(c*networkSize);
hopfield = Hopfield(networkSize);
%% Create pattern matrix
patternMatrix = zeros(nExemplars, networkSize);


for i = 1:nExemplars
    pattern = hopfield.GeneratePattern(p);
    patternMatrix(i,:) = pattern;
end
%% Simulate statistics on spurious states
pcData = zeros(1,nExemplars);
itData = zeros(1,nExemplars);
spuriousData = zeros(3,nExemplars);
energyData = zeros(2,nExemplars);

nSpuriousSamples = 100;

hopfield.ResetWeights();
for i = 1:nExemplars
    hopfield.AddPattern(patternMatrix(i,:),1);
    
    % Get performance information
    [pc, it] = hopfield.TestPatterns(hopfield,patternMatrix(1:i,:));
    pcData(i) = pc;
    itData(i) = it;
    
    % Samples and summarize spurious states information
    [stableStates, stateHist] = hopfield.GetSpuriousStates(nSpuriousSamples);
    isOriginalPattern = zeros(1,size(stableStates,1));
    for j = 1:size(stableStates,1)
        storedPatternIndex = find(ismember(patternMatrix(1:i,:),stableStates(j,:),'rows')==1);
        if ~(isempty(storedPatternIndex))
            isOriginalPattern(j) = 1; 
        else
            storedPatternIndex = find(ismember(patternMatrix(1:i,:),stableStates(j,:),'rows')==1);
            if ~(isempty(storedPatternIndex))
                isOriginalPattern(j) = 1;
            end
        end
    end
    spuriousStates = find(isOriginalPattern == 0);
    spuriousData(1,i) = length(stateHist);
    spuriousData(2,i) = sum(stateHist(isOriginalPattern==1))/nSpuriousSamples;
    spuriousData(3,i) = sum(stateHist(isOriginalPattern==0))/nSpuriousSamples;
    
    % Sample and summarize state energy information
    originalEnergy = 0;
    for j = 1:i
        originalEnergy = originalEnergy + hopfield.GetEnergy(patternMatrix(j,:));
    end
    energyData(1,i) = originalEnergy/i;
    
    spuriousEnergy = 0;
    for j = 1:size(stableStates,1)
        if ~isOriginalPattern(j)
            spuriousEnergy = spuriousEnergy + hopfield.GetEnergy(stableStates(j,:));
        end
    end
    energyData(2,i) = spuriousEnergy/(nSpuriousSamples - sum(isOriginalPattern));
    
end

clf,
subplot(2,3,1)
plot(1:nExemplars,pcData,'LineWidth',2)
line([0 nExemplars],[0.5 0.5],'Color','k','LineWidth',2,'LineStyle','--')
set(gca,'YLim',[0 1])

subplot(2,3,2),hold on
plot(1:nExemplars,spuriousData(2,:),'g','LineWidth',2)
plot(1:nExemplars,spuriousData(3,:),'r','LineWidth',2)
subplot(2,3,5),hold on
plot(1:nExemplars,spuriousData(1,:),'k','LineWidth',2)

subplot(2,3,3), hold on
plot(1:nExemplars,energyData(1,:),'g','LineWidth',2)
plot(1:nExemplars,energyData(2,:),'r','LineWidth',2)
%% Train a number of patterns under the capacity limit and test the
% unlearningr rule
nSpuriousSamples = 100;
nUnlearningEpochs = 300;

pcData = zeros(1,nUnlearningEpochs);
itData = zeros(1,nUnlearningEpochs);
spuriousData = zeros(3,nUnlearningEpochs);
energyData = zeros(2,nUnlearningEpochs);

maxTrainingExemplars = 5;
hopfield.ResetWeights();

for i = 1:maxTrainingExemplars
    hopfield.AddPattern(patternMatrix(i,:),1);
end

for i = 1:nUnlearningEpochs
    hopfield.Unlearn(0.01);
    
    % Get performance information
    %[pc, it] = hopfield.TestPatterns(hopfield,patternMatrix(1:maxTrainingExemplars,:));
    %pcData(i) = pc;
    %itData(i) = it;
    
    % Samples and summarize spurious states information
    [stableStates, stateHist] = hopfield.GetSpuriousStates(nSpuriousSamples);
    isOriginalPattern = zeros(1,size(stableStates,1));
    for j = 1:size(stableStates,1)
        storedPatternIndex = find(ismember(patternMatrix(1:maxTrainingExemplars,:),stableStates(j,:),'rows')==1);
        if ~(isempty(storedPatternIndex))
            isOriginalPattern(j) = 1;           
        end
    end
    spuriousStates = find(isOriginalPattern == 0);
    spuriousData(1,i) = length(spuriousStates);
    spuriousData(2,i) = sum(stateHist(isOriginalPattern==1))/nSpuriousSamples;
    spuriousData(3,i) = sum(stateHist(isOriginalPattern==0))/nSpuriousSamples;
    
    % Sample and summarize state energy information
    originalEnergy = 0;
    for j = 1:maxTrainingExemplars
        originalEnergy = originalEnergy + hopfield.GetEnergy(patternMatrix(j,:));
    end
    energyData(1,i) = originalEnergy/maxTrainingExemplars;
    
    spuriousEnergy = 0;
    for j = 1:size(stableStates,1)
        if ~isOriginalPattern(j)
            spuriousEnergy = spuriousEnergy + hopfield.GetEnergy(stableStates(j,:));
        end
    end
    energyData(2,i) = spuriousEnergy/(nSpuriousSamples - sum(isOriginalPattern));
end

clf,
subplot(1,3,1)
plot(1:nUnlearningEpochs,pcData,'LineWidth',2)
line([0 nUnlearningEpochs],[0.5 0.5],'Color','k','LineWidth',2,'LineStyle','--')
set(gca,'YLim',[0 1])

subplot(1,3,2),hold on
plot(1:nUnlearningEpochs,spuriousData(2,:),'g','LineWidth',2)
plot(1:nUnlearningEpochs,spuriousData(3,:),'r','LineWidth',2)

subplot(1,3,3), hold on
plot(1:nUnlearningEpochs,energyData(1,:),'g','LineWidth',2)
plot(1:nUnlearningEpochs,energyData(2,:),'r','LineWidth',2)
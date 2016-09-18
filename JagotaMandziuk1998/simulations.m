clc;clear;
N = 100;
P = 30;

hebNet = Hopfield(N);
deltaNet = Hopfield(N);
patternMatrix = hebNet.GeneratePatternMatrix(P);
stabilityResults = zeros(2,P);
finalOverlap = zeros(2,P);

for patternIndex = 1:P
    clc;display(['Adding pattern ' num2str(patternIndex) '/' num2str(P)]);
   hebNet.AddPattern(patternMatrix(patternIndex,:),1);
   
   zeroError = 0;
   while zeroError == 0
       err = zeros(1,patternIndex);
       for trainingIndex = 1:patternIndex
           err(trainingIndex) = abs(mean(deltaNet.PerformDeltaUpdate(patternMatrix(trainingIndex,:),1)));
       end
       
       if sum(err) == 0
           zeroError = 1;
       end
   end
   
   for testIndex = 1:patternIndex      
       stabilityResults(1,patternIndex) = stabilityResults(1,patternIndex) + hebNet.IsStablePattern(patternMatrix(testIndex,:));
       stabilityResults(2,patternIndex) = stabilityResults(2,patternIndex) + deltaNet.IsStablePattern(patternMatrix(testIndex,:));
       
       output1 = hebNet.Converge(patternMatrix(testIndex,:));
       output2 = deltaNet.Converge(patternMatrix(testIndex,:));
       finalOverlap(1,patternIndex) = finalOverlap(1,patternIndex) + (output1*patternMatrix(testIndex,:)')/N;
       finalOverlap(2,patternIndex) = finalOverlap(2,patternIndex) + (output2*patternMatrix(testIndex,:)')/N;
   end
   stabilityResults(:,patternIndex) = stabilityResults(:,patternIndex)/patternIndex;
   finalOverlap(:,patternIndex) = finalOverlap(:,patternIndex)/patternIndex;
end
%%
clf,subplot(1,2,1)
hold on
plot(stabilityResults(1,:),'b','LineWidth',2)
plot(stabilityResults(2,:),'g','LineWidth',2)
set(gca,'YLim',[0 1])
title('Stability results')
xlabel('Patterns added')
ylabel('Proportion stable')

subplot(1,2,2)
hold on
plot(finalOverlap(1,:),'b','LineWidth',2)
plot(finalOverlap(2,:),'g','LineWidth',2)
title('Overlap results')

%% Error correction capabilities
% Add patterns so that the load for the Hebb net remains under the capacity
% limit.
P = 10;
patternMatrix = hebNet.GeneratePatternMatrix(P);

hebNet.ResetWeightMatrix();
deltaNet.ResetWeightMatrix();

hebNet.AddPatternMatrix(patternMatrix);

zeroError = 0;
while zeroError == 0
   err = zeros(1,patternIndex);
   for trainingIndex = 1:P
       err(trainingIndex) = abs(mean(deltaNet.PerformDeltaUpdate(patternMatrix(trainingIndex,:),1/N)));
   end

   if sum(err) == 0
       zeroError = 1;
   end
end

bitsFlipped = [1:30];
nBitsFlipped = length(bitsFlipped);
errorCorrection = zeros(2,nBitsFlipped);

for i = 1:nBitsFlipped
    for j = 1:P
        inputPattern = patternMatrix(j,:);
        distortedPattern = hebNet.DistortPattern(inputPattern,bitsFlipped(i)/N);

        output1 = hebNet.Converge(distortedPattern);
        output2 = deltaNet.Converge(distortedPattern);

        errorCorrection(1,i) = errorCorrection(1,i) + (output1*inputPattern')/N;
        errorCorrection(2,i) = errorCorrection(2,i) + (output2*inputPattern')/N;
    end
end
errorCorrection = errorCorrection./P;
%%
clf,
hold on
plot(errorCorrection(1,:),'b','LineWidth',2);
plot(errorCorrection(2,:),'g','LineWidth',2);
set(gca,'YLim',[0 1.1]),xlabel('Bits flipped'),ylabel('Average final overlap')
title('Error correction results')
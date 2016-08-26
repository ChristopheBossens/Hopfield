%% Code for reproducing figure 1a
clc;clear
N = 1000;
h = Hopfield(N);
nSteps = 10000;

tau_1 = 1.5*N;
tau_2 = 0.2*N;
networkLoad = 0.02:0.005:0.20;
adaptationValues = [0.0:0.01:0.20];
nLoads = length(networkLoad);
nAdaptations = length(adaptationValues);
results = zeros(nAdaptations,2,nLoads);
h.UseStochasticDynamics(1/0.01);

for loadIndex = 1:nLoads
    % Generate patterns with different weight values
    P = round(networkLoad(loadIndex)*N);
    p = h.GeneratePatternMatrix(P,0.5);
    w = 0.5.*ones(1,P);
    w(1)= 1;
    h.ResetWeightMatrix();
    for i = 1:P
        h.AddPattern(p(i,:),w(i)/(N*sum(w)));
    end
    
    for adaptationIndex = 1:nAdaptations
        clc;display(['Testing load: ' num2str(loadIndex) '/' num2str(nLoads) ', adaptation: ' num2str(adaptationIndex) '/' num2str(nAdaptations)])
        A = adaptationValues(adaptationIndex);
        lastStateChange = zeros(1,N);
        
        currentState = p(2,:);
        h.ClampState(currentState);
        % Run the simulation for different timestaps
        for i = 1:nSteps
            nextUnit = randi(N,1);
            threshold = A./(1 + exp(-currentState.*(lastStateChange-tau_1)./tau_2));
            h.SetThreshold(threshold);
            finalState = h.UpdateUnit(nextUnit);
            
            lastStateChange = lastStateChange + 1;
            if (finalState(nextUnit) ~= currentState(nextUnit))
                lastStateChange(nextUnit) = 0;
                currentState(nextUnit) = finalState(nextUnit);
            end
        end
        strongOverlap = (finalState*p(1,:)')/N;
        weakOverlap   = (finalState*p(2,:)')/N;
        results(adaptationIndex,1,loadIndex) = strongOverlap;
        results(adaptationIndex,2,loadIndex) = weakOverlap;
    end
end


%% Create figure 1a
strongResult = squeeze(abs(results(:,1,:)));
weakResult = squeeze(abs(results(:,2,:)));

figure,
subplot(1,3,1),imshow(strongResult,[0 1],'InitialMagnification','fit')
set(gca,'XTick',[1:7:nLoads],'XTickLabel',networkLoad(1:7:nLoads))
set(gca,'YTick',[1:10:nAdaptations],'YTickLabel',adaptationValues(1:10:nAdaptations),'YDir','normal')
xlabel('Saturation'),ylabel('Adaptation'),
subplot(1,3,2),imshow(weakResult,[0 1],'InitialMagnification','fit')
set(gca,'XTick',[1:7:nLoads],'XTickLabel',networkLoad(1:7:nLoads))
set(gca,'YTick',[1:10:nAdaptations],'YTickLabel',adaptationValues(1:10:nAdaptations),'YDir','normal')
xlabel('Saturation'),ylabel('Adaptation'),
subplot(1,3,3)
imshow(strongResult-weakResult,[-1 1],'InitialMagnification','fit'),
colormap hot
colorbar
axis on, axis square
set(gca,'XTick',[1:7:nLoads],'XTickLabel',networkLoad(1:7:nLoads))
set(gca,'YTick',[1:10:nAdaptations],'YTickLabel',adaptationValues(1:10:nAdaptations),'YDir','normal')
xlabel('Saturation'),ylabel('Adaptation'),
%% Simulations for reproducing figure 1c,d,e
clc;clear
N = 1000;
nSteps = 30000;
h = Hopfield(N);

p = h.GeneratePatternMatrix(10,0.5);
w = 0.5.*ones(1,10);
w(1) = 1.0;
h.ResetWeightMatrix();
h.UseStochasticDynamics(1/0.01);
for i = 1:10
    h.AddPattern(p(i,:),w(i)/(sum(w)*N));
end

overlapData = zeros(3,2,nSteps);
tau_1 = 1.5*N;
tau_2 = 0.2*N;
A = [0.05 0.1 0.3];
for adaptationIndex = 1:length(A)
    initialState = p(2,:);
    h.ClampState(initialState);
    
    lastStateChange = N.*ones(1,N);
    for step = 1:nSteps
        nextUnit = randi(N,1);
        threshold = A(adaptationIndex)./(1 + exp(-initialState.*(lastStateChange-tau_1)./tau_2));
        h.SetThreshold(threshold);
        finalState = h.UpdateUnit(nextUnit);

        lastStateChange = lastStateChange + 1;
        if (initialState(nextUnit) ~= finalState(nextUnit))
            lastStateChange(nextUnit) = 0;
            initialState(nextUnit) = finalState(nextUnit);
        end
        
        overlapData(adaptationIndex,1,step) = abs(finalState*(p(1,:)')/N);
        overlapData(adaptationIndex,2,step) = abs(finalState*(p(2,:)')/N);
        
        if (mod(step,100)==0)
            clc;display([num2str(step) '/' num2str(nSteps)])
        end
    end
end

%% Generate figure 1cde
clf,
for i = 1:3
    subplot(3,1,i),hold on
    title(['A = ' num2str(A(i))])
    xlabel('Update step')
    ylabel('|m_u|')
    plot(squeeze(overlapData(i,1,:)))
    plot(squeeze(overlapData(i,2,:)))
    legend('Strong','weak')
    set(gca,'YLim',[-0.2 1])
end
%% Code for reproducing figure 1b
clc;clear
N = 1000;
h = Hopfield(N);
nSteps = 10000;

networkLoad = 0.02:0.005:0.20;
tValues = [0.001:0.002:0.17];
nLoads = length(networkLoad);
nT = length(tValues);

results = zeros(nT,nLoads);
for loadIndex = 1:nLoads
    alpha = networkLoad(loadIndex);
    P = round(alpha*N);
    
    % Generate pattern matrix with strong and weak patterns
    p = h.GeneratePatternMatrix(P,0.5);
    w = ones(1,P);
    w(1) = 3;
    h.ResetWeightMatrix()
    for i = 1:P
        h.AddPattern(p(i,:),w(i)/(N*sum(w)));
    end
    
    for tIndex = 1:nT
        % Test different patterns for a fixed amount of steps
        h.UseStochasticDynamics(1/tValues(tIndex));
        patternId = 2;
        initialState = p(patternId,:);
        for i = 1:nSteps
            nextUnit = randi(N,1);
            finalState = h.UpdateUnit(nextUnit);
            
            if (finalState(nextUnit) ~= initialState(nextUnit))
                initialState(nextUnit) = finalState(nextUnit);
            end
        end
        strongOverlap = (finalState*p(1,:)')/N;
        weakOverlap   = (finalState*p(patternId,:)')/N;
        results(tIndex,loadIndex) = abs(strongOverlap)-abs(weakOverlap);
    end  
end
%% Create the figure for experiment 1b
clf,
imshow(results,[-1 1],'InitialMagnification','fit'),colormap hot, colorbar
axis on, axis square
set(gca,'XTick',[1:7:nLoads],'XTickLabel',networkLoad(1:7:nLoads))
set(gca,'YTick',[1:10:nT],'YTickLabel',tValues(1:10:nT),'YDir','normal')
xlabel('Saturation'),ylabel('Temperature'),
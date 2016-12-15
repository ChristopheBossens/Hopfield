# Hopfield

The Hopfield network is one of the classical examples of a recurrent neural network. An important property of this network is that each unit is connected to every other unit in the network. This turns the network into a dynamical system in which the network will settle into attractor states that (hopefully) correspond to stored patterns in the network. Because of its similarities with certain magnetic systems, the model has receieved a considerable amount of attention from physicists. Their main approach is the use of techniques from statistical mechanics to formally analyse properties of the model such as the storage capacity. The model also received attention from psychologists and neuroscientists because it has characteristics that resemble aspects of human memory. A common connection that is made is the link between sustained neuronal activity in prefrontal cortex during a delayed match-to-sample task and the ability of the model to converge and remain in a state that corresponds to a stored exemplar. The model has aslo been adjusted so that it mimicks the recall performances that are obtained in list-learning experiments. Although the model does lack a lot of biological plausibility, recent research is beginning to incorporate more biological properties into these model. These included adding short-term depresssion and facilitation mechanisms, as well as replacing the original [-1, 1] units with units that are described by differential equations which mimick voltage fluctuations of biological neurons.

To better understand the Hopfield model I have read several papers that investigated this model, and to better understand these papers I have tried to replicate the simulations that were performed. To make my life a little bit easier I am developing corresponding Matlab functions that help with these simulations. In the long run, my goal is to incorporate findings from different studies into a single Hopfield model that can easily be used to perform new simulations. 

# How to use the model
```matlab
% Create a new Hopfield network with 50 units
myHopfield = Hopfield(50);

% The following configuration options are available
myHopfield.SetThreshold(0.2); % Sets the unit threshold (default = 0)
myHopfield.SetUnitModel('S'); % 'S' will use units with -1,1 activity values, 'V' will use units with 0,1
myHopfield.SetUpdateMode(updateMode); % Update mode can be 'sync' or 'async'


% Create a pattern in which 50% of the units are active
pattern = myHopfield.GeneratePattern(0.5)

% Add a pattern to the model
myHopfield.StorePattern(pattern);

% Create a distorted version of a pattern
distortedPattern = myHopfield.DistortPattern(pattern);

% Perform a single state update
networkState = myHopfield.UpdateState(distortedPattern)

% Network weights can be obtained with
weights = hopfield.GetWeightMatrix()

% Network can be reset using
myHopfield.ResetWeightMatrix()
```

# Example result
The following result was obtained by running the Hopfield_demo.m script. Patterns are loaded into the network sequentially and after each pattern is loaded we tested if the previous patterns if all loaded patterns were stable states. This was done by setting the network state to each pattern, iterating untill the network is stable and calculating the degree of overlap with the input pattern.

![Hopfield demo](/hopfield_demo.jpg?raw=true)


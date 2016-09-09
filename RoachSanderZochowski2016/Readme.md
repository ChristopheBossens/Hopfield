# Memory recall and spike-frequency adaptation

## Paper summary
A classical result in Hopfield neural networks is that starting from a noisy version of a stored pattern, the network will be able to recover the originally stored pattern. In that sense, it is said that the network remembers the original pattern. However, in this framework remembering a stored pattern occurs only when there is an external input which closely resembles the originally stored pattern. This is a limitation of the model because there are occassions where we suddenly remember something without having a clear external cue. To simulate this phenomenon with the Hopfield network, we must change how the dynamics behave. The present paper demonstrates that either increasing the global temperature of the network or incorporating a mechanism similar to spike frequency adaptation can provoke a switch in the stable memory state.

## Simulation results
I replicated the simulatios corresponding to figure 1. The first figure shows how changing an adaptation parameter affects the dynamics of the model. For low adaptation values. An original stable state will remain stable (black region), for higher adaptation values the stable state switches to another, more strong attractor. Finally for even higher adaptation values there is nog clear overlap with any of the original states.

![Spike frequency adaptation](/RoachSanderZochowski2016/figure1a.jpg?raw=true)

The same effects can be achieved by changing the global temperature of the network. It is however less clear how a global temperature is implemented in biological networks.

![Spike frequency adaptation](/RoachSanderZochowski2016/figure1b.jpg?raw=true)

The final plot shows individual simulation runs. In each case we start the network in a state corresponding to one of the stored patterns. Depending on the adaptation value, the network will remain in that state (no change in overlap with the strong attractor (blue) and original weak attractor (red)), it will switch to the strong attractor, or it will not show a preference for either attractor.

![Spike frequency adaptation](/RoachSanderZochowski2016/figure1cde.jpg?raw=true)
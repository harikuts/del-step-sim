# DEL-Step: An Interactive Decentralized Learning Simulator

A simple, easy-to-deploy, interactive, step-by-step simulator for decentralized learning, using Tensorflow. PyTorch integration planned for the future.

*Currently a work in progress.*

## Overview

DEL-Step is a Python-based step-by-step simulator meant to be used as a playground for decentralized learning with Tensorflow, the most widely used machine learning platform. Users can develop their own models and configure their own datasets in *learning_module.py* and then deploy them in a decentralized setting. Notable features of this simulator include:
* Configuring a customizable network with any number of nodes.
* Setting some network features, including network loss and latency.
* Processing step-by-step commands from the user to allow for user interaction at any step during runtime.
* Automating behavior through user-written scripts.
* Controlling different behaviors such as network activity, training, model sharing, and model aggregation.
* Selectively running the above behaviors independently or together, on selected nodes or all nodes.
* Analyzing, reporting, and logging available statistics at any step during runtime.
* Setting group-level controls and behaviors.
* Connecting to a network simulation backend, such as NS-3 or Mininet. *(Coming soon)*
* Creating and removing nodes during runtime. *(Coming soon)*
* Setting mobility patterns. *(Coming soon)*
* Pre-configured network scenarios. *(Coming soon)*

## Decentralized Learning

This simulator is currently based on decentralized *federated* learning.

*More information coming soon.*

### Model Aggregation

This simulator uses a modified model aggregation algorithm, in order to account for recent interactions between nodes more common in real-world deployments. This algorithm relies on an encounter tracking table. This mechanism is explained below, along with paramterization that allows for a gradient of behaviors from standard aggregation to this modified aggregation.

*More information coming soon.*

## Runtime Commands

The following includes all commands currently in the simulator. To best understand the use of these commands, the user must understand the proper scope of each command.

### Scopes

Most commands in the simulator operate in one of three scopes:
* **System-level operations** involve the entire system -- the network graph itself and all nodes and connections within the graph. These commands usually involve commands with the simulator interface, settting simulation parameters, and multi-level operations (like steps).
* **Internode-level operations** involve operations that occur between nodes -- model exchanges and other communications. These commands usually have to do with transmission of model information.
* **Node-level operations** involve operations that occur within the node. These operations are usually individual counterparts to other operations, such as individual steps, individual information, individual training and testing, etc.

Along with those scopes, this simulator also features an additional setting, if desired:
* _Group-level operations_ involve user-defined groups of nodes within the network graph. These operations include many of the same ones in the other scopes but occur within a specified group.

### Interface
The following commands are used to view information about the simulation itself.

#### *neighborhood*
**Usage:** `neighborhood`

This command shows you all nodes currently in the system.

#### *guestbook*
**Usage:** `guestbook [node]     `

This command shows you all network-connected neighbors of the specified node `[node]*`.

### Learning
The following commands are used to train, aggregate, and evaluate the models within each node.

#### *train*
**Usage:** `train [node]`

This command deploys the training process within the specified node `[node]`. The training process involves running one iteration of the learning model within the node on local data.

**Usage:** `train`

This command by itself runs the training process in all nodes within the system.

#### *test*
**Usage:** `test [node]`

This command deploys the model evaluation process within the specified node `[node]`. The process calls `evaluate` method of the Tensorflow model within the node.

**Usage:** `test`

This command by itself runs tests on all nodes within the system.

### Steps

Steps are the main mechanisms employed to advance the simulation. There are 3 types of step behaviors -- training steps, aggregation steps, and network steps. However, the actual implementation of these behaviors is currently through `step`, a combination training and aggregation step, and `nstep`, the network step.

#### *step*

This step serves as a general processing step. If there is new model information that has just been received, then this step runs model aggregation. If not, then the model is trained, as done with `train`.

**Usage:** `step [node]`

Runs the general processing step on the specified node `[node]`.

**Usage:** `step`

Runs the general processing step on all nodes in the system.

#### *nstep*

This step is the network processing step. Until this step is executed, information that is to be sent waits at output buffers of each node. When this step is executed, all network events that have been queued up are enacted in the order they were queued. This is the main step metric for simulation advancement.

**Usage:** `nstep [node]`

Runs the network processing step on the specified node `[node]`.

**Usage:** `nstep`

Runs the network processing step on all nodes in the system.

#### *autonet*

**Usage:** `autonet`

Toggles automatic network processing after each commands and will return the current status of this toggle once command is used. Enabling autonet is useful if you're not worried specifically about controlling network elements and are focused solely on model performance.

### Communicating Models

The following commands involve transmission of model information between nodes.

#### *exchange*

**Usage:** `exchange [node1] [node2]`

This command enacts an exchange of model information between `[node1]` amd `[node2]`.

#### *flood*

**Usage:** `flood [node]`

This command has the specified node `[node]` transmit models to all its graph-connected neighbors.

**Usage:** `flood`

All nodes within the simulation flood all their nieghbors. __This command is not recommended__  as there are no communication redundancy checks in place.

### Group-Level Commands

Group-level operatability was introduced to perform experiments within groups within the larger system. Eventually, all major commands will have group-level counter parts.

#### *list*

**Usage:** `group list`

This command is used to view all groups currently in the system.

#### *roster*

**Usage:** `group roster [group]`

This command is used to display all members of the specified group `[group]`.

#### *membership*

**Usage:** `group membership [node]`

This command is used to view what groups the node `[node]` is a part of. A single node can have membership in multiple groups.

#### *create*

**Usage:** `group create [groupname]`

This command creates a new group with the name `[groupname]`. If a group already exists with this name, it will be overwritten.

#### *add*

**Usage:** `group [group] add [list of nodes]`

This commands adds all nodes listed in `[list of nodes]` to the group `[group]`.

#### *remove*

**Usage:** `group [group] remove [list of nodes]`

This command removes all nodes listend in `[list of nodes]` from the group `[group]`.

#### *share*

**Usage:** `group [group] share`

This command facilitates every possible model exchange within the group `[group]`, such that each node has exchanged once with all other nodes. Redundancy is accounted for.

#### *train*

**Usage:** `group [group] train`

This command performs a training step on each node within the group `[group]`.

### Logging

These commands involve logging and saving information and results from the simulator.

#### *log*

**Usage:** `log`

This command returns a full log of all commands and results that have been executed in the simulation thus far.

**Usage:** `log results`

This command returns a log of only results that have been obtained from the simulation thus far.

### Scripting

This simulator also allows the user to create scripts as a list of commands to execute. Commands are written with each executable command and arguments written per line. Commands in a script are queued for execution in the simulator.

***Example:***
```
train 10.0.0.1
train 10.0.0.2
exchange 10.0.0.1 10.0.0.2
step
test 10.0.0.1
test 10.0.0.2
```

The above example individually trains two nodes, *10.0.0.1* and *10.0.0.2*, exchanges model information between them, uses a general processing step to aggregate the newly found model information within each node, then evaluates each of the nodes.

#### *load*

**Usage:** `load [script]`

Runs a script with filename `[script]` and queues commands from the script for the simulator to execute.

## Files

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

### Model Aggregation

This simulator uses a modified model aggregation algorithm, in order to account for recent interactions between nodes more common in real-world deployments. This algorithm relies on an encounter tracking table. This mechanism is explained below, along with paramterization that allows for a gradient of behaviors from standard aggregation to this modified aggregation.

*More information coming soon*

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
**Usage:** `guestbook [node]`

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

### Steps

#### *step*

#### *nstep*

#### *autonet*

### Communicating Models

#### *exchange*

#### *flood*

### Group-Level Commands

#### *list*

#### *roster*

#### *create*

#### *add*

#### *remove*

#### *membership*

#### *share*

#### *train*

### Scripting

#### *load*

## Files

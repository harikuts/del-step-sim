#!/usr/bin/python

from dummy_net import DummyNet
from dummy_net import Packet

from learning_module import ModelIncubator
from learning_module import Model

from client import Client

import logger

import threading

# Command Error (to be used within console)
class CommandError(BaseException):
    def __init__(self, error_message=None):
        self.error_message = error_message
    def __str__(self):
        return("CommandError: " + self.error_message)

# The main console
class Console:
    def __init__(self, clientDict):
        self.clients = clientDict
        # Init command processing
        self.cmd_queue = []
        # Init groups and command list
        self.groups = {}
        self.group_commands = {'list', 'roster', 'create', 'membership'}
        self.automatic_net_flag = False
        # Init log
        self.log = logger.Log()
    def run(self):
        while True:
            # Process command
            # Begin tracking for logging
            self.log.beginEntry(logger.CommandEntry)
            try:
                # Any inits go here
                switch_autonet = None
                # If cmd_queue is not empty, pop a command
                if len(self.cmd_queue):
                    cmd = self.cmd_queue.pop(0)
                    print("\nQ>" + cmd)
                else:
                    cmd = input("\n>>")
                # Record command
                self.log.NewEntry.set_cmd(cmd.strip())
                # Process command
                cmd = cmd.strip().split(" ")
                cmd = [c.strip() for c in cmd]
                if cmd[0] == "exit":
                    break
                elif cmd[0] == "load":
                    self.load_script(cmd[1])
                elif cmd[0] == "step":
                    if len(cmd) == 1:
                        self.step()
                    else:
                        self.istep(cmd[1])
                # Network processes step
                elif cmd[0] == "nstep":
                    if len(cmd) == 1:
                        self.nstep()
                    else:
                        self.instep(cmd[1])
                # Switch for auto net processes after each command
                elif cmd[0] == "autonet":
                    switch_autonet = self.toggle_autonet()
                elif cmd[0] == "train":
                    if len(cmd) == 1:
                        self.tstep()
                    else:
                        self.itstep(cmd[1])
                elif cmd[0] == "neighborhood":
                    print (self.get_all_addrs())
                elif cmd[0] == "exchange":
                    self.exchange(cmd[1], cmd[2])
                elif cmd[0] == "flood":
                    if len(cmd) == 1:
                        self.floodall()  
                    else:
                        self.flood(cmd[1])
                elif cmd[0] == "guestbook":
                    self.iguest(cmd[1])
                elif cmd[0] == "test":
                    self.test(cmd[1])
                # Group commands
                elif cmd[0] == "group":
                    # List all groups
                    if cmd[1] == "list":
                        print("Active groups:", self.groups.keys())
                    # List members of a group
                    elif cmd[1] == "roster":
                        print(self.get_group_roster(cmd[2]))
                    # See IP's membership
                    elif cmd[1] == "membership":
                        self.group_membership(cmd[2])
                    # Create new group
                    elif cmd[1] == "create":
                        self.groups[cmd[2]] = []
                    # Update network graph connections
                    elif cmd[1] == "connect":
                        self.update_group_connections()
                    # GROUP LEVEL COMMANDS
                    # Add member to a group
                    elif cmd[1] in self.groups:
                        groupname = cmd[1]
                        if cmd[2] == "add":
                            self.group_add(groupname, cmd[3:])
                        # Remove member from a group
                        elif cmd[2] == "remove":
                            self.group_remove(groupname, cmd[3:])
                        # Share within a group
                        elif cmd[2] == "share":
                            self.group_share(groupname)
                        # Train within a group
                        elif cmd[2] == "train":
                            self.group_train(groupname)
                        # Aggregate within a group
                        # Report error
                        else:
                            raise CommandError("Not a valid group-specific command.")
                    else:
                        raise CommandError("Group command invalid.")
                # Log commands
                elif cmd[0] == "cycle":
                    self.next_cycle()
                elif cmd[0] == "log":
                    self.log_all()
                else:
                    raise CommandError("Command does not exist.")
                # Process network step if auto net
                if self.automatic_net_flag:
                    self.nstep()
                    print("Network processes executed.")
                if switch_autonet is not None:
                    self.automatic_net_flag = switch_autonet
                # If we made it to the end of the command loop, we're successful
                self.log.NewEntry.set_success(True)
            except CommandError as e:
                print("Command did not work:")
                print(str(e))
                self.log.NewEntry.set_success(False)
            except Exception as e:
                print("Command did not work. Please check arguments.")
                self.log.NewEntry.set_success(False)
            # Commit to log
            self.log.commitEntry()

    # COMMAND OPERATIONS
    # SCRIPTING
    def load_script(self, filename):
        try:
            with open(filename, "r") as f:
                self.cmd_queue = self.cmd_queue + f.readlines()
        except:
            raise CommandError("Failed to load file: " + filename)
    # SYSTEM LEVEL COMMANDS
    def get_all_addrs(self):
        return str(list(self.clients.keys()))
    def step(self):
        for client in self.clients.values():
            client.process()
    def nstep(self):
        # do twice to cover all interactions
        for client in self.clients.values():
            client.net.step()
        for client in self.clients.values():
            client.net.step()
    def toggle_autonet(self):
        will_toggle = not self.automatic_net_flag
        print ("Automatic network processes", ("enabled" if will_toggle else "disabled"))
        return will_toggle
    def tstep(self, subset=None):
        nodeset = self.clients.keys() if subset is None else subset
        selected_clients = [self.clients[addr] for addr in nodeset]
        for client in selected_clients:
            client.train_model()
    def floodall(self):
        for client in self.clients.values():
            for neighbor in client.net.neighbors.keys():
                client.transmit_model(neighbor)
    # INTERNODE LEVEL COMMANDS
    def exchange(self, ip1, ip2):
        self.clients[ip1].transmit_model(ip2)
        self.clients[ip2].transmit_model(ip1)
    def flood(self, ip):
        for neighbor in self.clients[ip].net.neighbors.keys():
            self.clients[ip].transmit_model(neighbor)
    # NODE LEVEL COMMANDS
    def istep(self, ip):
        self.clients[ip].process()
    def instep(self, ip):
        self.clients[ip].net.step()
    def itstep(self, ip):
        self.clients[ip].train_model()
    def iguest(self, ip):
        print(self.clients[ip].guest_book)
    def test(self, ip):
        results = self.clients[ip].model.test()
        print("LOSS:", results[0], "ACCURACY:", results[1])
    # GROUP COMMANDS
    def get_group_roster(self, groupname):
        if groupname in self.groups.keys():
            return self.groups[groupname]
        else:
            raise CommandError("Group name not valid. Current groups: " + str(self.groups.keys()))
    def create_group(self, name):
        if name not in self.group_commands:
            self.groups[name] = []
        else:
            raise CommandError("Name not allowed as it is reserved for commands.")
        # print("Current groups:", self.groups.keys())
    def group_add(self, groupname, iplist):
        if groupname in self.groups.keys():
            for ip in iplist:
                if ip in self.clients.keys():
                    self.groups[groupname].append(ip)
                    print("Added", ip)
                else:
                    raise CommandError("IP address " + ip + " not valid.")
                print("Updated roster:", self.groups[groupname])
        else:
            raise CommandError("Group name not valid. Current groups: " + str(self.groups.keys()))
    def group_remove(self, groupname, iplist):
        if groupname in self.groups.keys():
            for ip in iplist:
                if ip in self.groups[groupname]:
                    self.groups[groupname].remove(ip)
                    print("Removed", ip)
                else:
                    print("IP address ", ip, "not found in group.")
                raise CommandError("Updated roster:" + str(self.groups[groupname]))
        else:
            raise CommandError("Group name not valid. Current groups: " + str(self.groups.keys()))
    def group_membership(self, ip):
        if ip in self.clients.keys():
            memberships = []
            for group in self.groups.keys():
                if ip in self.groups[group]:
                    memberships.append(group)
            print(ip, "is a member of:", memberships)
        else:
            raise CommandError("IP address not valid.")
    def group_share(self, groupname):
        if groupname in self.groups.keys():
            members = self.groups[groupname][:]
            # Hit every combination of members
            while len(members) >= 2:
                node1 = members.pop(0)
                for node2 in members:
                    # self.exchange(node1, node2)
                    print("Exchanging between", node1, "and", node2)
                    self.exchange(node1, node2)
        else:
            raise CommandError("Group name not valid. Current groups: " + str(self.groups.keys()))
    def group_train(self, groupname):
        if groupname in self.groups.keys():
            self.tstep(subset=self.groups[groupname])
        else:
            raise CommandError("Group name not valid. Current groups: " + str(self.groups.keys()))
    def update_group_connections(self):
        pass
    # LOG COMMANDS
    def next_cycle(self):
        self.log.new_step()
    def log_all(self):
        self.log.printLog()

import time
from datetime import datetime

# Create a graph
graph = {}
# graph["10.0.0.1"] = ["10.0.0.2", "10.0.0.3"]
# graph["10.0.0.2"] = ["10.0.0.1", "10.0.0.3"]
# graph["10.0.0.3"] = ["10.0.0.1", "10.0.0.2"]
# graph["10.0.0.4"] = ["10.0.0.1", "10.0.0.3"]

from dummy_net import build_fully_connected_graph

nodes = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4", "10.0.0.5"]
for node in nodes:
    graph[node] = []
graph = build_fully_connected_graph(graph)

print("Created network graph.")

# Create nodes for the virtual network
ipRegistry = {}
for addr in graph.keys():
    newNode = DummyNet(addr, graph[addr])
    ipRegistry[addr] = newNode
# Build network (decentralized)
for addr in graph.keys():
    ipRegistry[addr].init_network(ipRegistry)
print("Registered nodes in network graph.")

# Create Incubator with data ratios
MI = ModelIncubator([0.2, 0.1, 0.2, 0.3, 0.2])

# Create clients
clientDict = {}
ind = 0
for ip in ipRegistry.keys():
    print("Creating client ", ind, " with IP ", ip, ".")
    clientDict[ip] = Client(netNode=ipRegistry[ip], model=Model(data=MI.data_shares[ind], test_data=MI.test_data))
    ind += 1
print("Clients created and linked to nodes.")

# clientDict["10.0.0.1"].transmit(str(time.time()), "10.0.0.2")
# time.sleep(2.5)
# clientDict["10.0.0.2"].transmit(str(time.time()), "10.0.0.3")
# time.sleep(2.5)

# print("Begin experiment.")
# for i in range(10):
#     secrets.choice(list(clientDict.values())).transmit_model()
#     time.sleep(1)

# Start execution
console = Console(clientDict)
console.run()

print("Ending experiment.")
# Kill all nodes
for addr in graph.keys():
    ipRegistry[addr].kill()
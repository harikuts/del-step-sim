#!/usr/bin/python

from dummy_net import DummyNet
from dummy_net import Packet

from learning_module import ModelIncubator
from learning_module import Model

from client import Client

import threading

class Console:
    def __init__(self, clientDict):
        self.clients = clientDict
        self.cmd_queue = []
        self.groups = {}
        pass
    def run(self):
        while True:
            # Process command
            try:
                # If cmd_queue is not empty, pop a command
                if len(self.cmd_queue):
                    cmd = self.cmd_queue.pop(0)
                    print("\nQ>" + cmd)
                else:
                    cmd = input("\n>>")
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
                elif cmd[0] == "nstep":
                    if len(cmd) == 1:
                        self.nstep()
                    else:
                        self.instep(cmd[1])
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
                    if cmd[1] == "roster":
                        print(self.get_group_roster(cmd[1]))
                    # Create new group
                    if cmd[1] == "new":
                        self.groups[cmd[2]] = []
                    # Add member to a group
                    if cmd[1] == "add":
                        self.group_add(cmd[2], cmd[3:])
                    # Remove member from a group
                    if cmd[1] == "remove":
                        self.group_remove(cmd[2], cmd[3:])
                else:
                    print("Command does not exist.")
                # Process network step
                self.nstep()
                print("Network processes executed.")
            except Exception as e:
                print("Command did not work. Check arguments.")
                print(e)
    # SCRIPTING
    def load_script(self, filename):
        try:
            with open(filename, "r") as f:
                self.cmd_queue = self.cmd_queue + f.readlines()
        except:
            print("Failed to load file: " + filename)
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
    def tstep(self):
        for client in self.clients.values():
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
            print("Group name not valid. Current groups:", self.groups.keys())
    def create_group(self, name):
        self.groups[name] = []
        print("Current groups:", self.groups.keys())
    def group_add(self, groupname, iplist):
        if groupname in self.groups.keys():
            for ip in iplist:
                if ip in self.clients.keys():
                    self.groups[groupname].append(ip)
                    print("Added", ip)
                else:
                    print("IP address ", ip, "not valid.")
                print("Updated roster:", self.groups[groupname])
        else:
            print("Group name not valid. Current groups:", self.groups.keys())
    def group_remove(self, groupname, iplist):
        if groupname in self.groups.keys():
            for ip in iplist:
                if ip in self.groups[groupname]:
                    self.groups[groupname].remove(ip)
                    print("Removed", ip)
                else:
                    print("IP address ", ip, "not found in group.")
                print("Updated roster:", self.groups[groupname])
        else:
            print("Group name not valid. Current groups:", self.groups.keys())
    def group_membership(self, ip):
        if ip in self.clients.keys():
            memberships = []
            for group in self.groups.keys():
                if ip in self.groups[group]:
                    memberships.append(group)
            print(ip, "is a member of:", memberships)
        else:
            print("IP address not valid.")

import time
from datetime import datetime

# Create a graph
graph = {}
graph["10.0.0.1"] = ["10.0.0.2", "10.0.0.3"]
graph["10.0.0.2"] = ["10.0.0.1", "10.0.0.3"]
graph["10.0.0.3"] = ["10.0.0.1", "10.0.0.2"]
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
MI = ModelIncubator([0.5, 0.3, 0.2])

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
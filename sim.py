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
        pass
    def run(self):
        while True:
            # Process command
            try:
                cmd = input("\n>>").strip().split(" ")
                cmd = [c.strip() for c in cmd]
                if cmd[0] == "exit":
                    break
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
                elif cmd[0] == "tstep":
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
                else:
                    print("Command does not exist.")
                # Process network step
                self.nstep()
                print("Network processes executed.")
            except Exception as e:
                print("Command did not work. Check arguments.")
                print(e)
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
    clientDict[ip] = Client(netNode=ipRegistry[ip], model=Model(data=MI.data_shares[ind]))
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
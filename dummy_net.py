#!/usr/bin/python

import time
import threading
import traceback

# Packet class
class Packet:
    def __init__(self, src, dest, data):
        self.src = src
        self.dest = dest
        self.data = data
    def __str__(self):
        return "src=" + self.src + ",dest=" + self.dest + ",data=" + str(self.data)

class DummyNet:
    def __init__(self, address, neighbor_addrs = []):
        self.ip = address
        self.neighbor_addrs = neighbor_addrs
        self.outbox = []
        self.receiver = []
        self.inbox = []
        self.active = True
        
    # Second stage initialization to build the 'network connections'.
    def init_network(self, registry):
        # Build neighbors
        self.build_neighbors(registry)
        # Init sender and receiver processes (removed because threading is causing latency issues)
        # self.init_sender()
        # self.init_receiver()

    def step(self):
        self.__send()
        self.__receive()
        
    # Used to set active flag such that send/receive processes terminate.
    def kill(self):
        self.active = False
        
    # Networ reporting function.
    def print_(self, message):
        # Network level print messaging
        output = "NET::[" + str(self.ip) + "]::"
        try:
            output = output + str(message)
        except:
            output = output + "Message not printable."
        print(output)
        
    # Given registry, builds neighbor dictionary.
    def build_neighbors(self, registry):
        # The registry is built as a dictionary with key IP address an entry DummyNet object
        self.neighbors = {}
        for addr in self.neighbor_addrs:
            self.neighbors[addr] = registry[addr]
    
    # Starts sender service.
    def init_sender(self):
        threading.Thread(target=self.__send, args=()).start()
        pass
    
    # Starts receiver service.
    def init_receiver(self):
        threading.Thread(target=self.__receive, args=()).start()
        pass
    
    # Network layer send function.
    def __send(self):
        if self.active:
            # Send packet, if failed, print exception.
            try:
                while len(self.outbox):
                    packet = self.outbox.pop(0)
                    self.neighbors[packet.dest].receiver.append(packet)
                    self.print_("Sent: " + str(packet))
            except Exception as e:
                self.print_("Sending error has occurred.")
                traceback.print_exc()
    
    # Receiving/processing function.
    def __receive(self):
        if self.active:
            try:
                while len(self.receiver):
                    packet = self.receiver.pop(0)
                    self.inbox.append(packet)
                    self.print_("Received: " + str(packet))
            except Exception as e:
                self.print_("Receiving error has occurred.")
                traceback.print_exc()
        pass
    
    # Application layer send function.
    def send(self, payload, address):
        # Create packet
        packet = Packet(self.ip, address, payload)
        # Load packet into outbox
        self.outbox.append(packet)
        
    # Application layer receive function, gets from inbox buffer.
    def receive(self):
        # If buffer is not empty, return packet, else return None.
        try:
            message = self.inbox.pop(0)
            self.print_("Found message.")
            return message
        except:
            self.print_("No messages.")
            return None
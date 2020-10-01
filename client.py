#!/usr/bin/python

import secrets

# A single record.
class Record:
    def __init__(self, ip, weights, expiry):
        self.ip = ip
        self.weights = weights
        self.expiry = expiry
    def step(self, s=1):
        self.expiry -= 1
        if self.expiry <= 0:
            return True
        else:
            return False

# Holds and manages records and requests.
class GuestBook:
    def __init__(self):
        self.records = {}
    def encounter(self, ip, weights, expiry):
        self.records[ip] = Record(ip, weights, expiry)
    def step(self):
        # Increment another step and reduce expirations across records.
        for ip in self.records.keys():
            # If incremental step results in expiration, remove record.
            if self.records[ip].step():
                expunged = self.records.pop(ip)
                

class Client:
    def __init__(self, netNode=None, ip=None, neighbor_addrs=None, model=None ):
        if netNode is not None:
            self.net = netNode
        else:
            self.net = DummyNet(ip, neighbor_addrs)
        self.model_message = "hi it's me from " + self.net.ip
        self.model_ready = False
        # Set active flag
        self.active = True
        
        # Model with data
        self.model = model
            
    # Client reporting function
    def print_(self, message):
        # Client level print messaging
        output = "CLIENT::[" + str(self.net.ip) + "]::"
        try:
            output = output + str(message)
        except:
            output = output + "Message not printable."
        print(output)
        
    # Main run process as a state machine.
    def process(self):
        # While active, run
        # Check inbox
        packet = self.net.receive()
        if packet is not None:
            self.model_ready = False
            self.print_("Processing model from " + str(packet.src))
            self.print_("Aggregating model with new input.")
        else:
            # Train the model on local data
            self.print_("Training model.")
            self.model.step()
            self.model_ready = True            
    
    # Lowest level client transmission function.
    def transmit(self, payload, target_addr):
        self.net.send(payload, target_addr)
        
    # Model transfer function.
    def transmit_model(self, recipient):
        # Select recipient
        # Transmit model to recipient
        if self.model_ready:
            self.transmit(self.model_message, recipient)
            self.print_("Transmitting model to " + recipient)
        else:
            self.print_("Model still processing.")
    # Select random recipient.
    def select_random_recv(self):
        return secrets.choice(list(self.net.neighbors.keys()))
#!/usr/bin/python

import secrets
import random
from dummy_net import DummyNet

EXPIRY = 2

# A single record.
class Record:
    def __init__(self, ip, shared_weights, expiry, debug=False):
        self.ip = ip
        self.data_size = shared_weights[0]
        self.weights = shared_weights[1]
        self.expiry = expiry
        self.debug = debug
        # import pdb
        # pdb.set_trace()
    def step(self, s=1):
        self.expiry -= 1
        if self.expiry <= 0:
            return True
        else:
            return False
    def __str__(self):
        return ("\t" + str(self.ip) + " :: EXP" + str(self.expiry) + " :: DAT" + str(self.data_size) + " :: " + \
            str(self.weights[0][0][0]))

# Holds and manages records and requests.
class GuestBook:
    def __init__(self, debug=False):
        self.records = {}
        self.debug = debug
    def encounter(self, ip, shared_weights, expiry):
        self.records[ip] = Record(ip, shared_weights, expiry)
    def step(self):
        # Increment another step and reduce expirations across records.
        for ip in self.records.keys():
            # If incremental step results in expiration, remove record.
            if self.records[ip].step():
                expunged = self.records.pop(ip)
        # If debug, print
        if self.debug:
            print(self)
    def __str__(self):
        if len(self.records.values()):
            return "\n".join([str(record) for record in self.records.values()])
        else:
            return str(None)
                

class Client:
    def __init__(self, netNode=None, ip=None, neighbor_addrs=None, model=None ):
        if netNode is not None:
            self.net = netNode
        else:
            self.net = DummyNet(ip, neighbor_addrs)
        # self.model_message = "hi it's me from " + self.net.ip
        self.model_ready = False
        # Set active flag
        self.active = True
        
        # Model with data
        self.model = model

        # Guestbook
        self.guest_book = GuestBook()
            
    # Client reporting function
    def print_(self, message):
        # Client level print messaging
        output = "CLIENT::[" + str(self.net.ip) + "]::"
        try:
            output = output + str(message)
        except:
            output = output + "Message not printable."
        print(output)
        
    # Main run process as a state machine. Combined aggregate and train step (shouldn't necessarily be used).
    def process(self):
        recvd = self.recv_aggregate_model()
        if not recvd:
            self.train_model()  
        # Step the guest book
        self.guest_book.step()

    # # Single aggregation step
    # def aggregate(self):
    #     recvd = self.recv_aggregate_model()
    #     if not recvd:
    #         self.print_("Nothing to be aggregated.")
    #     self.guest_book.step()

    # # Aggregates all models that need to be aggregated, enabling serial makes it such that each single aggregation is a step
    # def aggregate_full(self, serial=False):
    #     recvd = True
    #     while recvd:
    #         recvd = self.recv_aggregate_model()
    #         if recvd and serial:
    #             self.guest_book.step()
    #     if not serial:
    #         self.guest_book.step()

    def aggregate(self):
        # Check inbox
        packet = self.net.receive()
        self.model_ready = False
        if packet is not None:
            self.print_("Processing model from " + str(packet.src))
            self.guest_book.encounter(packet.src, packet.data, EXPIRY)
            # Get list of (size, model weights)
            self.print_("Guestbook, currently: " + str(self.guest_book))
            self.print_("Aggregating model with new input.")
            size_weight_list = [(record.data_size, record.weights) for record in self.guest_book.records.values()]
            self.model.aggregate(size_weight_list)
        else:
            return False
        self.guest_book.step()

    def aggregate_full(self, serial=False):
        # Check inbox (receive_all if receiving all at once, receive to receive one at a time)
        packet = self.net.receive()
        while packet is not None:
            self.model_ready = False
            self.print_("Processing model from " + str(packet.src))
            self.guest_book.encounter(packet.src, packet.data, EXPIRY)
            if serial:
                self.guest_book.step()
            packet = self.net.receive()
        if not serial:
            self.guest_book.step()
        # Get list of (size, model weights) then AGGREGATE
        self.print_("Guestbook, currently: " + str(self.guest_book))
        self.print_("Aggregating model with new input.")
        size_weight_list = [(record.data_size, record.weights) for record in self.guest_book.records.values()]
        self.model.aggregate(size_weight_list)
        self.guest_book.step()
    
    def train_model(self):
        # Train the model on local data
        self.print_("Training model.")
        self.model.step()
        self.model_ready = True          
    
    # # Aggregation function redirects to model-level aggregation.
    # def aggregate(self):
    #     # Redirect to model-level aggregation
    #     weight_list = [(rec.data_size, rec.weights) for rec in self.guest_book.records]
    #     self.model.aggregate(weight_list)
    #     # Update expiry table

    # Lowest level client transmission function.
    def transmit(self, payload, target_addr):
        self.net.send(payload, target_addr)
        
    # Model transfer function.
    def transmit_model(self, recipient):
        # Select recipient
        # Transmit model to recipient
        # for now always make model_ready True, do not require training after, user can manually do so if they'd like
        self.model_ready = True
        if self.model_ready:
            self.transmit(self.model.sharing_model, recipient)
            self.print_("Transmitting model weights to " + recipient)
        else:
            self.print_("Model still processing.")
    # Select random recipient.
    def select_random_recv(self, num):
        return random.sample(list(self.net.neighbors.keys()), num)
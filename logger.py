#!/usr/bin/python

import pdb

class Log:
    def __init__(self):
        self.entries = []
        self.new_step()

    def add_entry(self, entry):
        self.entries.append(entry)

    def new_step(self):
        # Will carry a list of log entries each
        self.entries.append([])

class Entry:
    def __init__(self, cur_cycle):
        self.cycle = cur_cycle

# Command Entry: [Command String] [Status: OK or ERROR]
class CommandEntry(Entry):
    def __init__(self, cur_cycle, cmd_string, status, involved_nodes=None):
        super().__init__(cur_cycle)
        self.commandString = cmd_string
        self.status = status
        self.involved_nodes = involved_nodes
    def get_entry(self):
        return("COMMAND,%d,%s,%s" % (self.cycle, self.commandString, self.status))
    def __str__(self):
        return("COMMAND\t%d\t%s\t%s" % (self.cycle, self.commandString, self.status))

class ResultEntry(Entry):
    def __init__(self, cur_cycle, node, value_list):
        super().__init__(cur_cycle)
        self.node = node
        self.values = value_list
    def get_entry(self):
        return("RESULT,%d,%s,%s" % (self.cycle, self.node, ','.join(self.values)))
    def __str__(self):
        return("RESULT\t%d\t%s\t%s" % (self.cycle, self.node, '\t'.join(self.values)))
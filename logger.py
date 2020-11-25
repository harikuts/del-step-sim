#!/usr/bin/python

import pdb

class LogError(BaseException):
    pass

class Log:
    def __init__(self):
        # Set list of cycles, each cycle will contain a list of entries
        self.cycles = []
        # Set current counter and take the first step to initalize
        self.cur_counter = -1
        self.new_step()
        # Set entries
        self.NewEntry = None
    # Begin NewEntry, has to be called with the class constructor
    def beginEntry(self, constructor):
        # Build NewEntry, then check to see if it's an actual entry
        try:
            self.NewEntry = constructor(self.current_cycle())
        except:
            raise LogError
    # Commit entry, moves entry from NewEntry to the log, resets NewEntry
    def commitEntry(self):
        # Verify that entry is existent complete
        if self.NewEntry is not None and self.NewEntry.verify_complete():
            self.cycles[self.current_cycle()].append(self.NewEntry)
            self.NewEntry = None
        else:
            raise LogError
    # Result entries don't need a add and commit, they are just directly added
    def commitResult(self, node, results):
        newResult = ResultEntry(self.current_cycle(), node, results)
        self.cycles[self.current_cycle()].append(newResult)
    # New step advances the simulation (for the sake of records)
    def new_step(self):
        # Append a new cycle (empty list of entries) and increment counter
        self.cycles.append([])
        self.cur_counter += 1
    # Returns current cycle
    def current_cycle(self):
        # Because we can't move back in time yet, we have to assert the counter is correct
        try:
            assert self.cur_counter == (len(self.cycles) - 1)
        except:
            raise LogError
        return self.cur_counter
    # Prints entire log
    def printLog(self):
        lines = []
        for cycle in self.cycles:
            for entry in cycle:
                lines.append(str(entry))
        corpus = '\n'.join(lines)
        print(corpus)
    # Get results only
    def getResults(self):
        results = []
        for cycle in self.cycles:
            for entry in cycle:
                if isinstance(entry, ResultEntry):
                    results.append(entry)
        return results
    # Print results only
    def printResults(self):
        results = self.getResults()
        lines = [str(result) for result in results]
        corpus = '\n'.join(lines)
        print(corpus)


class Entry:
    def __init__(self, cur_cycle):
        self.cycle = cur_cycle
    # By default, all entries will be verified. Specific verifications are implemented in the child classes.
    def verify_complete(self):
        return True

# Command Entry: [Command String] [Status: OK or ERROR]
class CommandEntry(Entry):
    # Initialization takes in the current cycle of the Log, other information is optional at the start
    def __init__(self, cur_cycle, cmd_string=None, success=None, involved_nodes=None):
        super().__init__(cur_cycle)
        self.commandString = cmd_string
        self.success = success
        self.involved_nodes = involved_nodes
    # SETTING COMMANDS
    # Sets the command string
    def set_cmd(self, cmd_string):
        self.commandString = cmd_string
    # Sets the nodes involved with the command
    def set_nodes(self, involved_nodes):
        self.involved_nodes = involved_nodes
    # Sets the success of the command, boolean
    def set_success(self, success):
        self.success = success
    # Verify all fields are completed
    def verify_complete(self):
        # Compile a list of required fields and check if they've been completed
        required_fields = [self.commandString, self.success]
        fields_complete = [True if val is not None else False for val in required_fields]
        return True if all(fields_complete) else False
    # REPORTING
    def get_entry(self):
        return("%d,COMMAND,%s,%s" % (self.cycle, ("OK" if self.success else "ER"), self.commandString))
    def __str__(self):
        return("%d\tCOMMAND\t%s\t[%s]" % (self.cycle, ("OK" if self.success else "ER"), self.commandString))

class ResultEntry(Entry):
    def __init__(self, cur_cycle, node, value_list):
        super().__init__(cur_cycle)
        self.node = node
        self.values = value_list
        self.str_values = [str(value) for value in value_list]
    def get_entry(self):
        return("%d,RESULT,%s,%s" % (self.cycle, self.node, ','.join(self.str_values)))
    def __str__(self):
        return("%d\tRESULT\t%s\t%s" % (self.cycle, self.node, '\t'.join(self.str_values)))

# Test the logger
if __name__ == "__main__":
    log = Log()
    log.beginEntry(CommandEntry)
    log.NewEntry.set_cmd("group create group1")
    log.NewEntry.set_nodes(["10.0.0.1", "10.0.0.2", "10.0.0.3"])
    log.NewEntry.set_success(True)
    log.commitEntry()
    pdb.set_trace()
    log.commitEntry()
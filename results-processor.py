import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(dest='filename', help="Include filename here.")
parser.add_argument(dest='outputname', help="Include output filename here.")
args = parser.parse_args()
filename = args.filename
output = args.outputname

# PREPROCESSING
rich_info = []
with open(filename, 'r') as f:
    header = f.readline().strip().split(",")
    for line in f.readlines():
        rich_line = {}
        line = line.strip().split(",")
        for i in range(len(line)):
            rich_line[header[i]] = line[i]
        rich_info.append(rich_line)
index = {}
# Get indices
header.remove("LOSS")
header.remove("ACC")
for title in header:
    indexing = [entry[title] for entry in rich_info]
    indexing = np.unique(indexing)
    if title is "CYCLE":
        indexing = [int(i) for i in indexing]
    index[title] = sorted(indexing)
    print(title, index[title])


# BUILD
# Build hierarchy
hierarchy = ["CYCLE", "NODE", "SCOPE"]
tree = {}
for cycle in index[hierarchy[0]]:
    tree[cycle] = {}
    for node in index[hierarchy[1]]:
        tree[cycle][node] = {}
        for scope in index[hierarchy[2]]:
            tree[cycle][node][scope] = None
for entry in rich_info:
    tree[entry[hierarchy[0]]][entry[hierarchy[1]]][entry[hierarchy[2]]] = entry["ACC"]


# PROCESSING
# Make output file header
output_header = ["cycle",] + index["SCOPE"]
print(output_header)
# Get each entry
total_nodes = float(len(index["NODE"]))
entries = []
for cycle in index["CYCLE"]:
    # Create entry as a list
    entry = [cycle,]
    for scope in index["SCOPE"]:
        # Take average of values
        s = float(0)
        for node in index["NODE"]:
            s += float(tree[cycle][node][scope])
        avg = s / total_nodes
        entry.append(str(avg))
    entries.append(','.join(entry))
# Write file
with open(output, 'w') as o:
    # Combine header and entries
    output_header = ','.join(output_header)
    lines = [output_header,] + entries
    corpus = '\n'.join(lines)
    o.write(corpus)
    
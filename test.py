
import sys

input_file = sys.argv[2]

# env = Tln_env(input_file + ".tln")

fi = open(input_file + ".funct", "r")

lines = fi.readlines()[1:]

lines = [list(map(int, l.strip().split(" "))) for l in lines]

print(lines)

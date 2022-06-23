from palette import palette
import sys

palette_expanded_high = []
palette_expanded_low = []


def get_colour(p):
    if p > 1:
        p = 1

    for i in range(len(palette)):
        if palette[i][0] <= p <= palette[i + 1][0]:
            s0 = palette[i]
            s1 = palette[i + 1]
            pos = (p - s0[0]) / (s1[0] - s0[0])
            rpos = 1 - pos
            r = rpos * s0[1] + pos * s1[0]
            g = rpos * s0[2] + pos * s1[2]
            b = rpos * s0[3] + pos * s1[3]

            return r, g, b


max_iter_high = int(sys.argv[1])
max_iter_low = int(sys.argv[2])

for i in range(max_iter_high):
    palette_expanded_high.append([int(x * 255) for x in get_colour(i / max_iter_high)])

for i in range(max_iter_low):
    palette_expanded_low.append([int(x * 255) for x in get_colour(i / max_iter_low)])


def write_palette(f, palette, max_iter, name):
    f.write(
        f"__device__ int {name}[{max_iter}][3] = {'{'}\n"
    )

    for i in range(max_iter):
        f.write(
            f"{'{'}{palette[i][0]}, {palette[i][1]}, {palette[i][2]}{'}'},\n"
        )
    f.write("};\n\n")


with open("palette.hpp", mode="w") as f:
    f.write("#pragma once\n")
    write_palette(f, palette_expanded_high, max_iter_high, "palette_high")
    write_palette(f, palette_expanded_low, max_iter_low, "palette_low")

import json
import os
import sys

all_rare_words = set()
dirname = sys.argv[1]

with open("local/rareword.all.txt") as fin:
    for line in fin:
        all_rare_words.add(line.strip())

perutt_blist = {}
with open(os.path.join(dirname, "text")) as fin:
    for line in fin:
        uttname = line.split()[0]
        content = ''.join(line.split()[1:]).replace(' ', '')
        print(content)
        perutt_blist[uttname] = []
        # for word in content:
        #     if word in all_rare_words and word not in perutt_blist[uttname]:
        #         perutt_blist[uttname].append(word)
        for bword in all_rare_words:
            if bword in content and bword not in perutt_blist[uttname]:
                perutt_blist[uttname].append(bword)

with open(os.path.join(dirname, "perutt_blist.json"), "w") as fout:
    json.dump(perutt_blist, fout, indent=4)
